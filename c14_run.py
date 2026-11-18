# -*- coding: utf-8 -*-
"""
C14: the seven conditions of Figure 1 in ONE run, for BM25 and one or more dense retrievers
(pre-registered in PREREG_C14.md).

  conditions (field attached to every candidate node; GT-free)
    text        nothing attached                                   (1)
    own         the node's own semantic relations (four rules)     (2)  ours
    other       the same relations of a random other node, same page (3)  control
    qm          the relations of another node that holds question words  (3b) control, secondary
    parent      parent text (own text removed)                     (4)  structural
    siblings    two previous + two next siblings                   (5)  structural
    tagpath     ancestor element names, root -> node              (6)  structural
    parent_nosem  parent text with the node's semantic-relation tokens removed (7)
    prev / prev_nosem  previous node in document order, and the same with semantic tokens removed (extra)

  budget: every field is cut to CAP=64 tokens (primary, as in Table 2).
          BM25 also reports 'matched': each field cut to the token count of the node's own relations.
  retrievers: BM25 (field added as one BM25F field, lambda=1) and each --dense model
          (node text[:64] + " | " + field, cosine to the query).

All ranking functions are GT-free; gold is used for evaluation only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import c12_lib as X
import c4_lib as C
import c8_robust as R
import c9_common as S
import idc_lib as L
from c12_run import Dense, page_shuffle_donor

CAP = 64
CONDS = ("text", "own", "other", "qm", "parent", "siblings", "tagpath", "parent_nosem", "prev", "prev_nosem")
FIELD_CONDS = ("parent", "siblings", "tagpath", "parent_nosem", "prev", "prev_nosem")      # own/other/qm come from rank_all
LABEL = {"text": "(1) text only", "own": "(2) own semantic relations", "other": "(3) another node's relations",
         "qm": "(3b) question-matched other node", "parent": "(4) parent", "siblings": "(5) siblings",
         "tagpath": "(6) tag path", "parent_nosem": "(7) parent - semantic words",
         "prev": "(+) previous node (doc order)", "prev_nosem": "(+) previous node - semantic words"}


def code_hash():
    h = hashlib.sha256()
    for f in ("c14_run.py", "c12_lib.py", "c12_run.py", "c4_lib.py", "idc_lib.py", "c8_robust.py"):
        p = Path(__file__).with_name(f)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ fields (GT-free)
def tag_path(pg, i):
    """Element names from the root to the node (DOM ancestor chain), e.g. 'html body table tr td'."""
    names, cur = [], pg.els[i]
    while cur is not None and getattr(cur, "name", None) and cur.name != "[document]":
        names.append(cur.name)
        cur = cur.parent
    return " ".join(reversed(names))


def parent_nosem(pg, i):
    return X._minus_sem(R.ctx_parent(pg, i), pg, i)


def prev_nosem(pg, i):
    return X._minus_sem(X._prev_node(pg, i), pg, i)


FIELD_FN = {"parent": R.ctx_parent, "siblings": R.ctx_sibling, "tagpath": tag_path, "parent_nosem": parent_nosem,
            "prev": X._prev_node, "prev_nosem": prev_nosem}


def field_string(pg, cond, i, cap=CAP):
    """Raw field string for a condition, cut to `cap` whitespace tokens ('matched' = own-relation length)."""
    if cond == "own":
        s = C.rel_string(pg, i)
    else:
        s = FIELD_FN[cond](pg, i)
    if cap == "matched":
        cap = len(C.rel_string(pg, i).split())
    return " ".join((s or "").split()[:cap])


def field_counter(pg, cond, i, cap):
    cache = pg.__dict__.setdefault(f"_c14_{cond}_{cap}", {})
    if i not in cache:
        t = L.tokenize(field_string(pg, cond, i, cap))
        cache[i] = (Counter(t), len(t))
    return cache[i]


def model_tag(name):
    n = name.split("/")[-1].lower()
    for k in ("bge-small", "bge-base", "bge-large", "e5-small", "e5-base", "e5-large"):
        if k in n:
            return k.replace("-", "_")
    return "dense"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--role", required=True, choices=["dev", "select", "confirm", "test"])
    ap.add_argument("--split-file")
    ap.add_argument("--sites")
    ap.add_argument("--per-site", type=int, default=100)
    ap.add_argument("--max-per-page", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dense", default="", help="comma-separated sentence-transformers models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.role in ("select", "confirm"):
        if not a.split_file:
            sys.exit("--split-file required")
        sites = set(json.loads(Path(a.split_file).read_text(encoding="utf-8"))[a.role])
    else:
        sites = set(a.sites.split(",")) if a.sites else None

    models = []
    for name in [m.strip() for m in a.dense.split(",") if m.strip()]:
        if a.device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                sys.exit("CUDA not available (use --device cpu)")
        d = Dense(name, a.device, batch=a.batch)
        models.append((model_tag(name), name, d))
        print("[dense]", name, "->", model_tag(name))

    rows, _ = S.load_diverse(a.data, sites, a.per_site, a.max_per_page)
    by_page = OrderedDict()
    for r in rows:
        by_page.setdefault(r.page_key, []).append(r)

    per_q, recs = {}, []
    t0 = time.time()
    for n, (pk, qs) in enumerate(by_page.items(), 1):
        pg = L.parse_page(qs[0].html_path.read_text(encoding="utf-8", errors="replace"))
        cands = [i for i in range(len(pg.els)) if pg.key[i]]
        if not cands:
            continue
        # dense document embeddings, one set per model and condition (other = page-level donor, as in C12)
        emb = {}
        if models:
            donor_p = page_shuffle_donor(cands, f"{a.seed}|{pk}")
            base = {i: X._cap(pg.text[i], CAP) for i in cands}
            docs = {"text": [base[i] for i in cands],
                    "own": [base[i] + " | " + field_string(pg, "own", i) for i in cands],
                    "other": [base[i] + " | " + field_string(pg, "own", donor_p[i]) for i in cands]}
            for c in FIELD_CONDS:
                docs[c] = [base[i] + " | " + field_string(pg, c, i) for i in cands]
            uniq = list(OrderedDict.fromkeys(t for v in docs.values() for t in v))
            for tag, _, d in models:
                vec = dict(zip(uniq, d.enc(uniq)))
                emb[tag] = {c: [vec[t] for t in v] for c, v in docs.items()}

        for r in qs:
            g = pg.tid2i.get(r.element_id)
            if g is None or not pg.key[g]:
                continue
            q = L.tokenize(r.question)
            _, aux = C.rank_all(pg, r.question, seed=a.seed)            # GT-free
            sc, cs, donor, qmd = aux["sc"], aux["cs"], aux["shuf_donor"], aux["qm_donor"]
            scores = {"bm25_text": sc,
                      "bm25_own": {i: round(sc[i] + C.LAMBDA * cs[i], 9) for i in cands},
                      "bm25_other": {i: round(sc[i] + C.LAMBDA * cs[donor[i]], 9) for i in cands},
                      "bm25_qm": {i: round(sc[i] + (C.LAMBDA * cs[qmd[i]] if qmd[i] is not None else 0.0), 9)
                                  for i in cands}}
            for c in FIELD_CONDS:
                for cap, suf in ((CAP, ""), ("matched", "_matched")):
                    scores[f"bm25_{c}{suf}"] = {i: round(sc[i] + C.LAMBDA * C.bm25_counter(pg, q, field_counter(pg, c, i, cap)), 9)
                                                for i in cands}
            for tag, _, d in models:
                qv = d.enc([r.question], query=True)[0]
                for c, vs in emb[tag].items():
                    scores[f"{tag}_{c}"] = {i: round(float(vs[x] @ qv), 6) for x, i in enumerate(cands)}

            mm = {"_diag": {"page": pk, "silent": int(sc[g] == 0),
                            "len": {c: len(field_string(pg, c, g).split()) for c in ("own",) + FIELD_CONDS}}}
            for m, s in scores.items():
                tm = R.tie_metrics(s, cands, g)
                mm[m] = tm
                recs.append({"qid": r.qid, "site": r.site, "method": m,
                             **{k: v for k, v in tm.items() if k.startswith("rand:") or k == "tie_size"}})
            per_q[(r.qid, r.site)] = mm
        del pg
        if n % 50 == 0:
            print(f"  pages {n}/{len(by_page)}  q={len(per_q)}  {time.time() - t0:.0f}s")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if recs:
        keys = list(OrderedDict.fromkeys(k for r in recs for k in r))
        with open(out / "c14_rows.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(recs)

    vals = list(per_q.values())
    methods = [m for m in OrderedDict.fromkeys(k for v in vals for k in v) if not m.startswith("_")]
    table = {m: {k: round(C.mean(v[m][k] for v in vals if m in v), 4)
                 for k in ("rand:hit@1", "rand:hit@10", "rand:hit@30", "rand:hit@100", "best:hit@30", "worst:hit@30")
                 if all(k in v[m] for v in vals if m in v)}
             for m in methods}
    flen = {c: round(C.mean(v["_diag"]["len"][c] for v in vals), 2) for c in ("own",) + FIELD_CONDS} if vals else {}

    def T(tid, A, B, metric="rand:hit@30", kind=""):
        t = C.paired(per_q, A, B, metric)
        if not t:
            return None
        diffs = [(site, mm["_diag"]["page"], mm[A][metric] - mm[B][metric])
                 for (qid, site), mm in per_q.items() if A in mm and B in mm]
        _, lo2, hi2 = S.two_level_boot(diffs)
        t.update({"id": tid, "type": kind, "page_ci95": [round(lo2, 4), round(hi2, 4)]})
        return t

    retr = ["bm25"] + [tag for tag, _, _ in models]
    tests = []
    for rt in retr:
        fam = [T(f"P1_{rt}", f"{rt}_own", f"{rt}_parent", kind="PRIMARY P1: own semantic vs parent"),
               T(f"P2_{rt}", f"{rt}_own", f"{rt}_parent_nosem", kind="PRIMARY P2: own semantic vs parent - semantic words"),
               T(f"P3_{rt}", f"{rt}_parent", f"{rt}_parent_nosem", kind="PRIMARY P3: parent vs parent - semantic words"),
               T(f"P4_{rt}", f"{rt}_own", f"{rt}_other", kind="PRIMARY P4: own vs another node's (replication)"),
               T(f"P5_{rt}", f"{rt}_prev", f"{rt}_prev_nosem", kind="PRIMARY P5: previous node vs previous node - semantic words")]
        fam = [t for t in fam if t]
        for t, adj in zip(fam, R.holm([t["sign_p"] for t in fam])):
            t["holm_p"] = float(f"{adj:.3g}")
        tests += fam
        tests += [T(f"S1_{rt}", f"{rt}_own", f"{rt}_siblings", kind="own vs siblings"),
                  T(f"S2_{rt}", f"{rt}_own", f"{rt}_tagpath", kind="own vs tag path"),
                  T(f"S3_{rt}", f"{rt}_tagpath", f"{rt}_text", kind="tag path vs text only"),
                  T(f"S4_{rt}", f"{rt}_parent_nosem", f"{rt}_text", kind="parent - semantic words vs text only"),
                  T(f"S5_{rt}", f"{rt}_own", f"{rt}_prev", kind="own vs previous node")]
    tests += [T("M1", "bm25_own", "bm25_parent_matched", kind="BM25 own vs parent, equal tokens"),
              T("M2", "bm25_own", "bm25_parent_nosem_matched", kind="BM25 own vs parent - semantic, equal tokens"),
              T("Q1", "bm25_own", "bm25_qm", kind="BM25 own vs question-matched other node")]
    tests = [t for t in tests if t]

    summ = {"experiment": "C14 seven conditions x retrievers (PREREG_C14.md)", "role": a.role, "code_hash": code_hash(),
            "rel_version": C.REL_VERSION, "cap": CAP, "dense": [name for _, name, _ in models], "n": len(per_q),
            "n_pages": len({mm["_diag"]["page"] for mm in per_q.values()}), "sites": sorted({k[1] for k in per_q}),
            "silent_share": round(C.mean(v["_diag"]["silent"] for v in vals), 4) if vals else None,
            "mean_field_tokens_at_gold": flen, "methods": table, "tests": tests}
    (out / "c14_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== C14 [{a.role}] n={len(per_q)} pages={summ['n_pages']} retrievers={retr} hash={summ['code_hash']} ===")
    print("mean field tokens at gold:", json.dumps(flen))
    print(f"\nRecall@30 (expected under random ties); Hit@1 in brackets")
    print(f"   {'condition':34s}" + "".join(f"{rt:>20s}" for rt in retr))
    for c in CONDS:
        cells = []
        for rt in retr:
            v = table.get(f"{rt}_{c}")
            cells.append(f"{v['rand:hit@30']:.3f} [{v['rand:hit@1']:.3f}]" if v else "-")
        print(f"   {LABEL[c]:34s}" + "".join(f"{x:>20s}" for x in cells))
    print("   BM25, equal tokens (matched):  " + "  ".join(
        f"{c}={table[f'bm25_{c}_matched']['rand:hit@30']:.3f}" for c in FIELD_CONDS if f"bm25_{c}_matched" in table))
    print("\n--- paired tests (site CI, site->page CI) ---")
    for t in tests:
        print(f"{t['id']:14s} {t['A']} - {t['B']} n={t['n']} diff={t['diff'] * 100:+.1f}%p "
              f"CI[{t['ci95'][0] * 100:+.1f},{t['ci95'][1] * 100:+.1f}] pageCI[{t['page_ci95'][0] * 100:+.1f},{t['page_ci95'][1] * 100:+.1f}] "
              f"sites+/-={t['sites_pos_neg'][0]}/{t['sites_pos_neg'][1]}" + (f" holm={t['holm_p']}" if "holm_p" in t else "")
              + f"  | {t['type']}")
    print(f"[saved] {out.resolve()}")


if __name__ == "__main__":
    main()
