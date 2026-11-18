# -*- coding: utf-8 -*-
"""
C16: answers to the reviewer questions the paper is most likely to get (pre-registered in PREREG_C16.md).

  Q-count   "You compare four semantic relations with ONE structural relation."
            -> struct_all: all four structural relations together (a1 parent + a2 siblings +
               a3 previous/next node + a4 tag path), word-deduplicated, cut to 64 tokens,
               plus the same with the node's semantic-relation tokens removed (struct_all_nosem),
               plus BM25 versions cut to the length of the node's own relations (matched).
  Q-a3      "Why does the previous node help (+15.3 on Test)?"
            -> prev vs prev_nosem (previous node with semantic tokens removed).
  Q-all8    "Does structure add anything on top of semantics?"
            -> all8: own semantic relations followed by struct_all, cut to 64 tokens.
  Q-seed    "The other-node control uses one random donor."
            -> own - other for BM25 with three donor seeds (13, 17, 23).

Retrievers: BM25 (relation field added as one BM25F-like field, lambda = 1) and each --dense model
(node text[:64] + " | " + field). Same data, sites and sampling as C12/C14. GT-free ranking.
"""
from __future__ import annotations

import argparse
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
SEEDS = (13, 17, 23)
CONDS = ("text", "own", "other", "struct_all", "struct_all_nosem", "all8", "prev", "prev_nosem")
LABEL = {"text": "text only", "own": "own semantic relations (b1-b4)", "other": "another node's relations",
         "struct_all": "all structural relations (a1-a4)", "struct_all_nosem": "all structural - semantic words",
         "all8": "semantic + structural (b1-b4 + a1-a4)", "prev": "previous node (a3)",
         "prev_nosem": "previous node - semantic words"}


def code_hash():
    h = hashlib.sha256()
    for f in ("c16_run.py", "c12_lib.py", "c12_run.py", "c4_lib.py", "idc_lib.py", "c8_robust.py"):
        p = Path(__file__).with_name(f)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ fields (GT-free)
def tag_path(pg, i):
    names, cur = [], pg.els[i]
    while cur is not None and getattr(cur, "name", None) and cur.name != "[document]":
        names.append(cur.name)
        cur = cur.parent
    return " ".join(reversed(names))


def dedup_words(text):
    seen, out = set(), []
    for w in (text or "").split():
        k = w.lower()
        if k not in seen:
            seen.add(k)
            out.append(w)
    return " ".join(out)


def struct_all(pg, i):
    """a1 parent + a2 siblings + a3 previous and next node + a4 tag path, word-deduplicated."""
    parts = (R.ctx_parent(pg, i), R.ctx_sibling(pg, i), X._prev_node(pg, i), X._next_node(pg, i), tag_path(pg, i))
    return dedup_words(" ".join(p for p in parts if p))


FIELDS = {
    "own": lambda pg, i: C.rel_string(pg, i),
    "struct_all": struct_all,
    "struct_all_nosem": lambda pg, i: X._minus_sem(struct_all(pg, i), pg, i),
    "all8": lambda pg, i: dedup_words(C.rel_string(pg, i) + " " + struct_all(pg, i)),
    "prev": X._prev_node,
    "prev_nosem": lambda pg, i: X._minus_sem(X._prev_node(pg, i), pg, i),
}


def field_string(pg, cond, i, cap=CAP):
    s = FIELDS[cond](pg, i)
    if cap == "matched":
        cap = len(C.rel_string(pg, i).split())
    return " ".join((s or "").split()[:cap])


def field_counter(pg, cond, i, cap):
    cache = pg.__dict__.setdefault(f"_c16_{cond}_{cap}", {})
    if i not in cache:
        t = L.tokenize(field_string(pg, cond, i, cap))
        cache[i] = (Counter(t), len(t))
    return cache[i]


def model_tag(name):
    n = name.split("/")[-1].lower()
    for k in ("bge-small", "bge-base", "bge-large", "e5-small", "e5-base", "e5-large", "gte-base", "gte-large"):
        if k in n:
            return k.replace("-", "_")
    return "dense"


def pick_device(want):
    if want != "auto":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


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
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.role in ("select", "confirm"):
        if not a.split_file:
            sys.exit("--split-file required")
        sites = set(json.loads(Path(a.split_file).read_text(encoding="utf-8"))[a.role])
    else:
        sites = set(a.sites.split(",")) if a.sites else None
    device = pick_device(a.device)
    models = []
    for name in [m.strip() for m in a.dense.split(",") if m.strip()]:
        models.append((model_tag(name), name, Dense(name, device, batch=a.batch)))
        print(f"[dense] {name} -> {model_tag(name)} on {device}")

    rows, _ = S.load_diverse(a.data, sites, a.per_site, a.max_per_page)
    by_page = OrderedDict()
    for r in rows:
        by_page.setdefault(r.page_key, []).append(r)

    per_q = {}
    t0 = time.time()
    dense_conds = ("text", "own", "other", "struct_all", "struct_all_nosem", "all8", "prev", "prev_nosem")
    for n, (pk, qs) in enumerate(by_page.items(), 1):
        pg = L.parse_page(qs[0].html_path.read_text(encoding="utf-8", errors="replace"))
        cands = [i for i in range(len(pg.els)) if pg.key[i]]
        if not cands:
            continue
        emb = {}
        if models:
            donor_p = page_shuffle_donor(cands, f"{a.seed}|{pk}")
            base = {i: X._cap(pg.text[i], CAP) for i in cands}
            docs = {"text": [base[i] for i in cands],
                    "other": [base[i] + " | " + field_string(pg, "own", donor_p[i]) for i in cands]}
            for c in dense_conds:
                if c not in docs:
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
            _, aux = C.rank_all(pg, r.question, seed=a.seed)
            sc, cs, donor = aux["sc"], aux["cs"], aux["shuf_donor"]
            scores = {"bm25_text": sc,
                      "bm25_own": {i: round(sc[i] + C.LAMBDA * cs[i], 9) for i in cands},
                      "bm25_other": {i: round(sc[i] + C.LAMBDA * cs[donor[i]], 9) for i in cands}}
            for c in ("struct_all", "struct_all_nosem", "all8", "prev", "prev_nosem"):
                scores[f"bm25_{c}"] = {i: round(sc[i] + C.LAMBDA * C.bm25_counter(pg, q, field_counter(pg, c, i, CAP)), 9)
                                       for i in cands}
            for c in ("struct_all", "struct_all_nosem"):
                scores[f"bm25_{c}_matched"] = {
                    i: round(sc[i] + C.LAMBDA * C.bm25_counter(pg, q, field_counter(pg, c, i, "matched")), 9) for i in cands}
            for s in SEEDS:                                  # donor-seed robustness of the other-node control
                if s == a.seed:
                    scores[f"bm25_other_s{s}"] = scores["bm25_other"]
                    continue
                dn = C.rank_all(pg, r.question, seed=s)[1]["shuf_donor"]
                scores[f"bm25_other_s{s}"] = {i: round(sc[i] + C.LAMBDA * cs[dn[i]], 9) for i in cands}
            for tag, _, d in models:
                qv = d.enc([r.question], query=True)[0]
                for c, vs in emb[tag].items():
                    scores[f"{tag}_{c}"] = {i: round(float(vs[x] @ qv), 6) for x, i in enumerate(cands)}
            mm = {"_diag": {"page": pk, "len": {c: len(field_string(pg, c, g).split()) for c in FIELDS}}}
            for m, s_ in scores.items():
                mm[m] = R.tie_metrics(s_, cands, g)
            per_q[(r.qid, r.site)] = mm
        del pg
        if n % 50 == 0:
            print(f"  pages {n}/{len(by_page)}  q={len(per_q)}  {time.time() - t0:.0f}s")

    vals = list(per_q.values())
    methods = [m for m in OrderedDict.fromkeys(k for v in vals for k in v) if not m.startswith("_")]
    table = {m: {k: round(C.mean(v[m][k] for v in vals if m in v), 4) for k in ("rand:hit@1", "rand:hit@30")} for m in methods}
    flen = {c: round(C.mean(v["_diag"]["len"][c] for v in vals), 2) for c in FIELDS} if vals else {}

    def T(tid, A, B, kind=""):
        t = C.paired(per_q, A, B, "rand:hit@30")
        if not t:
            return None
        diffs = [(site, mm["_diag"]["page"], mm[A]["rand:hit@30"] - mm[B]["rand:hit@30"])
                 for (qid, site), mm in per_q.items() if A in mm and B in mm]
        _, lo2, hi2 = S.two_level_boot(diffs)
        t.update({"id": tid, "type": kind, "page_ci95": [round(lo2, 4), round(hi2, 4)]})
        return t

    retr = ["bm25"] + [tag for tag, _, _ in models]
    tests = []
    for rt in retr:
        fam = [T(f"K1_{rt}", f"{rt}_own", f"{rt}_struct_all", "PRIMARY K1: own semantic vs ALL structural (relation count)"),
               T(f"K2_{rt}", f"{rt}_struct_all", f"{rt}_struct_all_nosem", "PRIMARY K2: all structural vs all structural - semantic words"),
               T(f"K3_{rt}", f"{rt}_prev", f"{rt}_prev_nosem", "PRIMARY K3: previous node vs previous node - semantic words")]
        fam = [t for t in fam if t]
        for t, adj in zip(fam, R.holm([t["sign_p"] for t in fam])):
            t["holm_p"] = float(f"{adj:.3g}")
        tests += fam
        tests += [T(f"S1_{rt}", f"{rt}_all8", f"{rt}_own", "all eight vs own semantic (does structure add?)"),
                  T(f"S2_{rt}", f"{rt}_struct_all_nosem", f"{rt}_text", "all structural - semantic vs text only"),
                  T(f"S3_{rt}", f"{rt}_struct_all", f"{rt}_text", "all structural vs text only")]
    tests += [T("M1", "bm25_own", "bm25_struct_all_matched", "BM25 own vs all structural, equal tokens"),
              T("M2", "bm25_struct_all_matched", "bm25_struct_all_nosem_matched", "BM25 all structural vs - semantic, equal tokens")]
    tests += [T(f"SEED_{s}", "bm25_own", f"bm25_other_s{s}", f"BM25 own vs other node, donor seed {s}") for s in SEEDS]
    tests = [t for t in tests if t]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = {"experiment": "C16 reviewer questions (PREREG_C16.md)", "role": a.role, "code_hash": code_hash(),
            "rel_version": C.REL_VERSION, "cap": CAP, "dense": [nm for _, nm, _ in models], "device": device,
            "n": len(per_q), "n_pages": len({mm["_diag"]["page"] for mm in vals}),
            "mean_field_tokens_at_gold": flen, "methods": table, "tests": tests}
    (out / "c16_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== C16 [{a.role}] n={len(per_q)} retrievers={retr} hash={summ['code_hash']} ===")
    print("mean field tokens at gold:", json.dumps(flen))
    print(f"\nRecall@30 (Hit@1)\n   {'condition':40s}" + "".join(f"{rt:>18s}" for rt in retr))
    for c in CONDS:
        cells = [f"{table[f'{rt}_{c}']['rand:hit@30']:.3f} ({table[f'{rt}_{c}']['rand:hit@1']:.3f})" if f"{rt}_{c}" in table else "-"
                 for rt in retr]
        print(f"   {LABEL[c]:40s}" + "".join(f"{x:>18s}" for x in cells))
    print("   BM25 equal tokens: " + "  ".join(f"{c}={table[f'bm25_{c}_matched']['rand:hit@30']:.3f}"
                                              for c in ("struct_all", "struct_all_nosem")))
    print("   BM25 other node by donor seed: " + "  ".join(f"s{s}={table[f'bm25_other_s{s}']['rand:hit@30']:.3f}" for s in SEEDS))
    print("\n--- paired tests (site CI, site->page CI) ---")
    for t in tests:
        print(f"{t['id']:12s} {t['A']} - {t['B']} n={t['n']} diff={t['diff'] * 100:+.1f}%p "
              f"CI[{t['ci95'][0] * 100:+.1f},{t['ci95'][1] * 100:+.1f}] pageCI[{t['page_ci95'][0] * 100:+.1f},{t['page_ci95'][1] * 100:+.1f}] "
              f"sites+/-={t['sites_pos_neg'][0]}/{t['sites_pos_neg'][1]}" + (f" holm={t['holm_p']}" if "holm_p" in t else "")
              + f"  | {t['type']}")
    print(f"[saved] {out.resolve()}")


if __name__ == "__main__":
    main()
