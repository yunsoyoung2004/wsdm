# -*- coding: utf-8 -*-
"""
C12: grounding and limits of the four relation rules (pre-registered in PREREG_C12.md).

  A  where do question-answer bridge words sit relative to the gold node? (descriptive, uses gold)
  B  unused description relations (caption, dt/dd, label[for], figcaption/legend, attributes)
  C  which of the four rules matters (leave-one-out, single rule, +parent)
  D  dense retriever (--dense): is the own-relation effect specific to BM25?
  E  header-word copying (--perturb): replace relation-vocabulary words in the question by
     WordNet synonyms that do not occur on the page (GT-free), then re-rank
  G  chunk baseline: 64-word windows of page text vs node packing, same 64-word budget

All ranking functions are GT-free; gold is used for evaluation and for audit A only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import c12_lib as X
import c4_lib as C
import c8_robust as R
import c9_common as S
import c9_qa as Q
import idc_lib as L

BUDGET = 64


def code_hash():
    h = hashlib.sha256()
    for f in ("c12_lib.py", "c12_run.py", "c4_lib.py", "idc_lib.py", "c8_robust.py", "c9_qa.py"):
        p = Path(__file__).with_name(f)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ dense encoder
class Dense:
    QUERY_PREFIX = {"bge": "Represent this sentence for searching relevant passages: ", "e5": "query: "}
    DOC_PREFIX = {"e5": "passage: "}

    def __init__(self, name, device, max_len=128, batch=128):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name, device=device)
        self.m.max_seq_length = max_len
        self.batch = batch
        fam = "bge" if "bge" in name.lower() else ("e5" if "e5" in name.lower() else "")
        self.qp, self.dp = self.QUERY_PREFIX.get(fam, ""), self.DOC_PREFIX.get(fam, "")

    def enc(self, texts, query=False):
        pre = self.qp if query else self.dp
        return self.m.encode([pre + (t or ".") for t in texts], batch_size=self.batch,
                             normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def page_shuffle_donor(cands, seed):
    rng = random.Random(seed)
    out = {}
    for i in cands:
        j = rng.choice(cands) if len(cands) > 1 else i
        while j == i and len(cands) > 1:
            j = rng.choice(cands)
        out[i] = j
    return out


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
    ap.add_argument("--dense", default="", help="sentence-transformers model, e.g. BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--perturb", action="store_true", help="E: WordNet synonym perturbation (needs nltk wordnet)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.role in ("select", "confirm"):
        if not a.split_file:
            sys.exit("--split-file required")
        sites = set(json.loads(Path(a.split_file).read_text(encoding="utf-8"))[a.role])
    else:
        sites = set(a.sites.split(",")) if a.sites else None

    dense = None
    if a.dense:
        if a.device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                sys.exit("CUDA not available (use --device cpu)")
            print("[gpu]", torch.cuda.get_device_name(0))
        dense = Dense(a.dense, a.device)
        print("[dense]", a.dense)
    syn = X.wordnet_synonyms() if a.perturb else None

    rows, _ = S.load_diverse(a.data, sites, a.per_site, a.max_per_page)
    by_page = OrderedDict()
    for r in rows:
        by_page.setdefault(r.page_key, []).append(r)

    per_q, recs, audit_rows, pert_rows = {}, [], [], []
    t0 = time.time()
    for n, (pk, qs) in enumerate(by_page.items(), 1):
        pg = L.parse_page(qs[0].html_path.read_text(encoding="utf-8", errors="replace"))
        cands = [i for i in range(len(pg.els)) if pg.key[i]]
        if not cands:
            continue
        ptoks = set(X.toks(X.page_text(pg)))
        emb = {}
        if dense is not None:
            donor_p = page_shuffle_donor(cands, f"{a.seed}|{pk}")
            texts = {
                "text": [X._cap(pg.text[i], 64) for i in cands],
                "rel": [X._cap(pg.text[i], 64) + " | " + C.rel_string(pg, i) for i in cands],
                "shuf": [X._cap(pg.text[i], 64) + " | " + C.rel_string(pg, donor_p[i]) for i in cands],
                "parent": [X._cap(pg.text[i], 64) + " | " + " ".join(R.ctx_parent(pg, i).split()[:len(C.rel_string(pg, i).split())])
                           for i in cands],
            }
            uniq = list(OrderedDict.fromkeys(t for v in texts.values() for t in v))
            vec = dict(zip(uniq, dense.enc(uniq)))
            emb = {k: [vec[t] for t in v] for k, v in texts.items()}
        for r in qs:
            g = pg.tid2i.get(r.element_id)
            if g is None or not pg.key[g]:
                continue
            q = L.tokenize(r.question)
            _, aux = C.rank_all(pg, r.question, seed=a.seed)        # GT-free
            sc, cs, donor = aux["sc"], aux["cs"], aux["shuf_donor"]
            scores = {"bm25": sc,
                      "fusion_rel": {i: round(sc[i] + C.LAMBDA * cs[i], 9) for i in cands},
                      "fusion_shuf": {i: round(sc[i] + C.LAMBDA * cs[donor[i]], 9) for i in cands}}
            for name, fn in X.ABLATIONS.items():
                if name == "rel_all":
                    continue
                scores[f"fusion_{name}"] = {i: round(sc[i] + C.LAMBDA * C.bm25_counter(pg, q, X.counter_for(pg, name, i, fn)), 9)
                                            for i in cands}
            # H: each relation type alone, own vs another node's (same donor as fusion_shuf), equal tokens
            has = {}
            for name in X.TYPES:
                cnt = {i: C.bm25_counter(pg, q, X.counter_for(pg, f"type_{name}", i, lambda p, j, nm=name: X.type_text(p, nm, j)))
                       for i in cands}
                scores[f"t_{name}"] = {i: round(sc[i] + C.LAMBDA * cnt[i], 9) for i in cands}
                scores[f"t_{name}_shuf"] = {i: round(sc[i] + C.LAMBDA * cnt[donor[i]], 9) for i in cands}
                has[name] = int(bool(X.type_text(pg, name, g).strip()))
            # how much a semantic relation is contained in the structural relation that locates it (gold node)
            def _ov(sem, struct):
                a, b = set(X.toks(X.type_text(pg, sem, g))), set(X.toks(X.type_text(pg, struct, g)))
                return round(len(a & b) / len(a), 4) if a else None
            overlap = {"label_in_prev_sibling": _ov("m_prev_label", "s_prev_sibling"),
                       "row_header_in_parent": _ov("m_row_header", "s_parent"),
                       "heading_in_prev_in_order": _ov("m_heading", "s_prev_in_order")}
            qvec = None
            if dense is not None:
                qvec = dense.enc([r.question], query=True)[0]
                for k in ("text", "rel", "shuf", "parent"):
                    scores[f"dense_{k}"] = {i: round(float(emb[k][x] @ qvec), 6) for x, i in enumerate(cands)}
            if syn is not None:
                qp, nrep = X.perturb(r.question, X.page_relation_vocab(pg, cands), ptoks, syn)
                qpt = L.tokenize(qp)
                scp = pg.bm25.scores(qpt)
                csp = {i: C.bm25_counter(pg, qpt, C.rel_tokens(pg, i)) for i in cands}
                scores["p_bm25"] = scp
                scores["p_fusion_rel"] = {i: round(scp[i] + C.LAMBDA * csp[i], 9) for i in cands}
                scores["p_fusion_shuf"] = {i: round(scp[i] + C.LAMBDA * csp[donor[i]], 9) for i in cands}
                if dense is not None:
                    pv = dense.enc([qp], query=True)[0]
                    for k in ("text", "rel", "shuf"):
                        scores[f"p_dense_{k}"] = {i: round(float(emb[k][x] @ pv), 6) for x, i in enumerate(cands)}
                pert_rows.append({"qid": r.qid, "site": r.site, "question": r.question, "perturbed": qp, "n_replaced": nrep})

            mm = {"_diag": {"page": pk, "silent": int(sc[g] == 0), "bridge": int(cs[g] > 0),
                            **{f"has_{k}": v for k, v in has.items()}, "overlap": overlap}}
            for m, s in scores.items():
                tm = R.tie_metrics(s, cands, g)
                mm[m] = tm
                recs.append({"qid": r.qid, "site": r.site, "method": m,
                             **{k: v for k, v in tm.items() if k.startswith("rand:") or k == "tie_size"}})
            if syn is not None:
                mm["_diag"]["perturbed"] = int(pert_rows[-1]["n_replaced"] > 0)

            # G: chunk baseline vs node packing, same word budget (span answers only)
            if r.answer.strip().lower() not in ("yes", "no"):
                ans = Q.normalize(r.answer)
                ctxs = {m: Q.pack(pg, Q.rand_order(scores[m], cands, f"{a.seed}|{r.qid}"), BUDGET)[1]
                        for m in ("bm25", "fusion_rel", "fusion_shuf") + (("dense_rel",) if dense is not None else ())}
                ctxs["chunk"] = X.chunk_top(pg, r.question, BUDGET, seed=f"{a.seed}|{r.qid}")
                for m, text in ctxs.items():
                    mm[f"G_{m}"] = {"ans": int(bool(ans) and ans in Q.normalize(text))}

            ar = X.audit(pg, g, r.question)
            pres = ar.pop("present")
            audit_rows.append({"qid": r.qid, "site": r.site, "silent": mm["_diag"]["silent"], **ar,
                               **{f"present_{k}": v for k, v in pres.items()}})
            per_q[(r.qid, r.site)] = mm
        del pg
        if n % 50 == 0:
            print(f"  pages {n}/{len(by_page)}  q={len(per_q)}  {time.time() - t0:.0f}s")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def dump(name, rs):
        if not rs:
            return
        keys = list(OrderedDict.fromkeys(k for r in rs for k in r))
        with open(out / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rs)
    dump("c12_rows.csv", recs)
    dump("c12_audit.csv", audit_rows)
    dump("c12_perturb.csv", pert_rows)

    vals = list(per_q.values())
    methods = [m for m in OrderedDict.fromkeys(k for v in vals for k in v) if not m.startswith(("_", "G_"))]
    table = {m: {k: round(C.mean(v[m][k] for v in vals if m in v), 4) for k in ("rand:hit@1", "rand:hit@10", "rand:hit@30", "rand:hit@100")}
             for m in methods}
    gtable = {m: round(C.mean(v[m]["ans"] for v in vals if m in v), 4)
              for m in OrderedDict.fromkeys(k for v in vals for k in v) if m.startswith("G_")}

    # ---- A: audit summary
    def frac(key, keep=lambda r: True):
        rs = [r for r in audit_rows if keep(r)]
        return round(C.mean(r[key] for r in rs), 4) if rs else None
    has_bt = lambda r: r["n_bridge_tokens"] > 0
    audit = {"n": len(audit_rows),
             "has_bridge_tokens": round(C.mean(int(has_bt(r)) for r in audit_rows), 4) if audit_rows else None,
             "among_questions_with_bridge_tokens": {f"in_{c}": frac(f"in_{c}", has_bt) for c in X.AUDIT_CATS + ("rules",)},
             "nowhere_on_page": frac("nowhere", has_bt),
             "rules_missed_but_found_in": {c: frac(f"adds_{c}", has_bt) for c in X.EXTRA_CATS},
             "present_at_gold": {k: frac(f"present_{k}") for k in ("caption", "dtdd", "label_for", "figlegend", "attrs_ext")},
             "silent_only": {f"in_{c}": frac(f"in_{c}", lambda r: has_bt(r) and r["silent"]) for c in X.AUDIT_CATS + ("rules",)}}

    # ---- tests
    def T(tid, A, B, metric="rand:hit@30", keep=lambda m: True, kind=""):
        t = C.paired(per_q, A, B, metric, keep)
        if not t:
            return None
        diffs = [(site, mm["_diag"]["page"], mm[A][metric] - mm[B][metric])
                 for (qid, site), mm in per_q.items() if A in mm and B in mm and keep(mm)]
        _, lo2, hi2 = S.two_level_boot(diffs)
        t.update({"id": tid, "type": kind, "page_ci95": [round(lo2, 4), round(hi2, 4)]})
        return t

    tests = []
    c1 = [T(f"C-1_{p}", "fusion_rel", f"fusion_only_{p}", kind="PRIMARY C-1: all rules vs single rule") for p in C.PARTS]
    c1 = [t for t in c1 if t]
    for t, adj in zip(c1, R.holm([t["sign_p"] for t in c1])):
        t["holm_p"] = float(f"{adj:.3g}")
    tests += c1
    tests += [T(f"C-2_{p}", "fusion_rel", f"fusion_minus_{p}", kind="C-2: loss when removing one rule") for p in C.PARTS]
    tests += [T("C-3", "fusion_rel_parent", "fusion_rel", kind="relations + parent vs relations"),
              T("B-2", "fusion_ext", "fusion_rel", kind="extended description relations vs relations"),
              T("REF", "fusion_rel", "fusion_shuf", kind="reference: own vs shuffled (C8 R1)")]
    if dense is not None:
        tests += [T("D-1", "dense_rel", "dense_shuf", kind="PRIMARY D-1: dense own vs shuffled relations"),
                  T("D-2", "dense_text", "bm25", kind="D-2: dense text-only vs BM25"),
                  T("D-2s", "dense_text", "bm25", keep=lambda m: m["_diag"]["silent"] == 1, kind="D-2 on silent questions"),
                  T("D-4", "dense_rel", "dense_text", kind="dense own relations vs dense text"),
                  T("D-5", "dense_rel", "dense_parent", kind="dense own relations vs dense parent (equal tokens)"),
                  T("D-6", "fusion_rel", "dense_text", kind="BM25+relations vs dense text-only")]
    if syn is not None:
        tests += [T("E-2", "p_fusion_rel", "p_fusion_shuf", kind="PRIMARY E-2: perturbed question, BM25 own vs shuffled"),
                  T("E-2p", "p_fusion_rel", "p_fusion_shuf", keep=lambda m: m["_diag"].get("perturbed") == 1,
                    kind="E-2 restricted to questions actually perturbed")]
        if dense is not None:
            tests += [T("E-3", "p_dense_rel", "p_dense_shuf", kind="perturbed question, dense own vs shuffled")]
    types_h = {}
    for name in X.TYPES:
        A, B = f"t_{name}", f"t_{name}_shuf"
        present = lambda m, nm=name: m["_diag"][f"has_{nm}"] == 1
        t_all = T(f"H_{name}", A, B, kind="H: one relation type alone, own vs another node's (all questions)")
        t_pre = T(f"H_{name}|present", A, B, keep=present, kind="H: same, only where the gold node has this relation")
        tests += [t for t in (t_all, t_pre) if t]
        vs = [v for v in vals if A in v]
        types_h[name] = {"family": "structural (DOM)" if name.startswith("s_") else "semantic (HTML)",
                         "present_at_gold": round(C.mean(v["_diag"][f"has_{name}"] for v in vs), 4) if vs else None,
                         "R@30_own": table.get(A, {}).get("rand:hit@30"), "R@30_other": table.get(B, {}).get("rand:hit@30"),
                         "own_minus_other": t_all["diff"] if t_all else None,
                         "ci95": t_all["ci95"] if t_all else None,
                         "own_minus_other_if_present": t_pre["diff"] if t_pre else None,
                         "ci95_if_present": t_pre["ci95"] if t_pre else None}
    overlap_h = {}
    for k in ("label_in_prev_sibling", "row_header_in_parent", "heading_in_prev_in_order"):
        xs = [v["_diag"]["overlap"][k] for v in vals if v["_diag"]["overlap"][k] is not None]
        overlap_h[k] = {"n": len(xs), "mean_token_share": round(C.mean(xs), 4) if xs else None,
                        "share_fully_contained": round(C.mean(int(x >= 0.999) for x in xs), 4) if xs else None}
    tests += [T("G-1", "G_fusion_rel", "G_chunk", metric="ans", kind="PRIMARY G-1: node+relations packing vs top chunk (ans in ctx, 64 words)"),
              T("G-1b", "G_bm25", "G_chunk", metric="ans", kind="BM25 node packing vs top chunk")]
    tests = [t for t in tests if t]

    pert = None
    if syn is not None and pert_rows:
        pert = {"n": len(pert_rows), "share_perturbed": round(C.mean(int(r["n_replaced"] > 0) for r in pert_rows), 4),
                "mean_replaced": round(C.mean(r["n_replaced"] for r in pert_rows), 3)}
    summ = {"experiment": "C12 grounding and limits (PREREG_C12.md)", "role": a.role, "code_hash": code_hash(),
            "rel_version": C.REL_VERSION, "dense": a.dense or None, "perturb": bool(syn), "n": len(per_q),
            "n_pages": len({mm["_diag"]["page"] for mm in per_q.values()}), "sites": sorted({k[1] for k in per_q}),
            "audit_A": audit, "types_H": types_h, "overlap_H": overlap_h, "methods_rand": table, "chunk_G_ans_in_ctx": gtable, "perturb_E": pert, "tests": tests}
    (out / "c12_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== C12 [{a.role}] n={len(per_q)} pages={summ['n_pages']} dense={a.dense or '-'} perturb={bool(syn)} hash={summ['code_hash']} ===")
    print("\nA  where bridge words sit (share of questions that have bridge words)")
    for k, v in audit["among_questions_with_bridge_tokens"].items():
        print(f"   {k:16s} {v}")
    print(f"   nowhere on page  {audit['nowhere_on_page']}")
    print("   rules missed, found in:", json.dumps(audit["rules_missed_but_found_in"]))
    print("   present at gold:", json.dumps(audit["present_at_gold"]))
    print(f"\nH  one relation type at a time ({X.TYPE_CAP} tokens), Recall@30: own vs another node's")
    print(f"   {'type':18s}{'family':18s}{'present':>8s}{'own':>7s}{'other':>7s}{'diff':>8s}   CI          {'diff|present':>12s}")
    for name, h in types_h.items():
        ci = h["ci95"] or [float("nan")] * 2
        dp = h["own_minus_other_if_present"]
        print(f"   {name:18s}{h['family']:18s}{h['present_at_gold'] or 0:8.3f}{h['R@30_own'] or 0:7.3f}{h['R@30_other'] or 0:7.3f}"
              f"{(h['own_minus_other'] or 0) * 100:+7.1f}p [{ci[0] * 100:+.1f},{ci[1] * 100:+.1f}]"
              + (f"{dp * 100:+11.1f}p" if dp is not None else "         n/a"))
    print("   overlap (gold node, share of semantic tokens inside the structural relation):", json.dumps(overlap_h))
    print("\nRecall@30 / Hit@1 (expected under random ties)")
    for m, v in table.items():
        if m.startswith("t_"):
            continue
        print(f"   {m:26s} R@30 {v['rand:hit@30']:.3f}   Hit@1 {v['rand:hit@1']:.3f}")
    print("\nG  answer string in 64-word context:", json.dumps(gtable))
    if pert:
        print("E  perturbation:", json.dumps(pert))
    print("\n--- paired tests (site CI, site->page CI) ---")
    for t in tests:
        print(f"{t['id']:8s} {t['A']} - {t['B']} [{t['metric']}] n={t['n']} diff={t['diff'] * 100:+.1f}%p "
              f"CI[{t['ci95'][0] * 100:+.1f},{t['ci95'][1] * 100:+.1f}] pageCI[{t['page_ci95'][0] * 100:+.1f},{t['page_ci95'][1] * 100:+.1f}] "
              f"sites+/-={t['sites_pos_neg'][0]}/{t['sites_pos_neg'][1]}" + (f" holm={t['holm_p']}" if "holm_p" in t else "")
              + f"  | {t['type']}")
    print(f"[saved] {out.resolve()}")


if __name__ == "__main__":
    main()
