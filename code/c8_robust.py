# -*- coding: utf-8 -*-
"""
C8: secondary analyses that close reviewer holes (pre-registered in PREREG_C8.md BEFORE running).

E3  Robustness of the relational-bridge result (H1)
    - tie handling: doc-order (as reported), EXPECTED under uniform random tie-break (analytic),
      and best/worst-case bounds. Conservative H1 = fusion(worst ties) - shuffled(best ties).
    - K curve (1..100) and reader token cost
    - lambda sensitivity (TRAIN sites only)
E1  Is it just "context expansion"? Same fusion, different context, BM25:
      parent text, neighbouring siblings, tag/attribute path, relations (v1), shuffled relations
    at two budgets: cap64 (every context up to 64 tokens) and matched (each node's context
    truncated to that node's relation length -> identical token count per node).
E4  Coverage: how many official questions are span questions with an evidence node.

Gold is used only for evaluation; scoring functions never see it.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import c4_lib as C
import c9_common as S
import idc_lib as L

KS = (1, 5, 10, 20, 30, 50, 100)
CAP = 64
CTX_TYPES = ("parent", "sibling", "xpath", "rel")
ATTRS = ("class", "id", "itemprop", "name", "property", "data-label")


# ------------------------------------------------------------------ contexts (GT-free)
def ctx_parent(pg, i):
    p = pg.parent[i]
    if p < 0:
        return ""
    own = pg.text[i]
    return pg.text[p].replace(own, " ", 1) if own else pg.text[p]


def ctx_sibling(pg, i):
    el = pg.els[i]
    prev = [L._t(x) for x in el.find_previous_siblings(limit=2)]
    nxt = [L._t(x) for x in el.find_next_siblings(limit=2)]
    return " ".join(t for t in reversed(prev) if t) + " " + " ".join(t for t in nxt if t)


def ctx_xpath(pg, i):
    parts, cur = [], pg.els[i]
    for _ in range(5):
        if cur is None or not getattr(cur, "name", None) or cur.name in ("[document]",):
            break
        bits = [cur.name]
        for a in ATTRS:
            v = cur.get(a) if hasattr(cur, "get") else None
            if v:
                bits.append(" ".join(v) if isinstance(v, list) else str(v))
        parts.append(" ".join(bits))
        cur = cur.parent
    return " ".join(reversed(parts))


CTX_FN = {"parent": ctx_parent, "sibling": ctx_sibling, "xpath": ctx_xpath, "rel": lambda pg, i: C.rel_string(pg, i)}


def ctx_counter(pg, kind, i, cap):
    key = f"_c8_{kind}_{cap}"
    cache = pg.__dict__.setdefault(key, {})
    if i not in cache:
        toks = CTX_FN[kind](pg, i).split()
        if cap == "matched":
            n = len(C.rel_string(pg, i).split())
            toks = toks[:n]
        else:
            toks = toks[:cap]
        t = L.tokenize(" ".join(toks))
        cache[i] = (Counter(t), len(t))
    return cache[i]


# ------------------------------------------------------------------ tie-aware evaluation
def gold_rank_stats(score, cands, g):
    sg = score[g]
    greater = sum(1 for i in cands if score[i] > sg)
    equal = sum(1 for i in cands if score[i] == sg)
    before = sum(1 for i in cands if score[i] == sg and i < g)   # doc-order tie break (index)
    return greater, equal, before


def tie_metrics(score, cands, g):
    gr, eq, bf = gold_rank_stats(score, cands, g)
    out = {}
    for k in KS:
        out[f"doc:hit@{k}"] = int(gr + bf + 1 <= k)
        out[f"best:hit@{k}"] = int(gr + 1 <= k)
        out[f"worst:hit@{k}"] = int(gr + eq <= k)
        out[f"rand:hit@{k}"] = min(1.0, max(0.0, (k - gr) / eq))
    out["tie_size"] = eq
    return out


# ------------------------------------------------------------------ coverage (E4)
def coverage(data, sites):
    rows = []
    for p in Path(data).rglob("*.csv"):
        try:
            with open(p, encoding="utf-8-sig", newline="") as f:
                rd = csv.DictReader(f)
                cm = L._colmap(rd.fieldnames or [])
                if not ({"question", "element_id", "answer", "qid"} <= set(cm)):
                    continue
                per = Counter()
                site = None
                for r in rd:
                    qid = str(r[cm["qid"]]).strip()
                    site = L.site_from(qid, p)
                    if sites and site not in sites:
                        break
                    per["total"] += 1
                    try:
                        eid = int(float(r[cm["element_id"]]))
                    except ValueError:
                        eid = -1
                    ans = str(r[cm["answer"]]).strip().lower()
                    per["no_node"] += int(eid < 0)
                    per["yes_no_answer"] += int(ans in ("yes", "no"))
                    per["span_with_node"] += int(eid >= 0)
                if site and (not sites or site in sites) and per["total"]:
                    rows.append({"site": site, **per})
        except Exception:
            continue
    tot = Counter()
    for r in rows:
        tot.update({k: v for k, v in r.items() if k != "site"})
    return rows, dict(tot)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, m, run = [0.0] * len(ps), len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--role", required=True, choices=["dev", "train", "select", "confirm"])
    ap.add_argument("--split-file")
    ap.add_argument("--sites")
    ap.add_argument("--max-per-site", type=int, default=100)
    ap.add_argument("--lambdas", default="0.25,0.5,1,2,4")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    ap.add_argument("--diverse", action="store_true", help="random pages, <= --max-per-page questions per page")
    ap.add_argument("--max-per-page", type=int, default=3)
    a = ap.parse_args()
    if a.role in ("train", "select", "confirm"):
        if not a.split_file:
            sys.exit("--split-file required")
        sites = set(json.loads(Path(a.split_file).read_text(encoding="utf-8"))[a.role])
    else:
        sites = set(a.sites.split(",")) if a.sites else None
    do_lambda = a.role == "train"
    lambdas = [float(x) for x in a.lambdas.split(",")]

    if a.diverse:
        rows, skipped = S.load_diverse(a.data, sites, a.max_per_site, a.max_per_page)
    else:
        rows, skipped, _ = L.load_rows(Path(a.data), sites, None, a.max_per_site)
    by_page = OrderedDict()
    for r in rows:
        by_page.setdefault(r.page_key, []).append(r)

    per_q, recs = {}, []
    for n, (pk, qs) in enumerate(by_page.items(), 1):
        pg = L.parse_page(qs[0].html_path.read_text(encoding="utf-8", errors="replace"))
        cands = [i for i in range(len(pg.els)) if pg.key[i]]
        for r in qs:
            g = pg.tid2i.get(r.element_id)
            if g is None or not pg.key[g]:
                continue
            q = L.tokenize(r.question)
            ranked, aux = C.rank_all(pg, r.question, seed=a.seed)  # no gold; gives sc, cs, shuf donor
            sc, cs, donor = aux["sc"], aux["cs"], aux["shuf_donor"]
            scores = {"bm25": sc,
                      "fusion_rel": {i: round(sc[i] + C.LAMBDA * cs[i], 9) for i in cands},
                      "fusion_shuf": {i: round(sc[i] + C.LAMBDA * cs[donor[i]], 9) for i in cands}}
            for kind in ("parent", "sibling", "xpath"):
                for cap in (CAP, "matched"):
                    scores[f"fusion_{kind}_{cap}"] = {
                        i: round(sc[i] + C.LAMBDA * C.bm25_counter(pg, q, ctx_counter(pg, kind, i, cap)), 9) for i in cands}
            if do_lambda:
                for lam in lambdas:
                    scores[f"lam{lam}_rel"] = {i: round(sc[i] + lam * cs[i], 9) for i in cands}
                    scores[f"lam{lam}_shuf"] = {i: round(sc[i] + lam * cs[donor[i]], 9) for i in cands}
            mm = {"_diag": {"bridge": int(cs[g] > 0), "silent": int(sc[g] == 0), "page": r.page_key}}
            for m, s in scores.items():
                tm = tie_metrics(s, cands, g)
                mm[m] = tm
                recs.append({"qid": r.qid, "site": r.site, "method": m, **tm})
            # reader cost: tokens of the top-k node texts (doc-order ranking)
            for m in ("bm25", "fusion_rel"):
                order = sorted(cands, key=lambda i: (-scores[m][i], i))
                mm[m].update({f"tokens@{k}": sum(len(pg.toks[i]) for i in order[:k]) for k in (10, 30, 100)})
            per_q[(r.qid, r.site)] = mm
        del pg
        if n % 100 == 0:
            print(f"  pages {n}/{len(by_page)}  q={len(per_q)}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(OrderedDict.fromkeys(k for r in recs for k in r))
    with open(out / "c8_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)

    methods = list(OrderedDict.fromkeys(r["method"] for r in recs))
    vals = list(per_q.values())
    table = {m: {k: round(C.mean(v[m][k] for v in vals), 4)
                 for k in vals[0][m] if k.startswith(("doc:", "rand:", "best:", "worst:", "tokens@"))}
             for m in methods}

    def T(tid, A, B, metric, keep=lambda m: True, kind="secondary"):
        t = C.paired(per_q, A, B, metric, keep)
        if t:
            t.update({"id": tid, "type": kind})
        return t

    tests = [
        T("R0", "fusion_rel", "fusion_shuf", "doc:hit@30", kind="replicates H1 (doc-order ties)"),
        T("R1", "fusion_rel", "fusion_shuf", "rand:hit@30", kind="H1 under random tie-break (expected)"),
        # conservative: fusion's ties broken AGAINST gold vs shuffled's ties broken FOR gold
        None,
        T("R3_10", "fusion_rel", "bm25", "rand:hit@10"), T("R3_30", "fusion_rel", "bm25", "rand:hit@30"),
        T("R3_50", "fusion_rel", "bm25", "rand:hit@50"), T("R3_100", "fusion_rel", "bm25", "rand:hit@100"),
        T("B1", "fusion_rel", f"fusion_parent_{CAP}", "rand:hit@30", kind="relations vs parent text (cap64)"),
        T("B2", "fusion_rel", f"fusion_sibling_{CAP}", "rand:hit@30", kind="relations vs siblings (cap64)"),
        T("B3", "fusion_rel", f"fusion_xpath_{CAP}", "rand:hit@30", kind="relations vs tag/attr path (cap64)"),
        T("B4", "fusion_rel", "fusion_parent_matched", "rand:hit@30", kind="relations vs parent (equal tokens)"),
        T("B5", "fusion_rel", "fusion_sibling_matched", "rand:hit@30", kind="relations vs siblings (equal tokens)"),
        T("B6", "fusion_rel", "fusion_xpath_matched", "rand:hit@30", kind="relations vs tag/attr (equal tokens)"),
        T("B7", f"fusion_parent_{CAP}", "bm25", "rand:hit@30", kind="parent context alone vs bm25"),
    ]
    # R2 conservative bound computed directly
    by_site, flat = defaultdict(list), []
    for (qid, site), mm in per_q.items():
        d = mm["fusion_rel"]["worst:hit@30"] - mm["fusion_shuf"]["best:hit@30"]
        by_site[site].append(d)
        flat.append(d)
    est, lo, hi = C.cluster_boot(by_site)
    p, pos, neg = C.sign_test(flat)
    tests[2] = {"id": "R2", "type": "CONSERVATIVE H1: fusion worst-case ties - shuffled best-case ties",
                "A": "fusion_rel|worst", "B": "fusion_shuf|best", "metric": "hit@30", "n": len(flat),
                "diff": round(est, 4), "ci95": [round(lo, 4), round(hi, 4)], "wins": [pos, neg],
                "sign_p": float(f"{p:.3g}"), "sites_pos_neg": [sum(1 for v in by_site.values() if C.mean(v) > 0),
                                                              sum(1 for v in by_site.values() if C.mean(v) < 0)],
                "n_sites": len(by_site)}
    tests = [t for t in tests if t]
    # site -> page two-level bootstrap (accounts for paraphrase clusters within a page)
    for t in tests:
        if t["id"] == "R2":
            diffs = [(site, mm["_diag"]["page"], mm["fusion_rel"]["worst:hit@30"] - mm["fusion_shuf"]["best:hit@30"])
                     for (qid, site), mm in per_q.items()]
        else:
            diffs = [(site, mm["_diag"]["page"], mm[t["A"]][t["metric"]] - mm[t["B"]][t["metric"]])
                     for (qid, site), mm in per_q.items() if t["A"] in mm and t["B"] in mm]
        _, lo2, hi2 = S.two_level_boot(diffs)
        t["page_ci95"] = [round(lo2, 4), round(hi2, 4)]
    fam = [t for t in tests if t["id"] in ("R1", "R2", "B1", "B2", "B3", "B4", "B5", "B6")]
    for t, adj in zip(fam, holm([t["sign_p"] for t in fam])):
        t["holm_p"] = float(f"{adj:.3g}")
    lam = None
    if do_lambda:
        lam = {}
        for x in lambdas:
            r_ = C.mean(v[f"lam{x}_rel"]["rand:hit@30"] for v in vals)
            s_ = C.mean(v[f"lam{x}_shuf"]["rand:hit@30"] for v in vals)
            lam[str(x)] = {"rel@30": round(r_, 4), "shuf@30": round(s_, 4), "gap": round(r_ - s_, 4)}
    sub = {}
    for nm, keep in (("bridge", lambda m: m["_diag"]["bridge"] == 1), ("non_bridge", lambda m: m["_diag"]["bridge"] == 0),
                     ("silent", lambda m: m["_diag"]["silent"] == 1)):
        vs = [v for v in vals if keep(v)]
        sub[nm] = {"n": len(vs), **{m: round(C.mean(v[m]["rand:hit@30"] for v in vs), 4)
                                    for m in ["bm25", "fusion_rel", "fusion_shuf", f"fusion_parent_{CAP}",
                                              f"fusion_sibling_{CAP}", f"fusion_xpath_{CAP}"]}} if vs else {"n": 0}
    cov_rows, cov_tot = coverage(a.data, sites)
    summ = {"experiment": "C8 secondary analyses (pre-registered in PREREG_C8.md)", "role": a.role,
            "sample": "diverse(random pages, <=%d/page)" % a.max_per_page if a.diverse else "first-N per site",
            "n_pages": len({mm["_diag"]["page"] for mm in per_q.values()}),
            "rel_version": C.REL_VERSION, "sites": sorted({k[1] for k in per_q}), "n": len(per_q),
            "methods": table, "tests": tests, "lambda_grid_train_only": lam, "subsets_rand@30": sub,
            "mean_tie_size_at_gold": {m: round(C.mean(v[m]["tie_size"] for v in vals), 2) for m in ("bm25", "fusion_rel", "fusion_shuf")},
            "coverage_E4": {"total": cov_tot, "by_site": cov_rows}}
    (out / "c8_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== C8 [{a.role}] rel={C.REL_VERSION} n={len(per_q)} pages={summ['n_pages']} sample={summ['sample']} sites={summ['sites']} ===")
    print("\nE3  Recall@K (expected under random ties) and reader tokens")
    head = ["bm25", "fusion_rel", "fusion_shuf"]
    print(f"{'K':>5s}" + "".join(f"{m:>14s}" for m in head))
    for k in KS:
        print(f"{k:5d}" + "".join(f"{table[m][f'rand:hit@{k}']:14.3f}" for m in head))
    print("tokens@30  bm25={:.0f}  fusion={:.0f}".format(table["bm25"]["tokens@30"], table["fusion_rel"]["tokens@30"]))
    print("mean tie size at gold:", summ["mean_tie_size_at_gold"])
    print("\nE1  context baselines, Recall@30 (random ties)")
    for m in methods:
        if m.startswith("fusion_") or m == "bm25":
            print(f"  {m:28s} {table[m]['rand:hit@30']:.3f}   hit@1 {table[m]['rand:hit@1']:.3f}")
    print("\nsubsets (rand@30):", json.dumps(sub))
    if lam:
        print("\nlambda grid (TRAIN):", json.dumps(lam))
    print("\n--- paired tests (site-cluster bootstrap; Holm over R1,R2,B1-B6) ---")
    for t in tests:
        print(f"{t['id']:6s} {t['A']} - {t['B']} [{t['metric']}] n={t['n']} diff={t['diff'] * 100:+.1f}%p "
              f"CI[{t['ci95'][0] * 100:+.1f},{t['ci95'][1] * 100:+.1f}] pageCI[{t['page_ci95'][0] * 100:+.1f},{t['page_ci95'][1] * 100:+.1f}] wins {t['wins'][0]}:{t['wins'][1]} "
              f"sites+/-={t['sites_pos_neg'][0]}/{t['sites_pos_neg'][1]} p={t['sign_p']}"
              + (f" holm={t['holm_p']}" if "holm_p" in t else "") + f"  | {t['type']}")
    print("\nE4 coverage:", json.dumps(cov_tot))
    print(f"[saved] {out.resolve()}")


if __name__ == "__main__":
    main()
