# -*- coding: utf-8 -*-
"""
C4: confirm the relational-bridge finding on unseen sites + explore new findings.

  split   : hash-split official TRAIN sites into train / select / confirm (dev sites excluded)
  run     : --role dev     -> exploration on already-seen Dev (prereg tests + E1..E6)
            --role select  -> exploration + extractor development sites (never confirm)
            --role confirm -> prereg tests ONLY, exploration disabled, runs ONCE (lock file)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import c4_lib as C
import idc_lib as L

HERE = Path(__file__).resolve().parent
DEV_SITES = ["au01", "au08", "bo17", "ga10", "ga12", "jo11", "mo02", "ph04", "sp09", "sp10"]

PREREG = [
    # id, A, B, metric, subset, expectation
    ("H1", "fusion_all", "fusion_shuf", "hit@30", "all", "PRIMARY: >0"),
    ("H2", "fusion_all", "fusion_qm", "hit@30", "qm_full", ">0 (beyond lexical overlap)"),
    ("H3a", "fusion_all", "fusion_shuf", "hit@30", "bridge", ">0"),
    ("H3b", "fusion_all", "fusion_shuf", "hit@30", "non_bridge", "<=0"),
    ("H4", "fusion_all", "bm25", "hit@30", "all", ">0"),
    ("H5", "fusion_need", "fusion_all", "hit@30", "all", "~0 (gate adds nothing)"),
    ("H6a", "tieonly", "bm25", "hit@30", "all", ">0"),
    ("H6b", "tieonly", "bm25", "hit@100", "all", ">0"),
    ("H7", "fusion_all", "fusion_shuf", "hit@1", "all", "secondary"),
]
SUBSETS = {
    "all": lambda m: True,
    "qm_full": lambda m: m["_diag"]["qm_full"] == 1,
    "bridge": lambda m: m["_diag"]["bridge"] == 1,
    "non_bridge": lambda m: m["_diag"]["bridge"] == 0,
}


def code_hash():
    h = hashlib.sha256()
    for f in ("idc_lib.py", "c4_lib.py", "c4_run.py"):
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def dump(path, rows):
    if not rows:
        return
    keys = list(OrderedDict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ split
def cmd_split(a):
    exclude = set(DEV_SITES) | set(filter(None, (a.exclude_sites or "").split(",")))
    _, _, sites = L.load_rows(Path(a.train_data), max_per_site=1)
    sites = [s for s in sites if s not in exclude]
    if len(sites) < 10:
        sys.exit(f"[split] only {len(sites)} non-dev sites found under {a.train_data}: {sites}")
    sp = C.split_sites(sites)
    sp["excluded"] = sorted(exclude)
    sp["sha256_16"] = hashlib.sha256(json.dumps({k: sp[k] for k in ("train", "select", "confirm")},
                                                sort_keys=True).encode()).hexdigest()[:16]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sp, indent=2), encoding="utf-8")
    print(json.dumps(sp, indent=2))


# ------------------------------------------------------------------ run
def cmd_run(a):
    role = a.role
    explore = role not in ("confirm", "test")
    sites = None
    split_info = None
    if role in ("select", "confirm", "train"):
        if not a.split_file:
            sys.exit("--split-file is required for this role")
        split_info = json.loads(Path(a.split_file).read_text(encoding="utf-8"))
        sites = set(split_info[role])
    elif a.sites:
        sites = set(a.sites.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if role == "test":
        lock = Path(a.data).resolve() / f"TEST_USED_{C.REL_VERSION}.lock"
    else:
        lock = Path(a.split_file).resolve().parent / "CONFIRM_USED.lock" if role == "confirm" else None
    if lock is not None and lock.exists() and not a.i_understand_this_voids_confirmation:
        sys.exit(f"[confirm] already run once ({lock.read_text(encoding='utf-8').strip()}). "
                 f"Re-running confirm sites after seeing results makes them exploratory.")

    rows, skipped, _ = L.load_rows(Path(a.data), sites, None, a.max_per_site)
    if not rows:
        sys.exit("[run] 0 rows loaded")
    if lock is not None:  # lock only once data is confirmed loadable
        lock.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} code={code_hash()} out={out}", encoding="utf-8")
    by_page = OrderedDict()
    for r in rows:
        by_page.setdefault(r.page_key, []).append(r)

    per_q, recs, diags, audit = {}, [], [], defaultdict(int)
    t0 = time.time()
    for n, (pk, qs) in enumerate(by_page.items(), 1):
        pg = L.parse_page(qs[0].html_path.read_text(encoding="utf-8", errors="replace"))
        for r in qs:
            g = pg.tid2i.get(r.element_id)
            if g is None or not pg.key[g]:
                audit["gt_missing_or_empty"] += 1
                continue
            audit["evaluated"] += 1
            ranked, aux = C.rank_all(pg, r.question, seed=a.seed, explore=explore)  # no gold
            dg = C.diagnostics(pg, g, r.question, aux)
            dg.update({"qid": r.qid, "site": r.site, "answer_in_gold": int(r.answer.lower().strip() in pg.key[g])})
            diags.append(dg)
            mm = {"_diag": dg}
            for m, lst in ranked.items():
                ev = C.metrics(lst, g)
                mm[m] = ev
                recs.append({"qid": r.qid, "site": r.site, "method": m, **ev})
            per_q[(r.qid, r.site)] = mm
        del pg
        if n % 50 == 0:
            print(f"  pages {n}/{len(by_page)} q={audit['evaluated']} {time.time() - t0:.0f}s")

    dump(out / "rows.csv", recs)
    dump(out / "diagnostics.csv", diags)

    methods = list(OrderedDict.fromkeys(r["method"] for r in recs))
    table = {}
    for m in methods:
        rs = [x for x in recs if x["method"] == m]
        table[m] = {"n": len(rs), **{k: round(C.mean(x[k] for x in rs), 4)
                                     for k in ("hit@1", "hit@10", "hit@30", "hit@100", "mrr@10")}}
    by_site = []
    for m in methods:
        b = defaultdict(list)
        for x in recs:
            if x["method"] == m:
                b[x["site"]].append(x)
        for s, rs in sorted(b.items()):
            by_site.append({"method": m, "site": s, "n": len(rs),
                            **{k: round(C.mean(x[k] for x in rs), 4) for k in ("hit@1", "hit@30", "hit@100")}})
    dump(out / "by_site.csv", by_site)

    tests = []
    for hid, A, B, met, sub, exp in PREREG:
        t = C.paired(per_q, A, B, met, SUBSETS[sub])
        if t:
            t.update({"id": hid, "subset": sub, "expectation": exp})
            tests.append(t)
    h1 = next((t for t in tests if t["id"] == "H1"), None)
    verdict = None
    if h1:
        verdict = bool(h1["ci95"][0] > 0 and h1["sites_pos_neg"][0] > h1["sites_pos_neg"][1])

    D = diags
    desc = {
        "n": len(D),
        "gold_bm25_zero": round(C.mean(d["gold_bm25_zero"] for d in D), 4),
        "gold_dup_text": round(C.mean(d["gold_dup_text"] for d in D), 4),
        "bridge_rate": round(C.mean(d["bridge"] for d in D), 4),
        "silent_and_bridge": round(C.mean(d["gold_bm25_zero"] * d["bridge"] for d in D), 4),
        "qm_full_rate": round(C.mean(d["qm_full"] for d in D), 4),
        "answer_in_gold": round(C.mean(d["answer_in_gold"] for d in D), 4),
    }
    summary = {"experiment": "C4 relational-bridge", "role": role, "args": vars(a),
               "code_sha256_16": code_hash(), "split_sha": split_info and split_info.get("sha256_16"),
               "sites": sorted({r.site for r in rows}), "audit": dict(audit), "skipped": skipped,
               "frozen": {"LAMBDA": C.LAMBDA, "CTX_MAX_TOKENS": L.CTX_MAX_TOKENS,
                          "LABEL_MAX_TOKENS": L.LABEL_MAX_TOKENS, "PARTS": C.PARTS,
                          "REL_VERSION": C.REL_VERSION},
               "descriptive": desc, "methods": table, "prereg_tests": tests,
               "H1_pass(CI>0 and more sites + than -)": verdict}
    if explore:
        summary["exploration"] = explore_tables(per_q, D)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report(summary)


# ------------------------------------------------------------------ exploration
def explore_tables(per_q, D):
    E = {}
    vals = list(per_q.values())
    # E1 which relation part carries the bridge?
    E["E1_relation_part_ablation"] = {}
    for m in ["fusion_all"] + [f"x_fusion_{p}" for p in C.PARTS] + ["bm25"]:
        E["E1_relation_part_ablation"][m] = {
            "hit@30": round(C.mean(v[m]["hit@30"] for v in vals), 4),
            "hit@1": round(C.mean(v[m]["hit@1"] for v in vals), 4)}
    E["E1_part_present_on_gold"] = {p: round(C.mean(d[f"has_{p}"] for d in D), 4) for p in C.PARTS}
    E["E1_part_bridges_question"] = {p: round(C.mean(d[f"key_{p}"] for d in D), 4) for p in C.PARTS}
    # E2 two-key addressing: how many independent relation parts link gold to question?
    grp = defaultdict(list)
    for v in vals:
        k = v["_diag"]["gold_keys"]
        grp[min(k, 3)].append(v)
    E["E2_gold_keys"] = {}
    for k in sorted(grp):
        vs = grp[k]
        E["E2_gold_keys"][f"{k}{'+' if k == 3 else ''}"] = {
            "n": len(vs), "share": round(len(vs) / len(vals), 4),
            "fusion_all_hit@30": round(C.mean(v["fusion_all"]["hit@30"] for v in vs), 4),
            "fusion_all_hit@1": round(C.mean(v["fusion_all"]["hit@1"] for v in vs), 4),
            "x_keycount_hit@1": round(C.mean(v["x_keycount"]["hit@1"] for v in vs), 4),
            "median_competitors_with_>=keys": sorted(v["_diag"]["n_nodes_keys_ge_gold"] for v in vs)[len(vs) // 2]}
    E["E2_keycount_vs_fusion"] = C.paired(per_q, "x_keycount", "fusion_all", "hit@1")
    # E3 what stays ambiguous even WITH relations?
    E["E3_ceilings"] = {
        "share_text_twin": round(C.mean(int(d["n_same_text"] > 1) for d in D), 4),
        "share_text_and_relation_twin": round(C.mean(int(d["n_same_text_and_rel"] > 1) for d in D), 4),
        "ceiling_hit1_text_only": round(C.mean(d["ceiling_text"] for d in D), 4),
        "ceiling_hit1_text_plus_relations": round(C.mean(d["ceiling_text_rel"] for d in D), 4)}
    # E4 answer-type compatibility as a GT-free signal
    E["E4_answer_type"] = {}
    for t in ("money", "number", "other"):
        ds = [d for d in D if d["qtype"] == t]
        if ds:
            E["E4_answer_type"][t] = {"share": round(len(ds) / len(D), 4),
                                      "gold_type_ok": round(C.mean(d["gold_type_ok"] for d in ds), 4),
                                      "cands_type_ok_frac": round(C.mean(d["n_type_ok"] / max(1, d["n_cands"]) for d in ds), 4)}
    E["E4_type_prior_vs_fusion_hit@1"] = C.paired(per_q, "x_fusion_type", "fusion_all", "hit@1",
                                                  lambda m: m["_diag"]["qtype"] != "other")
    E["E4_type_prior_vs_fusion_hit@30"] = C.paired(per_q, "x_fusion_type", "fusion_all", "hit@30",
                                                   lambda m: m["_diag"]["qtype"] != "other")
    # E5 bridge competition: many nodes share the question-matching relation (e.g. a column header)
    bins = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 10 ** 9)]
    E["E5_competition_bridge_questions"] = {}
    for lo, hi in bins:
        vs = [v for v in vals if v["_diag"]["bridge"] and lo <= v["_diag"]["n_bridge_nodes"] <= hi]
        if vs:
            E["E5_competition_bridge_questions"][f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = {
                "n": len(vs), "fusion_all_hit@30": round(C.mean(v["fusion_all"]["hit@30"] for v in vs), 4),
                "fusion_all_hit@1": round(C.mean(v["fusion_all"]["hit@1"] for v in vs), 4)}
    # E6 layout profile per site
    E["E6_site_profile"] = {}
    for s in sorted({k[1] for k in per_q}):
        ds = [v["_diag"] for k, v in per_q.items() if k[1] == s]
        E["E6_site_profile"][s] = {"n": len(ds), "bridge": round(C.mean(d["bridge"] for d in ds), 3),
                                   "silent": round(C.mean(d["gold_bm25_zero"] for d in ds), 3),
                                   **{f"has_{p}": round(C.mean(d[f"has_{p}"] for d in ds), 3) for p in C.PARTS},
                                   **{f"key_{p}": round(C.mean(d[f"key_{p}"] for d in ds), 3) for p in C.PARTS}}
    return E


def report(s):
    print(f"\n=== C4 [{s['role']}] rel={s['frozen']['REL_VERSION']} sites={s['sites']} audit={s['audit']} code={s['code_sha256_16']}")
    print("descriptive:", json.dumps(s["descriptive"]))
    print(f"\n{'method':18s}{'N':>6s}{'hit@1':>9s}{'hit@10':>9s}{'hit@30':>9s}{'hit@100':>9s}{'mrr@10':>9s}")
    for m, v in s["methods"].items():
        print(f"{m:18s}{v['n']:6d}{v['hit@1']:9.3f}{v['hit@10']:9.3f}{v['hit@30']:9.3f}{v['hit@100']:9.3f}{v['mrr@10']:9.3f}")
    print("\n--- pre-registered paired tests (site-cluster bootstrap 95% CI) ---")
    for t in s["prereg_tests"]:
        print(f"{t['id']:4s} {t['A']}-{t['B']} [{t['metric']},{t['subset']}] n={t['n']} diff={t['diff'] * 100:+.1f}%p "
              f"CI[{t['ci95'][0] * 100:+.1f},{t['ci95'][1] * 100:+.1f}] wins {t['wins'][0]}:{t['wins'][1]} "
              f"p={t['sign_p']} sites+/-={t['sites_pos_neg'][0]}/{t['sites_pos_neg'][1]}  expect {t['expectation']}")
    print(f"\nH1 PASS = {s['H1_pass(CI>0 and more sites + than -)']}")
    if "exploration" in s:
        print("\n--- exploration (NOT confirmatory) ---")
        for k, v in s["exploration"].items():
            print(f"{k}: {json.dumps(v, ensure_ascii=False)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("split")
    s1.add_argument("--train-data", required=True)
    s1.add_argument("--exclude-sites", default="")
    s1.add_argument("--out", default="results/c4/split.json")
    s2 = sub.add_parser("run")
    s2.add_argument("--data", required=True)
    s2.add_argument("--role", choices=["dev", "train", "select", "confirm", "test"], required=True)
    s2.add_argument("--split-file")
    s2.add_argument("--sites")
    s2.add_argument("--max-per-site", type=int, default=100)
    s2.add_argument("--seed", type=int, default=13)
    s2.add_argument("--out", required=True)
    s2.add_argument("--i-understand-this-voids-confirmation", action="store_true")
    a = ap.parse_args()
    cmd_split(a) if a.cmd == "split" else cmd_run(a)


if __name__ == "__main__":
    main()
