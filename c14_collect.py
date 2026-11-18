# -*- coding: utf-8 -*-
"""Collect C14 summaries of the four site sets into the Table 2 matrix and the pre-registered tests.
usage: python c14_collect.py --root results\\c14   ->  prints markdown, writes c14_report.md"""
import argparse
import json
from pathlib import Path

from c14_run import CONDS, LABEL

ROLES = ("dev", "select", "confirm", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    S = {}
    for r in ROLES:
        p = root / r / "c14_summary.json"
        if p.exists():
            S[r] = json.loads(p.read_text(encoding="utf-8"))
    if not S:
        raise SystemExit(f"no c14_summary.json under {root}")
    retr = []
    for s in S.values():
        for m in s["methods"]:
            rt = m.split("_")[0] if m.startswith("bm25") else "_".join(m.split("_")[:2])
            if rt not in retr:
                retr.append(rt)
    roles = [r for r in ROLES if r in S]
    L = [f"# C14 report  (hash {', '.join(sorted({s['code_hash'] for s in S.values()}))})", ""]
    for rt in retr:
        L += [f"## {rt}: Recall@30 (Hit@1)", "", "| condition | " + " | ".join(roles) + " |",
              "|---|" + "---|" * len(roles)]
        for c in CONDS:
            cells = []
            for r in roles:
                v = S[r]["methods"].get(f"{rt}_{c}")
                cells.append(f"{v['rand:hit@30']:.3f} ({v['rand:hit@1']:.3f})" if v else "-")
            L.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## BM25, equal tokens (matched), Recall@30", "", "| condition | " + " | ".join(roles) + " |",
          "|---|" + "---|" * len(roles)]
    for c in ("parent", "siblings", "tagpath", "parent_nosem"):
        L.append(f"| {LABEL[c]} | " + " | ".join(
            f"{S[r]['methods'][f'bm25_{c}_matched']['rand:hit@30']:.3f}" if f"bm25_{c}_matched" in S[r]["methods"] else "-"
            for r in roles) + " |")
    L += ["", "## tests (diff %p, site CI, site->page CI, sites +/-)", "", "| id | " + " | ".join(roles) + " |",
          "|---|" + "---|" * len(roles)]
    ids = []
    for s in S.values():
        for t in s["tests"]:
            if t["id"] not in ids:
                ids.append(t["id"])
    for tid in ids:
        cells = []
        for r in roles:
            t = next((t for t in S[r]["tests"] if t["id"] == tid), None)
            if not t:
                cells.append("-")
                continue
            ok = t["ci95"][0] > 0 and t["page_ci95"][0] > 0 and t["sites_pos_neg"][0] > t["sites_pos_neg"][1]
            cells.append(f"{'**' if ok else ''}{t['diff'] * 100:+.1f}{'**' if ok else ''} "
                         f"[{t['ci95'][0] * 100:+.1f},{t['ci95'][1] * 100:+.1f}] "
                         f"[{t['page_ci95'][0] * 100:+.1f},{t['page_ci95'][1] * 100:+.1f}] "
                         f"{t['sites_pos_neg'][0]}/{t['sites_pos_neg'][1]}")
        L.append(f"| {tid} | " + " | ".join(cells) + " |")
    L += ["", "bold = pass rule (both CIs > 0 and more positive sites).", "",
          "silent share: " + ", ".join(f"{r} {S[r]['silent_share']}" for r in roles),
          "mean field tokens at gold: " + "; ".join(f"{r} {json.dumps(S[r]['mean_field_tokens_at_gold'])}" for r in roles)]
    txt = "\n".join(L)
    (root / "c14_report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
