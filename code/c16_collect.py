# -*- coding: utf-8 -*-
"""Print C16 results of the four site sets as one markdown report (also saved as c16_report.md).
usage: python c16_collect.py --root results\\c16"""
import argparse
import json
from pathlib import Path

ROLES = ("dev", "select", "confirm", "test")
CONDS = ("text", "own", "other", "struct_all", "struct_all_nosem", "all8", "prev", "prev_nosem")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"results\c16")
    a = ap.parse_args()
    root = Path(a.root)
    S = {r: json.loads((root / r / "c16_summary.json").read_text(encoding="utf-8"))
         for r in ROLES if (root / r / "c16_summary.json").exists()}
    if not S:
        raise SystemExit(f"no c16_summary.json under {root}")
    roles = list(S)
    retr = ["bm25"] + sorted({m.rsplit("_", 1)[0] if False else "_".join(m.split("_")[:2])
                              for s in S.values() for m in s["methods"] if not m.startswith("bm25")})
    retr = [r for r in dict.fromkeys(retr)]
    L = [f"# C16 report (hash {', '.join(sorted({s['code_hash'] for s in S.values()}))})", ""]
    for rt in retr:
        L += [f"## {rt}: Recall@30", "", "| condition | " + " | ".join(roles) + " |", "|---|" + "---|" * len(roles)]
        for c in CONDS:
            L.append(f"| {c} | " + " | ".join(
                f"{S[r]['methods'][f'{rt}_{c}']['rand:hit@30']:.3f}" if f"{rt}_{c}" in S[r]["methods"] else "-" for r in roles) + " |")
        L.append("")
    L += ["## BM25 extra", "", "| condition | " + " | ".join(roles) + " |", "|---|" + "---|" * len(roles)]
    for m in ("bm25_struct_all_matched", "bm25_struct_all_nosem_matched", "bm25_other_s13", "bm25_other_s17", "bm25_other_s23"):
        L.append(f"| {m} | " + " | ".join(f"{S[r]['methods'][m]['rand:hit@30']:.3f}" if m in S[r]["methods"] else "-" for r in roles) + " |")
    L += ["", "## tests: diff %p [site CI] [site->page CI] sites+/- (bold = pass)", "",
          "| id | " + " | ".join(roles) + " |", "|---|" + "---|" * len(roles)]
    ids = list(dict.fromkeys(t["id"] for s in S.values() for t in s["tests"]))
    for tid in ids:
        cells = []
        for r in roles:
            t = next((t for t in S[r]["tests"] if t["id"] == tid), None)
            if not t:
                cells.append("-")
                continue
            ok = t["ci95"][0] > 0 and t["page_ci95"][0] > 0 and t["sites_pos_neg"][0] > t["sites_pos_neg"][1]
            b = "**" if ok else ""
            cells.append(f"{b}{t['diff'] * 100:+.1f}{b} [{t['ci95'][0] * 100:+.1f},{t['ci95'][1] * 100:+.1f}] "
                         f"[{t['page_ci95'][0] * 100:+.1f},{t['page_ci95'][1] * 100:+.1f}] {t['sites_pos_neg'][0]}/{t['sites_pos_neg'][1]}")
        L.append(f"| {tid} | " + " | ".join(cells) + " |")
    txt = "\n".join(L)
    (root / "c16_report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
