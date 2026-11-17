# -*- coding: utf-8 -*-
"""
C10: prepare the official WebSRC Test set for the frozen C8/C9 analyses.

1) unzip WebSRC_v1.0_test.zip -> official_test/
2) merge official answers (test_answers_flat.json) into every site's dataset.csv
   (question,id,element_id,answer_start,answer). If a site has no dataset.csv,
   build one from the answers + the site's processed_data HTML files.
3) audit: questions per site, with/without node, HTML resolvable, gold tid present in HTML.
No scores are computed here.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import idc_lib as L

COLS = ["question", "id", "element_id", "answer_start", "answer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="WebSRC_v1.0_test.zip")
    ap.add_argument("--answers", default="official_test_answers/test_answers_flat.json")
    ap.add_argument("--out", default="official_test")
    ap.add_argument("--audit-pages", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    if not out.exists():
        print(f"[unzip] {a.zip} -> {out}")
        with zipfile.ZipFile(a.zip) as z:
            z.extractall(out)
    ans = json.loads(Path(a.answers).read_text(encoding="utf-8-sig"))
    by_site = defaultdict(list)
    for qid, v in ans.items():
        by_site[qid[:4]].append(v)

    # locate site directories: folders that contain processed_data/*.html (or html files)
    site_dirs = {}
    for d in out.rglob("processed_data"):
        if d.is_dir() and any(d.glob("*.html")):
            sd = d.parent
            csvs = list(sd.glob("*.csv"))
            code = None
            for c in csvs:  # prefer the id prefix inside an existing csv
                with open(c, encoding="utf-8-sig", newline="") as f:
                    rd = csv.DictReader(f)
                    cm = L._colmap(rd.fieldnames or [])
                    if "qid" in cm:
                        first = next(rd, None)
                        if first:
                            code = str(first[cm["qid"]])[:4]
                            break
            if code is None:
                code = L.site_from("", sd / "dataset.csv")
            site_dirs[code] = sd
    print(f"[sites] html dirs found: {sorted(site_dirs)}")
    missing = sorted(set(by_site) - set(site_dirs))
    if missing:
        print(f"[warn] answer sites without HTML dir: {missing}")

    report = {}
    for code, sd in sorted(site_dirs.items()):
        qa = by_site.get(code, [])
        if not qa:
            continue
        dst = sd / "dataset.csv"
        if dst.exists() and not (sd / "dataset_original.csv").exists():
            shutil.copy(dst, sd / "dataset_original.csv")
        with open(dst, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for v in sorted(qa, key=lambda x: x["id"]):
                eid = v.get("element_id", -1)
                w.writerow([v.get("question", ""), v["id"], -1 if eid is None else eid,
                            v.get("answer_start", -1), v.get("answer", "")])
        stems = L._html_index(sd)
        c = Counter()
        for v in qa:
            c["questions"] += 1
            eid = v.get("element_id", -1)
            c["with_node"] += int(eid is not None and int(eid) >= 0)
            c["html_resolvable"] += int(L.resolve_page(v["id"], stems) is not None)
        # spot-check that gold tids exist in the HTML
        checked = ok = 0
        for v in sorted(qa, key=lambda x: x["id"])[:: max(1, len(qa) // 30)]:
            eid = v.get("element_id", -1)
            pk = L.resolve_page(v["id"], stems)
            if pk is None or eid is None or int(eid) < 0:
                continue
            pg = L.parse_page(stems[pk].read_text(encoding="utf-8", errors="replace"))
            g = pg.tid2i.get(str(int(eid)))
            checked += 1
            ok += int(g is not None and L.norm_key(v.get("answer", "")) in pg.key[g])
        c["spot_checked"], c["spot_gold_tid_and_answer_ok"] = checked, ok
        report[code] = dict(c)
    Path(out / "C10_prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    tot = Counter()
    for v in report.values():
        tot.update(v)
    print("TOTAL", dict(tot))
    rows, skipped, sites = L.load_rows(out, max_per_site=1)
    print(f"[loader check] sites={sites} skipped={skipped}")


if __name__ == "__main__":
    main()
