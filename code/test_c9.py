# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import c9_common as S
import c9_qa as Q
import idc_lib as L
import test_c4 as T4

HERE = Path(__file__).resolve().parent


class C9Tests(unittest.TestCase):
    def test_squad_metrics(self):
        self.assertEqual(Q.em("The $150.", "$150"), 1)
        self.assertEqual(Q.em("150 dollars", "$150"), 0)
        # official SQuAD normalisation deletes punctuation without a space: "10-Speed" -> "10speed"
        self.assertAlmostEqual(Q.f1("10-Speed Automatic", "10 Speed Automatic transmission"), 1 / 3)
        self.assertAlmostEqual(Q.f1("10 Speed Automatic", "10 Speed Automatic transmission"), 2 * 1 * 0.75 / 1.75)
        self.assertEqual(Q.f1("", ""), 1.0)

    def test_pack_skips_nested_and_respects_budget(self):
        html = ('<html><body tid="0"><div tid="1"><span tid="2">alpha beta gamma</span></div>'
                '<p tid="3">delta epsilon</p></body></html>')
        pg = L.parse_page(html)
        order = [pg.tid2i[t] for t in ("1", "2", "3")]           # parent, then its child, then other
        chosen, text = Q.pack(pg, order, budget_words=4)
        self.assertEqual([pg.tids[i] for i in chosen], ["1", "3"])  # child skipped (same text as parent)
        self.assertEqual(len(text.replace("|", " ").split()), 4)

    def test_diverse_sampler_caps_pages(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "data"
            T4.make_data(root)
            rows, _ = S.load_diverse(root, None, per_site=6, max_per_page=2, verbose=False)
            per_page = {}
            for r in rows:
                per_page[r.page_key] = per_page.get(r.page_key, 0) + 1
            self.assertLessEqual(max(per_page.values()), 2)
            again, _ = S.load_diverse(root, None, per_site=6, max_per_page=2, verbose=False)
            self.assertEqual([r.qid for r in rows], [r.qid for r in again])      # deterministic

    def test_two_level_boot(self):
        diffs = [(s, f"{s}/p{p}", 1.0) for s in "abc" for p in range(3)]
        est, lo, hi = S.two_level_boot(diffs, B=200)
        self.assertEqual((est, lo, hi), (1.0, 1.0, 1.0))

    @unittest.skipUnless(os.environ.get("C9_TINY_QA"), "set C9_TINY_QA=<local tiny QA model dir> for the e2e test")
    def test_end_to_end_tiny_reader(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "official_train"
            T4.make_data(root)
            sp = Path(d) / "split.json"
            subprocess.run([sys.executable, str(HERE / "c4_run.py"), "split", "--train-data", str(root), "--out", str(sp)],
                           capture_output=True, check=True)
            p = subprocess.run([sys.executable, str(HERE / "c9_qa.py"), "--data", str(root), "--role", "select",
                                "--split-file", str(sp), "--model", os.environ["C9_TINY_QA"], "--device", "cpu",
                                "--out", str(Path(d) / "qa")], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            s = json.loads((Path(d) / "qa" / "qa_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(s["methods"]["oracle_gold"]["gold_in_ctx"], 1.0)
            self.assertEqual(s["methods"]["oracle_gold"]["ans_in_ctx"], 1.0)
            self.assertTrue(any(t["id"].startswith("Q1@") for t in s["tests"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
