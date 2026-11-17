# -*- coding: utf-8 -*-
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import c4_lib as C
import c8_robust as R
import idc_lib as L
import test_c4 as T4

HERE = Path(__file__).resolve().parent


class C8Tests(unittest.TestCase):
    def test_analytic_ties_match_bruteforce(self):
        rng = random.Random(0)
        for _ in range(200):
            n = rng.randint(2, 40)
            cands = list(range(n))
            score = {i: rng.choice([0.0, 0.0, 1.0, 2.0, 2.0, 3.5]) for i in cands}
            g = rng.choice(cands)
            tm = R.tie_metrics(score, cands, g)
            for k in (1, 5, 10, 30):
                hits = 0
                for s in range(400):
                    perm = {i: rng.random() for i in cands}
                    order = sorted(cands, key=lambda i: (-score[i], perm[i]))
                    hits += int(g in order[:k])
                self.assertAlmostEqual(tm[f"rand:hit@{k}"], hits / 400, delta=0.1)
                self.assertLessEqual(tm[f"worst:hit@{k}"], tm[f"rand:hit@{k}"])
                self.assertLessEqual(tm[f"rand:hit@{k}"], tm[f"best:hit@{k}"])
                order = sorted(cands, key=lambda i: (-score[i], i))
                self.assertEqual(tm[f"doc:hit@{k}"], int(g in order[:k]))

    def test_matched_budget_equal_tokens(self):
        html, _ = T4.table_page(1)
        pg = L.parse_page(html)
        for i in range(len(pg.els)):
            n_rel = len(C.rel_string(pg, i).split())
            for kind in ("parent", "sibling", "xpath"):
                raw = R.CTX_FN[kind](pg, i).split()[:n_rel]
                self.assertLessEqual(len(raw), n_rel)

    def test_contexts_are_gold_free_and_nonempty(self):
        html, _ = T4.table_page(1)
        pg = L.parse_page(html)
        i = pg.tid2i["16"]                      # Beta price cell
        self.assertIn("Beta", R.ctx_sibling(pg, i))
        self.assertIn("td", R.ctx_xpath(pg, i))
        self.assertNotIn(pg.text[i], R.ctx_parent(pg, i).split()[:1])

    def test_holm(self):
        self.assertEqual(R.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_end_to_end_train_role(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "official_train"
            T4.make_data(root)
            sp = Path(d) / "split.json"
            subprocess.run([sys.executable, str(HERE / "c4_run.py"), "split", "--train-data", str(root), "--out", str(sp)],
                           capture_output=True, check=True)
            p = subprocess.run([sys.executable, str(HERE / "c8_robust.py"), "--data", str(root), "--role", "train",
                                "--split-file", str(sp), "--out", str(Path(d) / "c8")], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            s = json.loads((Path(d) / "c8" / "c8_summary.json").read_text(encoding="utf-8"))
            self.assertIsNotNone(s["lambda_grid_train_only"])
            self.assertTrue(any(t["id"] == "R2" for t in s["tests"]))
            self.assertEqual(s["coverage_E4"]["total"]["span_with_node"], s["n"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
