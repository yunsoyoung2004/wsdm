# -*- coding: utf-8 -*-
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import c14_run as M
import c4_lib as C
import idc_lib as L
import test_c4 as T4

HERE = Path(__file__).resolve().parent


class C14Tests(unittest.TestCase):
    def setUp(self):
        html, self.qs = T4.table_page(1)
        self.pg = L.parse_page(html)
        self.g = self.pg.tid2i[str(self.qs[2][1])]          # "What is the price of Beta?" -> price cell

    def test_tag_path_is_element_names_only(self):
        tp = M.tag_path(self.pg, self.g)
        self.assertTrue(tp.startswith("html body"), tp)
        self.assertTrue(tp.endswith("td"), tp)
        self.assertTrue(all(w.isalpha() for w in tp.split()))

    def test_parent_nosem_removes_semantic_tokens(self):
        sem = set(L.tokenize(C.rel_string(self.pg, self.g)))
        par = set(L.tokenize(M.field_string(self.pg, "parent", self.g)))
        nos = set(L.tokenize(M.field_string(self.pg, "parent_nosem", self.g)))
        self.assertTrue(sem & par, "row header should sit inside the parent row")
        self.assertFalse(sem & nos)
        self.assertTrue(nos <= par)

    def test_caps(self):
        for c in M.FIELD_CONDS + ("own",):
            self.assertLessEqual(len(M.field_string(self.pg, c, self.g).split()), M.CAP)
        n_own = len(C.rel_string(self.pg, self.g).split())
        for c in M.FIELD_CONDS:
            self.assertLessEqual(len(M.field_string(self.pg, c, self.g, "matched").split()), n_own)

    def test_end_to_end_bm25(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "official_train"
            T4.make_data(root)
            p = subprocess.run([sys.executable, str(HERE / "c14_run.py"), "--data", str(root), "--role", "dev",
                                "--out", str(Path(d) / "c14")], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            s = json.loads((Path(d) / "c14" / "c14_summary.json").read_text(encoding="utf-8"))
            for c in M.CONDS:
                self.assertIn(f"bm25_{c}", s["methods"])
            ids = {t["id"] for t in s["tests"]}
            self.assertTrue({"P1_bm25", "P2_bm25", "P3_bm25", "P4_bm25", "P5_bm25", "S2_bm25", "S5_bm25", "M1", "Q1"} <= ids)
            self.assertTrue(all("holm_p" in t for t in s["tests"] if t["id"].startswith("P")))


if __name__ == "__main__":
    unittest.main()
