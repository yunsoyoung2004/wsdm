# -*- coding: utf-8 -*-
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import c16_run as M
import c4_lib as C
import idc_lib as L
import test_c4 as T4

HERE = Path(__file__).resolve().parent


class C16Tests(unittest.TestCase):
    def setUp(self):
        html, self.qs = T4.table_page(1)
        self.pg = L.parse_page(html)
        self.g = self.pg.tid2i[str(self.qs[2][1])]

    def test_struct_all_contains_each_structural_relation(self):
        s = M.struct_all(self.pg, self.g).lower().split()
        self.assertIn("td", s)                                   # tag path
        self.assertEqual(len(s), len(set(s)))                    # deduplicated

    def test_nosem_removes_semantic_tokens(self):
        sem = set(L.tokenize(C.rel_string(self.pg, self.g)))
        for c in ("struct_all_nosem", "prev_nosem"):
            self.assertFalse(sem & set(L.tokenize(M.field_string(self.pg, c, self.g))), c)

    def test_caps(self):
        for c in M.FIELDS:
            self.assertLessEqual(len(M.field_string(self.pg, c, self.g).split()), M.CAP)
        n_own = len(C.rel_string(self.pg, self.g).split())
        self.assertLessEqual(len(M.field_string(self.pg, "struct_all", self.g, "matched").split()), n_own)

    def test_all8_starts_with_own(self):
        own = C.rel_string(self.pg, self.g).split()
        self.assertEqual(M.field_string(self.pg, "all8", self.g).split()[:len(own)], own)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "official_train"
            T4.make_data(root)
            p = subprocess.run([sys.executable, str(HERE / "c16_run.py"), "--data", str(root), "--role", "dev",
                                "--out", str(Path(d) / "c16")], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            s = json.loads((Path(d) / "c16" / "c16_summary.json").read_text(encoding="utf-8"))
            ids = {t["id"] for t in s["tests"]}
            self.assertTrue({"K1_bm25", "K2_bm25", "K3_bm25", "S1_bm25", "M1", "SEED_13", "SEED_17", "SEED_23"} <= ids)
            self.assertTrue(all("holm_p" in t for t in s["tests"] if t["id"].startswith("K")))


if __name__ == "__main__":
    unittest.main()
