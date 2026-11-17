# -*- coding: utf-8 -*-
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import c15_map_audit as M
import idc_lib as L
import test_c4 as T4

HERE = Path(__file__).resolve().parent

MARKED = """<html><body tid="0">
<section tid="1"><h2 tid="2">Bestsellers</h2>
<table tid="3">
<thead tid="4"><tr tid="5"><th tid="6">Title</th><th tid="7" colspan="2">Price</th></tr></thead>
<tr tid="8"><th tid="9" scope="row">Book A</th><td tid="10">$32.99</td><td tid="11">USD</td></tr>
<tr tid="12"><th tid="13" scope="row">Book B</th><td tid="14">$18.50</td><td tid="15">USD</td></tr>
</table></section>
<dl tid="16"><dt tid="17">Run Time</dt><dd tid="18">128 min</dd></dl>
<label tid="19" for="c">Colour</label><span tid="20" id="c">Blue</span>
</body></html>"""


class C15Tests(unittest.TestCase):
    def setUp(self):
        self.pg = L.parse_page(MARKED)
        self.el = lambda tid: self.pg.els[self.pg.tid2i[tid]]

    def test_grid_headers_with_colspan_and_scope(self):
        table = self.el("3")
        rows, cols = M.gold_headers(table, self.el("10"), {})
        self.assertEqual(rows, ["Book A"])
        self.assertEqual(cols, ["Price"])
        rows, cols = M.gold_headers(table, self.el("15"), {})     # under the colspan=2 header
        self.assertEqual(rows, ["Book B"])
        self.assertEqual(cols, ["Price"])

    def test_headers_attribute_wins(self):
        pg = L.parse_page('<table tid="0"><tr tid="1"><th tid="2" id="h1">Cost</th><th tid="3" id="h2">Year</th></tr>'
                          '<tr tid="4"><td tid="5" headers="h2">2020</td><td tid="6">x</td></tr></table>')
        rows, cols = M.gold_headers(pg.els[pg.tid2i["0"]], pg.els[pg.tid2i["5"]], {})
        self.assertEqual((rows, cols), ([], ["Year"]))

    def test_label_and_heading_gold(self):
        self.assertEqual(M.gold_label(self.el("18")), ("Run Time", "dt_dd"))
        root = self.el("0")
        while root.parent is not None:
            root = root.parent
        self.assertEqual(M.label_for_map(root)["c"], "Colour")
        order = {id(t): k for k, t in enumerate(root.descendants)}
        self.assertEqual(M.gold_section_heading(self.el("10"), order), "Bestsellers")
        self.assertIsNone(M.gold_section_heading(self.el("2"), order))      # the heading itself

    def test_scoring(self):
        s = M.score_pair("Price:", ["price"])
        self.assertEqual((s["pred_nonempty"], s["exact"]), (1, 1))
        s = M.score_pair("", ["Price"])
        self.assertEqual((s["pred_nonempty"], s["exact"], s["tok_f1"]), (0, 0, 0.0))

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "official_train"
            T4.make_data(root)
            p = subprocess.run([sys.executable, str(HERE / "c15_map_audit.py"), "--data", str(root), "--role", "dev",
                                "--out", str(Path(d) / "c15")], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            s = json.loads((Path(d) / "c15" / "c15_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(set(s["accuracy"]), {"row", "col", "label", "head"})
            self.assertIn("share_tables_with_th", s["markup"])


if __name__ == "__main__":
    unittest.main()
