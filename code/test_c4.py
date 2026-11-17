# -*- coding: utf-8 -*-
"""C4 tests on synthetic pages in the official WebSRC layout (logic only; real numbers need official data)."""
import csv
import inspect
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import c4_lib as C
import idc_lib as L

HERE = Path(__file__).resolve().parent


def table_page(seed):
    rng = random.Random(seed)
    names = ["Alpha", "Beta", "Gamma", "Delta", "Omega", "Sigma"]
    rows, tid = [], 10
    for n in names:
        price = rng.choice(["$150", "$200"])
        rows.append(f'<tr tid="{tid}"><td tid="{tid + 1}">{n}</td><td tid="{tid + 2}">{price}</td>'
                    f'<td tid="{tid + 3}">{rng.randint(1, 9)} kg</td></tr>')
        tid += 4
    html = ('<html><body tid="0"><h2 tid="1">Specifications</h2><table tid="2">'
            '<tr tid="3"><th tid="4">Model</th><th tid="5">Price</th><th tid="6">Weight</th></tr>'
            + "".join(rows) + '</table><p tid="9">Contact our store for details</p></body></html>')
    qs = []
    for k, n in enumerate(names):
        base = 10 + 4 * k
        qs.append((f"What is the price of {n}?", base + 2))
        qs.append((f"How heavy is {n}?", base + 3))
    return html, qs


def card_page(seed):
    rng = random.Random(seed)
    books = ["Siege", "Other Wife", "River", "Night Sky"]
    cards, tid, qs = [], 10, []
    for b in books:
        cards.append(f'<div tid="{tid}"><div tid="{tid + 1}">{b}</div><div tid="{tid + 2}">'
                     f'{rng.choice(["$32.99", "$12.00"])}</div></div>')
        qs.append((f"What is the cost of {b}?", tid + 2))
        tid += 3
    return '<html><body tid="0"><h1 tid="1">Books</h1>' + "".join(cards) + "</body></html>", qs


def make_data(root: Path, n_sites=12):
    for s in range(1, n_sites + 1):
        site = root / "test" / f"{s:02d}"
        (site / "processed_data").mkdir(parents=True)
        out = []
        for p, fn in enumerate((table_page, card_page)):
            html, qs = fn(s * 10 + p)
            pid = f"{s:02d}{p:05d}"
            (site / "processed_data" / f"{pid}.html").write_text(html, encoding="utf-8")
            pg = L.parse_page(html)
            for k, (q, tid) in enumerate(qs):
                out.append([q, f"te{pid}{k:05d}", tid, 0, pg.text[pg.tid2i[str(tid)]]])
        with open(site / "dataset.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "id", "element_id", "answer_start", "answer"])
            w.writerows(out)


class C4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "official_train"
        make_data(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_cli(self, *args):
        p = subprocess.run([sys.executable, str(HERE / "c4_run.py"), *args], capture_output=True,
                           text=True, encoding="utf-8")
        return p

    def test_split_deterministic_disjoint(self):
        s = [f"te{i:02d}" for i in range(1, 51)]
        a, b = C.split_sites(s), C.split_sites(list(reversed(s)))
        self.assertEqual(a, b)
        self.assertEqual((len(a["train"]), len(a["select"]), len(a["confirm"])), (30, 10, 10))
        self.assertFalse(set(a["train"]) & set(a["select"]) | set(a["train"]) & set(a["confirm"]))

    def test_rankers_never_see_gold(self):
        self.assertFalse({"g", "gold", "gt", "element_id", "answer"} & set(inspect.signature(C.rank_all).parameters))

    def test_relation_parts(self):
        html, _ = table_page(1)
        pg = L.parse_page(html)
        p = C.rel_parts(pg, pg.tid2i["16"])  # Beta price cell
        self.assertEqual((p["row"], p["col"], p["head"]), ("Beta", "Price", "Specifications"))

    def test_bridge_and_controls(self):
        html, _ = table_page(1)
        pg = L.parse_page(html)
        r, aux = C.rank_all(pg, "What is the price of Beta?", explore=True)
        g = pg.tid2i["16"]
        self.assertEqual(r["fusion_all"][0], g)                  # both keys (row+col) -> rank 1
        self.assertGreater(r["bm25"].index(g), r["fusion_all"].index(g))
        self.assertEqual(sorted(r["fusion_shuf"]), sorted(r["fusion_all"]))
        self.assertTrue(all(aux["shuf_donor"][i] != i for i in aux["shuf_donor"]))
        for i, j in aux["qm_donor"].items():                       # q-matched donors overlap question
            if j is not None:
                self.assertNotEqual(i, j)
                self.assertTrue(set(L.tokenize("What is the price of Beta?")) & C.rel_tokens(pg, j)[2])
        d = C.diagnostics(pg, g, "What is the price of Beta?", aux)
        self.assertEqual((d["gold_bm25_zero"], d["bridge"], d["gold_keys"]), (1, 1, 2))

    def test_tieonly_never_inverts(self):
        html, qs = table_page(3)
        pg = L.parse_page(html)
        for q, _ in qs:
            r, aux = C.rank_all(pg, q)
            self.assertEqual([aux["sc"][i] for i in r["tieonly"]], [aux["sc"][i] for i in r["bm25"]])

    def test_pipeline_and_confirm_lock(self):
        out = Path(self.tmp.name) / "res"
        sp = out / "split.json"
        p = self.run_cli("split", "--train-data", str(self.root), "--out", str(sp))
        self.assertEqual(p.returncode, 0, p.stderr[-1500:])
        p = self.run_cli("run", "--data", str(self.root), "--role", "select", "--split-file", str(sp),
                         "--out", str(out / "sel"))
        self.assertEqual(p.returncode, 0, p.stderr[-1500:])
        s = json.loads((out / "sel" / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("exploration", s)
        self.assertEqual(s["descriptive"]["answer_in_gold"], 1.0)
        self.assertTrue(any(t["id"] == "H1" for t in s["prereg_tests"]))
        p = self.run_cli("run", "--data", str(self.root), "--role", "confirm", "--split-file", str(sp),
                         "--out", str(out / "conf"))
        self.assertEqual(p.returncode, 0, p.stderr[-1500:])
        self.assertNotIn("exploration", json.loads((out / "conf" / "summary.json").read_text(encoding="utf-8")))
        p = self.run_cli("run", "--data", str(self.root), "--role", "confirm", "--split-file", str(sp),
                         "--out", str(out / "conf2"))
        self.assertNotEqual(p.returncode, 0)                       # second confirm is refused
        self.assertIn("already run once", p.stdout + p.stderr)



class RelV2Tests(unittest.TestCase):
    def v2(self, html, tid):
        old = C.REL_VERSION
        C.REL_VERSION = "v2"
        try:
            pg = L.parse_page(html)
            return C.rel_parts(pg, pg.tid2i[tid]), C.rel_string(pg, pg.tid2i[tid])
        finally:
            C.REL_VERSION = old

    def v1(self, html, tid):
        pg = L.parse_page(html)
        return C.rel_parts(pg, pg.tid2i[tid])

    def test_A_bare_text_key(self):          # mo07-style: "Run Time - <span>117 min.</span>"
        html = ('<html><body tid="0"><ul tid="1"><li tid="2">Release Date - <span tid="3">Jun 25, 1982</span></li>'
                '<li tid="4">Run Time - <span tid="5">117 min.</span></li></ul></body></html>')
        self.assertEqual(self.v1(html, "5")["label"], "Release Date - Jun 25, 1982")  # v1 borrows the previous field
        self.assertEqual(self.v2(html, "5")[0]["label"], "Run Time -")

    def test_B_no_neighbour_value_leak(self):  # au09-style: previous block is KEY+VALUE of another field
        html = ('<html><body tid="0"><div tid="1"><div tid="2"><span tid="3">TRANSMISSION</span>'
                '<span tid="4">10-Speed Automatic</span></div><div tid="5">SEATING 5 Seats</div></div></body></html>')
        self.assertIn("10-Speed", self.v1(html, "5")["label"])        # v1 leaks the neighbour's value
        p, s = self.v2(html, "5")
        self.assertNotIn("10-Speed", s)
        self.assertEqual(self.v2(html, "4")[0]["label"], "TRANSMISSION")  # true key still found

    def test_C_rank_column_skipped(self):    # un11-style rank tables
        html = ('<html><body tid="0"><table tid="1"><tr tid="2"><th tid="3">Rank</th><th tid="4">University</th>'
                '<th tid="5">Score</th></tr><tr tid="6"><td tid="7">222</td><td tid="8">Univ Adelaide</td>'
                '<td tid="9">78.2</td></tr></table></body></html>')
        self.assertEqual(self.v1(html, "9")["row"], "222")
        p, _ = self.v2(html, "9")
        self.assertEqual((p["row"], p["col"]), ("Univ Adelaide", "Score"))

    def test_strong_key_beats_value(self):   # jo05-style "Hours: <value>"
        html = ('<html><body tid="0"><div tid="1"><span tid="2">Location:</span><span tid="3">Dublin</span>'
                '<span tid="4">Hours:</span><span tid="5">Full Time</span></div></body></html>')
        self.assertEqual(self.v2(html, "5")[0]["label"], "Hours:")

    def test_env_switch(self):
        code = "import c4_lib as C; print(C.REL_VERSION)"
        for v in ("v1", "v2"):
            env = dict(__import__("os").environ, C4_REL=v)
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(HERE), env=env)
            self.assertEqual(out.stdout.strip(), v)


if __name__ == "__main__":
    unittest.main(verbosity=2)
