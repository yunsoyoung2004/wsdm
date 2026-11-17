# -*- coding: utf-8 -*-
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import c12_lib as X
import c4_lib as C
import idc_lib as L
import test_c4 as T4

HERE = Path(__file__).resolve().parent

EXT_HTML = """<html><body tid="0">
<h2 tid="1">Phones</h2>
<table tid="2"><caption tid="3">Spec sheet</caption>
<tr tid="4"><th tid="5">Model</th><th tid="6">Battery</th></tr>
<tr tid="7"><td tid="8">Nova</td><td tid="9"><span tid="10" itemprop="batteryCapacity" title="Battery life">4000 mAh</span></td></tr>
</table>
<dl tid="11"><dt tid="12">Weight</dt><dd tid="13">180 g</dd></dl>
<label tid="14" for="clr">Colour</label><span tid="15" id="clr">Blue</span>
<figure tid="16"><img tid="17" alt="Front camera" src="x.png"/><figcaption tid="18">Camera module</figcaption></figure>
</body></html>"""


class C12Tests(unittest.TestCase):
    def test_bridge_tokens_and_audit_on_table(self):
        html, qs = T4.table_page(1)
        pg = L.parse_page(html)
        q, tid = qs[2]                                   # "What is the price of Beta?"
        g = pg.tid2i[str(tid)]
        bt = X.bridge_tokens(q, pg.text[g])
        self.assertEqual(bt, {"price", "beta"})
        a = X.audit(pg, g, q)
        self.assertEqual(a["in_row"], 1)
        self.assertEqual(a["in_col"], 1)
        self.assertEqual(a["in_rules"], 1)
        self.assertEqual(a["nowhere"], 0)
        self.assertEqual(a["in_same_row"], 1)

    def test_extended_relations_are_found(self):
        pg = L.parse_page(EXT_HTML)
        i = pg.tid2i["10"]
        self.assertIn("Spec sheet", X.ctx_caption(pg, i))
        self.assertIn("batteryCapacity", X.ctx_attrs(pg, i, X.EXT_ATTR_KEYS))
        self.assertIn("battery", X.toks(X.ctx_attrs(pg, i, X.EXT_ATTR_KEYS)))   # camelCase split
        self.assertEqual(X.ctx_dtdd(pg, pg.tid2i["13"]), "Weight")
        self.assertEqual(X.ctx_label_for(pg, pg.tid2i["15"]), "Colour")
        self.assertIn("Camera module", X.ctx_figlegend(pg, pg.tid2i["17"]))
        ext = X.ext_string(pg, i)
        self.assertIn("Spec sheet", ext)
        self.assertLessEqual(len(ext.split()), X.EXT_CAP)

    def test_ablation_strings(self):
        html, qs = T4.table_page(1)
        pg = L.parse_page(html)
        i = pg.tid2i[str(qs[2][1])]
        full = C.rel_string(pg, i)
        self.assertIn("Beta", full)
        self.assertIn("Price", full)
        self.assertEqual(X.ABLATIONS["only_row"](pg, i), "Beta")
        self.assertNotIn("Price", X.ABLATIONS["minus_col"](pg, i))
        self.assertIn("Price", X.ABLATIONS["only_col"](pg, i))
        self.assertIn("Contact", X.rel_parent_string(pg, pg.tid2i["9"]) + " Contact")

    def test_chunks_respect_budget_and_overlap(self):
        html, _ = T4.table_page(3)
        pg = L.parse_page(html)
        cs = X.chunks(pg, size=8, stride=4)
        self.assertGreater(len(cs), 1)
        self.assertEqual(cs[0].split()[4:8], cs[1].split()[:4])
        top = X.chunk_top(pg, "What is the price of Beta?", budget=10, size=8, stride=4)
        self.assertLessEqual(len(top.replace("|", " ").split()), 10)

    def test_perturb_is_gold_free_and_avoids_page_words(self):
        syn = {"price": ["price", "cost", "terms"], "heavy": ["weighty"]}.get
        q = "What is the price of Beta?"
        out, n = X.perturb(q, vocab={"price", "beta"}, page_tokens={"price", "beta"}, synonyms=lambda w: syn(w) or [])
        self.assertEqual(out, "What is the cost of Beta?")
        self.assertEqual(n, 1)
        out2, n2 = X.perturb(q, vocab={"price"}, page_tokens={"price", "cost"}, synonyms=lambda w: syn(w) or [])
        self.assertEqual(out2, "What is the terms of Beta?")
        out3, n3 = X.perturb("How heavy is Beta?", vocab={"beta"}, page_tokens=set(), synonyms=lambda w: syn(w) or [])
        self.assertEqual((out3, n3), ("How heavy is Beta?", 0))   # not in relation vocab -> untouched

    def test_end_to_end_bm25_parts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "official_train"
            T4.make_data(root)
            p = subprocess.run([sys.executable, str(HERE / "c12_run.py"), "--data", str(root), "--role", "dev",
                                "--out", str(Path(d) / "c12")], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            s = json.loads((Path(d) / "c12" / "c12_summary.json").read_text(encoding="utf-8"))
            ids = {t["id"] for t in s["tests"]}
            self.assertTrue({"C-1_row", "C-1_col", "C-2_head", "C-3", "B-2", "REF", "G-1"} <= ids)
            self.assertGreater(s["audit_A"]["n"], 0)
            self.assertIn("in_rules", s["audit_A"]["among_questions_with_bridge_tokens"])
            self.assertTrue(all("holm_p" in t for t in s["tests"] if t["id"].startswith("C-1")))
            self.assertIn("G_chunk", s["chunk_G_ans_in_ctx"])
            self.assertEqual(set(s["types_H"]), set(X.TYPES))
            self.assertIn("H_m_row_header", ids)
            self.assertIn("H_s_parent|present", ids)
            self.assertIn("row_header_in_parent", s["overlap_H"])

    def test_one_type_contexts(self):
        html, qs = T4.table_page(1)
        pg = L.parse_page(html)
        i = pg.tid2i[str(qs[2][1])]                       # Beta price cell
        self.assertEqual(X.type_text(pg, "m_row_header", i), "Beta")
        self.assertEqual(X.type_text(pg, "m_col_header", i), "Price")
        self.assertEqual(X.type_text(pg, "m_heading", i), "Specifications")
        self.assertEqual(X.type_text(pg, "s_prev_sibling", i), "Beta")
        self.assertIn("Beta", X.type_text(pg, "s_parent", i))
        self.assertIn("td", X.type_text(pg, "s_tag_path", i))
        self.assertNotIn("$", X.type_text(pg, "s_tag_path", i))
        for name in X.TYPES:
            self.assertLessEqual(len(X.type_text(pg, name, i).split()), X.TYPE_CAP)
        par = X.type_text(pg, "s_parent_minus_sem", i)
        self.assertNotIn("Beta", par)                      # semantic part (row header) removed
        pg2 = L.parse_page(EXT_HTML)
        self.assertEqual(X.type_text(pg2, "m_dt_dd", pg2.tid2i["13"]), "Weight")
        self.assertEqual(X.type_text(pg2, "m_caption", pg2.tid2i["10"]), "Spec sheet")

    def test_dense_and_perturb_plumbing_with_stub_models(self):
        """Fake sentence-transformers + WordNet on PYTHONPATH: checks the D/E code paths end to end."""
        stub_st = (
            "import numpy as np, hashlib\n"
            "class SentenceTransformer:\n"
            "    def __init__(self, name, device=None): self.max_seq_length = 0\n"
            "    def encode(self, texts, batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):\n"
            "        out = []\n"
            "        for t in texts:\n"
            "            v = np.zeros(64)\n"
            "            for w in t.lower().split():\n"
            "                v[int(hashlib.md5(w.strip('?.,|').encode()).hexdigest(), 16) % 64] += 1\n"
            "            n = np.linalg.norm(v); out.append(v / n if n else v)\n"
            "        return np.array(out)\n")
        stub_wn = (
            "class _L:\n"
            "    def __init__(s, n): s._n = n\n"
            "    def name(s): return s._n\n"
            "class _S:\n"
            "    def __init__(s, ns): s._ns = ns\n"
            "    def lemmas(s): return [_L(n) for n in s._ns]\n"
            "SYN = {'price': ['price', 'cost'], 'weight': ['weight', 'heaviness'], 'model': ['model', 'version']}\n"
            "class wordnet:\n"
            "    @staticmethod\n"
            "    def synsets(w): return [_S(SYN[w])] if w in SYN else []\n")
        with tempfile.TemporaryDirectory() as d:
            stubs = Path(d) / "stubs"
            (stubs / "sentence_transformers").mkdir(parents=True)
            (stubs / "sentence_transformers" / "__init__.py").write_text(stub_st, encoding="utf-8")
            (stubs / "nltk" / "corpus").mkdir(parents=True)
            (stubs / "nltk" / "__init__.py").write_text("", encoding="utf-8")
            (stubs / "nltk" / "corpus" / "__init__.py").write_text(stub_wn, encoding="utf-8")
            root = Path(d) / "official_train"
            T4.make_data(root, n_sites=4)
            env = {**__import__("os").environ, "PYTHONPATH": str(stubs)}
            p = subprocess.run([sys.executable, str(HERE / "c12_run.py"), "--data", str(root), "--role", "dev",
                                "--dense", "stub-bge", "--device", "cpu", "--perturb", "--out", str(Path(d) / "c12")],
                               capture_output=True, text=True, encoding="utf-8", env=env)
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            s = json.loads((Path(d) / "c12" / "c12_summary.json").read_text(encoding="utf-8"))
            ids = {t["id"] for t in s["tests"]}
            self.assertTrue({"D-1", "D-2", "D-2s", "D-4", "D-5", "D-6", "E-2", "E-2p", "E-3"} <= ids)
            self.assertGreater(s["perturb_E"]["share_perturbed"], 0)
            rows = (Path(d) / "c12" / "c12_perturb.csv").read_text(encoding="utf-8")
            self.assertIn("cost", rows)

    def test_collect_report(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "official_train"
            T4.make_data(root, n_sites=4)
            res = Path(d) / "results" / "c12"
            for s in ("dev", "test"):
                p = subprocess.run([sys.executable, str(HERE / "c12_run.py"), "--data", str(root), "--role", "dev",
                                    "--out", str(res / s)], capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            p = subprocess.run([sys.executable, str(HERE / "c12_collect.py"), "--root", str(res)],
                               capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            rep = (res / "C12_report.md").read_text(encoding="utf-8")
            for frag in ("G-1", "C-1", "H-1", "s_parent_minus_sem", "m_row_header", "in_anywhere"):
                self.assertIn(frag, rep)

    def test_generative_reader_pipeline_with_stub(self):
        import c12_gen as G

        class Fake:
            def __init__(self, *a, **k):
                pass

            def answer(self, pairs):          # echo the first context segment
                return [c.split(" | ")[0] for _, c in pairs]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "official_train"
            T4.make_data(root, n_sites=4)
            old, G.GenReader = G.GenReader, Fake
            old_argv = sys.argv
            try:
                sys.argv = ["c12_gen.py", "--data", str(root), "--role", "dev", "--device", "cpu", "--out", str(Path(d) / "gen")]
                G.main()
            finally:
                G.GenReader, sys.argv = old, old_argv
            s = json.loads((Path(d) / "gen" / "gen_summary.json").read_text(encoding="utf-8"))
            self.assertEqual({t["id"] for t in s["tests"]}, {"F-1", "F-2"})
            self.assertEqual(s["methods"]["oracle_gold"]["f1"], 1.0)

    @unittest.skipUnless(importlib.util.find_spec("sentence_transformers"), "sentence-transformers not installed")
    def test_dense_mode_runs_cpu(self):  # pragma: no cover - optional dependency
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "official_train"
            T4.make_data(root, n_sites=3)
            p = subprocess.run([sys.executable, str(HERE / "c12_run.py"), "--data", str(root), "--role", "dev",
                                "--dense", "sentence-transformers/all-MiniLM-L6-v2", "--device", "cpu",
                                "--out", str(Path(d) / "c12")], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            s = json.loads((Path(d) / "c12" / "c12_summary.json").read_text(encoding="utf-8"))
            self.assertIn("D-1", {t["id"] for t in s["tests"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
