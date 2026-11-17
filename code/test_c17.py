# -*- coding: utf-8 -*-
"""Offline tests: a tiny random MarkupLM and a tiny BPE tokenizer are built on the fly (no download)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import c17_markuplm as M
import idc_lib as L
import test_c4 as T4

HERE = Path(__file__).resolve().parent
HTML = """<html><body tid="0"><h2 tid="1">Bestsellers</h2>
<table tid="2"><tr tid="3"><th tid="4">Title</th><th tid="5">Price</th></tr>
<tr tid="6"><td tid="7">Book A</td><td tid="8"><b tid="9">$32.99</b> USD</td></tr></table>
<script>var x = 1;</script></body></html>"""


def tiny_model(d):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import MarkupLMConfig, MarkupLMForQuestionAnswering, MarkupLMTokenizerFast
    bt = ByteLevelBPETokenizer()
    bt.train_from_iterator(["What is the price of Book A ? $32.99 USD Title Price Alpha Beta Gamma 150 200"] * 50,
                           vocab_size=400, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bt.save_model(d)
    vocab = json.load(open(f"{d}/vocab.json", encoding="utf-8"))
    merges = [tuple(l.split()) for l in open(f"{d}/merges.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
    tags = {t: i for i, t in enumerate(["html", "body", "div", "span", "table", "tr", "td", "th", "h2", "p", "a", "li",
                                        "ul", "b", "thead", "tbody", "dl", "dt", "dd", "label", "section"])}
    try:
        tok = MarkupLMTokenizerFast(tags_dict=tags, vocab=vocab, merges=merges)
    except TypeError:                                          # older transformers: file-based constructor
        tok = MarkupLMTokenizerFast(vocab_file=f"{d}/vocab.json", merges_file=f"{d}/merges.txt", tags_dict=tags)
    tok.save_pretrained(d)
    cfg = MarkupLMConfig(vocab_size=len(vocab) + 10, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=520, max_depth=50, pad_token_id=1)
    MarkupLMForQuestionAnswering(cfg).save_pretrained(d)


class C17Tests(unittest.TestCase):
    def setUp(self):
        self.pg = L.parse_page(HTML)

    def test_strings_xpath_owner(self):
        st = M.page_strings(self.pg)
        texts = [s[0] for s in st]
        self.assertNotIn("var x = 1;", texts)                    # script skipped
        self.assertIn("$32.99", texts)
        s = next(s for s in st if s[0] == "$32.99")
        self.assertEqual(s[1], "/html/body/table/tr[2]/td[2]/b")
        self.assertEqual(s[2], self.pg.tid2i["9"])
        usd = next(s for s in st if s[0] == "USD")
        self.assertEqual(usd[2], self.pg.tid2i["8"])

    def test_lca(self):
        self.assertEqual(M.lca(self.pg, self.pg.tid2i["9"], self.pg.tid2i["8"]), self.pg.tid2i["8"])
        self.assertEqual(M.lca(self.pg, self.pg.tid2i["7"], self.pg.tid2i["9"]), self.pg.tid2i["6"])

    def test_rrf_ties_share_rank(self):
        f = M.rrf({1: 1.0, 2: 1.0, 3: 0.0}, {1: 0.0, 2: 0.0, 3: 0.0}, cands=[1, 2, 3])
        self.assertEqual(f[1], f[2])
        self.assertGreater(f[1], f[3])

    def test_end_to_end_tiny_model(self):
        with tempfile.TemporaryDirectory() as d:
            mdir = Path(d) / "tiny"
            mdir.mkdir()
            tiny_model(str(mdir))
            root = Path(d) / "official_train"
            T4.make_data(root)
            p = subprocess.run([sys.executable, str(HERE / "c17_markuplm.py"), "--data", str(root), "--role", "dev",
                                "--model", str(mdir), "--device", "cpu", "--out", str(Path(d) / "c17")],
                               capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            s = json.loads((Path(d) / "c17" / "c17_summary.json").read_text(encoding="utf-8"))
            self.assertTrue({"bm25_text", "bm25_own", "markuplm", "rrf_markuplm_own"} <= set(s["methods"]))
            self.assertTrue({"L1", "L2"} <= {t["id"] for t in s["tests"]})


if __name__ == "__main__":
    unittest.main()
