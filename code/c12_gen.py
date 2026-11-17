# -*- coding: utf-8 -*-
"""
C12-F (optional): does the retrieval gain carry over to a small GENERATIVE reader?

Same evidence packing as C9 (64-word budget, nested duplicates skipped, random tie order), but the
reader is an instruction-tuned LLM that writes a short answer (greedy, max 16 new tokens).
Retrievers: bm25 | fusion_rel | fusion_shuf | oracle_gold. Scored with SQuAD EM/F1 against the
official answers; yes/no questions excluded (as in C9). Pre-registered in PREREG_C12.md (F).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import c4_lib as C
import c9_common as S
import c9_qa as Q
import idc_lib as L

RETR = ("bm25", "fusion_rel", "fusion_shuf")
PROMPT = ("Answer the question using only the context. Reply with the shortest exact phrase from the "
          "context, nothing else.\n\nContext: {ctx}\n\nQuestion: {q}\nAnswer:")


class GenReader:
    def __init__(self, name, device, max_new=16, batch=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        self.device, self.max_new, self.batch = device, max_new, batch

    def _prompt(self, q, ctx):
        msg = [{"role": "user", "content": PROMPT.format(ctx=ctx or ".", q=q)}]
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        return msg[0]["content"]

    def answer(self, pairs):
        out = []
        for s in range(0, len(pairs), self.batch):
            chunk = pairs[s:s + self.batch]
            enc = self.tok([self._prompt(q, c) for q, c in chunk], return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            out += [t.strip().split("\n")[0].strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--role", required=True, choices=["dev", "select", "confirm", "test"])
    ap.add_argument("--split-file")
    ap.add_argument("--sites")
    ap.add_argument("--per-site", type=int, default=100)
    ap.add_argument("--max-per-page", type=int, default=3)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.role in ("select", "confirm"):
        if not a.split_file:
            sys.exit("--split-file required")
        sites = set(json.loads(Path(a.split_file).read_text(encoding="utf-8"))[a.role])
    else:
        sites = set(a.sites.split(",")) if a.sites else None

    rows, _ = S.load_diverse(a.data, sites, a.per_site, a.max_per_page)
    by_page = OrderedDict()
    for r in rows:
        by_page.setdefault(r.page_key, []).append(r)
    items = []
    for pk, qs in by_page.items():
        pg = L.parse_page(qs[0].html_path.read_text(encoding="utf-8", errors="replace"))
        cands = [i for i in range(len(pg.els)) if pg.key[i]]
        for r in qs:
            if r.answer.strip().lower() in ("yes", "no"):
                continue
            g = pg.tid2i.get(r.element_id)
            if g is None or not pg.key[g]:
                continue
            _, aux = C.rank_all(pg, r.question, seed=a.seed)
            sc, cs, donor = aux["sc"], aux["cs"], aux["shuf_donor"]
            score = {"bm25": sc, "fusion_rel": {i: sc[i] + C.LAMBDA * cs[i] for i in cands},
                     "fusion_shuf": {i: sc[i] + C.LAMBDA * cs[donor[i]] for i in cands}}
            ctx = {m: Q.pack(pg, Q.rand_order(score[m], cands, f"{a.seed}|{r.qid}"), a.budget)[1] for m in RETR}
            ctx["oracle_gold"] = pg.text[g]
            items.append({"qid": r.qid, "site": r.site, "page": pk, "q": r.question, "answer": r.answer, "ctx": ctx})
        del pg
    print(f"[prep] {len(items)} span questions")

    reader = GenReader(a.model, a.device, batch=a.batch)
    uniq = list(OrderedDict.fromkeys((it["q"], c) for it in items for c in it["ctx"].values()))
    t0 = time.time()
    preds = dict(zip(uniq, reader.answer(uniq)))
    print(f"[gen] {len(uniq)} prompts in {time.time() - t0:.0f}s with {a.model}")

    per_q, recs = {}, []
    for it in items:
        mm = {"_diag": {"page": it["page"]}}
        for m, c in it["ctx"].items():
            p = preds[(it["q"], c)]
            mm[m] = {"em": Q.em(p, it["answer"]), "f1": Q.f1(p, it["answer"])}
            recs.append({"qid": it["qid"], "site": it["site"], "method": m, "pred": p, "answer": it["answer"], **mm[m]})
        per_q[(it["qid"], it["site"])] = mm
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gen_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    table = {m: {k: round(C.mean(r[k] for r in recs if r["method"] == m), 4) for k in ("em", "f1")}
             for m in OrderedDict.fromkeys(r["method"] for r in recs)}
    tests = []
    for tid, A, B in (("F-1", "fusion_rel", "bm25"), ("F-2", "fusion_rel", "fusion_shuf")):
        t = C.paired(per_q, A, B, "f1")
        diffs = [(s, mm["_diag"]["page"], mm[A]["f1"] - mm[B]["f1"]) for (q, s), mm in per_q.items()]
        _, lo, hi = S.two_level_boot(diffs)
        t.update({"id": tid, "page_ci95": [round(lo, 4), round(hi, 4)]})
        tests.append(t)
    summ = {"experiment": "C12-F generative reader", "role": a.role, "model": a.model, "budget": a.budget,
            "n": len(items), "methods": table, "tests": tests}
    (out / "gen_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== C12-F [{a.role}] n={len(items)} reader={a.model} ===")
    for m, v in table.items():
        print(f"   {m:14s} EM {v['em']:.3f}  F1 {v['f1']:.3f}")
    for t in tests:
        print(f"{t['id']:5s} {t['A']} - {t['B']} [f1] n={t['n']} diff={t['diff'] * 100:+.1f}%p "
              f"CI[{t['ci95'][0] * 100:+.1f},{t['ci95'][1] * 100:+.1f}] pageCI[{t['page_ci95'][0] * 100:+.1f},{t['page_ci95'][1] * 100:+.1f}] "
              f"sites+/-={t['sites_pos_neg'][0]}/{t['sites_pos_neg'][1]}")
    print(f"[saved] {out.resolve()}")


if __name__ == "__main__":
    main()
