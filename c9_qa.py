# -*- coding: utf-8 -*-
"""
C9 / E2: does better evidence retrieval give better ANSWERS? (official WebSRC answers, EM/F1)

For each question, each retriever ranks DOM nodes (GT-free). The top nodes are packed into a
fixed WORD budget (nested duplicates skipped so text is never repeated), then an off-the-shelf
extractive QA model (no fine-tuning -> no leakage) reads the context and extracts a span.

Retrievers: bm25 | fusion_rel | fusion_shuf (control) | fusion_parent_64 (strongest baseline, C8)
            oracle_gold (gold node text only = reader ceiling)
Ties are broken by a fixed-seed random permutation (no document-order advantage).
Yes/no questions are excluded (not extractable) and counted.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import string
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import c4_lib as C
import c8_robust as R
import c9_common as S
import idc_lib as L

RETRIEVERS = ("bm25", "fusion_rel", "fusion_shuf", "fusion_parent_64")


# ------------------------------------------------------------------ SQuAD-style metrics
def normalize(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, gold):
    return int(normalize(pred) == normalize(gold))


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


# ------------------------------------------------------------------ context packing (GT-free)
def pack(pg, order, budget_words, max_nodes=30):
    chosen, words = [], 0
    for i in order[:max_nodes]:
        if any(L.is_ancestor(pg, j, i) or L.is_ancestor(pg, i, j) for j in chosen):
            continue  # nested duplicate text
        toks = pg.text[i].split()
        if not toks:
            continue
        room = budget_words - words
        if room <= 0:
            break
        chosen.append(i)
        words += min(len(toks), room)
    parts, used = [], 0
    for i in chosen:
        toks = pg.text[i].split()
        take = toks[:max(0, budget_words - used)]
        used += len(take)
        parts.append(" ".join(take))
    return chosen, " | ".join(parts)


def rand_order(score, cands, seed):
    rng = random.Random(seed)
    perm = {i: rng.random() for i in cands}
    return sorted(cands, key=lambda i: (-score[i], perm[i]))


# ------------------------------------------------------------------ extractive QA
class Reader:
    def __init__(self, name, device, max_len=384, stride=128, batch=32, max_answer_tokens=30):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(name).to(device).eval()
        self.device, self.max_len, self.stride, self.batch, self.mat = device, max_len, stride, batch, max_answer_tokens

    def answer(self, pairs):
        """pairs: list of (question, context) -> list of answer strings (always non-null)."""
        torch = self.torch
        out = [""] * len(pairs)
        best = [-1e30] * len(pairs)
        for s in range(0, len(pairs), self.batch):
            chunk = pairs[s:s + self.batch]
            qs = [q for q, _ in chunk]
            cs = [c if c else "." for _, c in chunk]
            enc = self.tok(qs, cs, truncation="only_second", max_length=self.max_len, stride=self.stride,
                           return_overflowing_tokens=True, return_offsets_mapping=True, padding=True,
                           return_tensors="pt")
            mapping = enc.pop("overflow_to_sample_mapping").tolist()
            offsets = enc.pop("offset_mapping").tolist()
            seq_ids = [enc.sequence_ids(k) for k in range(len(mapping))]
            with torch.no_grad():
                res = self.model(**{k: v.to(self.device) for k, v in enc.items()})
            st, en = res.start_logits.float().cpu().tolist(), res.end_logits.float().cpu().tolist()
            for f, smp in enumerate(mapping):
                idx = s + smp
                ctx_ix = [t for t, sid in enumerate(seq_ids[f]) if sid == 1]
                if not ctx_ix:
                    continue
                top_s = sorted(ctx_ix, key=lambda t: -st[f][t])[:20]
                top_e = sorted(ctx_ix, key=lambda t: -en[f][t])[:20]
                for a in top_s:
                    for b in top_e:
                        if b < a or b - a + 1 > self.mat:
                            continue
                        sc = st[f][a] + en[f][b]
                        if sc > best[idx]:
                            best[idx] = sc
                            out[idx] = cs[smp][offsets[f][a][0]:offsets[f][b][1]]
        return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--role", required=True, choices=["dev", "select", "confirm", "test"])
    ap.add_argument("--split-file")
    ap.add_argument("--sites")
    ap.add_argument("--per-site", type=int, default=100)
    ap.add_argument("--max-per-page", type=int, default=3)
    ap.add_argument("--first-n", action="store_true", help="use first-N sampling instead of diverse pages")
    ap.add_argument("--budgets", default="64,256", help="context word budgets")
    ap.add_argument("--model", default="deepset/roberta-base-squad2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.role in ("select", "confirm"):
        if not a.split_file:
            sys.exit("--split-file required")
        sites = set(json.loads(Path(a.split_file).read_text(encoding="utf-8"))[a.role])
    else:
        sites = set(a.sites.split(",")) if a.sites else None
    budgets = [int(x) for x in a.budgets.split(",")]
    if a.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            sys.exit("CUDA not available")
        print("[gpu]", torch.cuda.get_device_name(0))

    if a.first_n:
        rows, _, _ = L.load_rows(Path(a.data), sites, None, a.per_site)
    else:
        rows, _ = S.load_diverse(a.data, sites, a.per_site, a.max_per_page)
    by_page = OrderedDict()
    for r in rows:
        by_page.setdefault(r.page_key, []).append(r)

    items, n_yesno = [], 0
    t0 = time.time()
    for pk, qs in by_page.items():
        pg = L.parse_page(qs[0].html_path.read_text(encoding="utf-8", errors="replace"))
        cands = [i for i in range(len(pg.els)) if pg.key[i]]
        for r in qs:
            if r.answer.strip().lower() in ("yes", "no"):
                n_yesno += 1
                continue
            g = pg.tid2i.get(r.element_id)
            if g is None or not pg.key[g]:
                continue
            q = L.tokenize(r.question)
            _, aux = C.rank_all(pg, r.question, seed=a.seed)  # GT-free
            sc, cs, donor = aux["sc"], aux["cs"], aux["shuf_donor"]
            score = {"bm25": sc,
                     "fusion_rel": {i: sc[i] + C.LAMBDA * cs[i] for i in cands},
                     "fusion_shuf": {i: sc[i] + C.LAMBDA * cs[donor[i]] for i in cands},
                     "fusion_parent_64": {i: sc[i] + C.LAMBDA * C.bm25_counter(pg, q, R.ctx_counter(pg, "parent", i, 64))
                                          for i in cands}}
            gold_cover = lambda chosen: int(any(j == g or L.is_ancestor(pg, j, g) for j in chosen))
            it = {"qid": r.qid, "site": r.site, "page": pk, "q": r.question, "answer": r.answer, "ctx": {}}
            for m in RETRIEVERS:
                order = rand_order(score[m], cands, f"{a.seed}|{r.qid}")
                for b in budgets:
                    chosen, text = pack(pg, order, b)
                    it["ctx"][(m, b)] = {"text": text, "gold_in_ctx": gold_cover(chosen),
                                         "ans_in_ctx": int(normalize(r.answer) in normalize(text)), "n_nodes": len(chosen)}
            it["ctx"][("oracle_gold", 0)] = {"text": pg.text[g], "gold_in_ctx": 1,
                                             "ans_in_ctx": int(normalize(r.answer) in normalize(pg.text[g])), "n_nodes": 1}
            items.append(it)
        del pg
    print(f"[prep] {len(items)} span questions ({n_yesno} yes/no excluded) in {time.time() - t0:.0f}s")

    reader = Reader(a.model, a.device, batch=a.batch)
    uniq = OrderedDict()
    for it in items:
        for c in it["ctx"].values():
            uniq[(it["q"], c["text"])] = None
    keys = list(uniq)
    print(f"[qa] {len(keys)} unique (question, context) pairs with {a.model}")
    t0 = time.time()
    preds = dict(zip(keys, reader.answer(keys)))
    print(f"[qa] done in {time.time() - t0:.0f}s")

    recs, per_q = [], {}
    for it in items:
        mm = {"_diag": {"page": it["page"]}}
        for (m, b), c in it["ctx"].items():
            pred = preds[(it["q"], c["text"])]
            name = m if m == "oracle_gold" else f"{m}@{b}"
            row = {"em": em(pred, it["answer"]), "f1": f1(pred, it["answer"]), **{k: c[k] for k in ("gold_in_ctx", "ans_in_ctx", "n_nodes")}}
            mm[name] = row
            recs.append({"qid": it["qid"], "site": it["site"], "method": name, "pred": pred, "answer": it["answer"], **row})
        per_q[(it["qid"], it["site"])] = mm

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "qa_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    methods = list(OrderedDict.fromkeys(r["method"] for r in recs))
    table = {m: {k: round(C.mean(r[k] for r in recs if r["method"] == m), 4) for k in ("em", "f1", "gold_in_ctx", "ans_in_ctx", "n_nodes")}
             for m in methods}
    tests = []
    for b in budgets:
        for tid, A, B in (("Q1", "fusion_rel", "bm25"), ("Q2", "fusion_rel", "fusion_shuf"), ("Q3", "fusion_rel", "fusion_parent_64")):
            for met in ("f1", "em"):
                t = C.paired(per_q, f"{A}@{b}", f"{B}@{b}", met)
                if not t:
                    continue
                diffs = [(site, mm["_diag"]["page"], mm[f"{A}@{b}"][met] - mm[f"{B}@{b}"][met]) for (q, site), mm in per_q.items()]
                _, lo2, hi2 = S.two_level_boot(diffs)
                t.update({"id": f"{tid}@{b}", "page_ci95": [round(lo2, 4), round(hi2, 4)]})
                tests.append(t)
    summ = {"experiment": "C9/E2 extractive QA on retrieved evidence (official answers)", "role": a.role,
            "rel_version": C.REL_VERSION, "reader": a.model, "budgets_words": budgets,
            "sample": "first-N" if a.first_n else f"diverse(<= {a.max_per_page}/page)",
            "n_span_questions": len(items), "n_yesno_excluded": n_yesno,
            "n_pages": len({it["page"] for it in items}), "sites": sorted({it["site"] for it in items}),
            "methods": table, "tests": tests}
    (out / "qa_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== C9/E2 QA [{a.role}] n={len(items)} pages={summ['n_pages']} reader={a.model} ===")
    print(f"{'method':24s}{'EM':>8s}{'F1':>8s}{'gold∈ctx':>10s}{'ans∈ctx':>9s}{'nodes':>7s}")
    for m, v in table.items():
        print(f"{m:24s}{v['em']:8.3f}{v['f1']:8.3f}{v['gold_in_ctx']:10.3f}{v['ans_in_ctx']:9.3f}{v['n_nodes']:7.1f}")
    print("\n--- paired tests (site CI and site->page CI) ---")
    for t in tests:
        print(f"{t['id']:7s} {t['A']} - {t['B']} [{t['metric']}] n={t['n']} diff={t['diff'] * 100:+.1f}%p "
              f"CI[{t['ci95'][0] * 100:+.1f},{t['ci95'][1] * 100:+.1f}] pageCI[{t['page_ci95'][0] * 100:+.1f},{t['page_ci95'][1] * 100:+.1f}] "
              f"wins {t['wins'][0]}:{t['wins'][1]} sites+/-={t['sites_pos_neg'][0]}/{t['sites_pos_neg'][1]}")
    print(f"[saved] {out.resolve()}")


if __name__ == "__main__":
    main()
