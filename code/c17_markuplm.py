# -*- coding: utf-8 -*-
"""
C17: a learned structure-aware reader as a node ranker (pre-registered in PREREG_C17.md).

  MarkupLM fine-tuned on WebSRC (microsoft/markuplm-base-finetuned-websrc) reads the page with XPath
  embeddings and predicts an answer span. We turn it into a node ranking:
    - text strings of the page are taken in document order with their XPath (same rule as the
      Hugging Face MarkupLM feature extractor; script/style/noscript skipped);
    - strings are packed into windows that fit 512 tokens with the question (overlapping windows);
    - for every start token the best end token within MAX_SPAN tokens gives one span score
      start_logit + end_logit; the span is assigned to the deepest tid node that contains both
      its first and its last string (lowest common tid ancestor);
    - node score = best span score assigned to it; nodes with no span get -inf (random ties).
  Compared on the SAME questions with BM25 text only and BM25 + own semantic relations, plus a
  reciprocal-rank fusion of MarkupLM and BM25 + own relations.

  Only dev and the official Test are fair: MarkupLM was fine-tuned on the WebSRC training split,
  which contains our train/select/confirm sites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import c4_lib as C
import c8_robust as R
import c9_common as S
import idc_lib as L

MODEL = "microsoft/markuplm-base-finetuned-websrc"
MAX_LEN, MAX_SPAN, STRIDE_FRAC, RRF_K = 512, 30, 0.5, 60
SKIP = {"script", "style", "noscript", "template"}
NEG = -1e9


def code_hash():
    h = hashlib.sha256()
    for f in ("c17_markuplm.py", "c4_lib.py", "idc_lib.py", "c8_robust.py"):
        p = Path(__file__).with_name(f)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ page -> strings with xpath and tid owner
def xpath_of(string_el):
    tags, subs = [], []
    child = string_el.parent
    for parent in child.parents:
        siblings = parent.find_all(child.name, recursive=False)
        tags.append(child.name)
        subs.append(0 if len(siblings) == 1 else next(i for i, s in enumerate(siblings, 1) if s is child))
        child = parent
    tags.reverse()
    subs.reverse()
    return "".join(f"/{t}" + (f"[{s}]" if s else "") for t, s in zip(tags, subs))


def page_strings(pg):
    """[(text, xpath, owner_index)] in document order; owner = deepest tid element containing the string."""
    from bs4 import NavigableString, Comment, Tag
    root = pg.els[0]
    while root.parent is not None:
        root = root.parent
    idx = {id(e): i for i, e in enumerate(pg.els)}
    out = []
    for el in root.descendants:
        if not isinstance(el, NavigableString) or isinstance(el, Comment) or not isinstance(el.parent, Tag):
            continue
        if el.parent.name in SKIP or any(p.name in SKIP for p in el.parents if isinstance(p, Tag)):
            continue
        t = " ".join(str(el).split())
        if not t:
            continue
        owner, p = -1, el.parent
        while p is not None:
            if id(p) in idx:
                owner = idx[id(p)]
                break
            p = p.parent
        if owner >= 0:
            out.append((t, xpath_of(el), owner))
    return out


def lca(pg, a, b):
    anc = set()
    x = a
    while x >= 0:
        anc.add(x)
        x = pg.parent[x]
    x = b
    while x >= 0 and x not in anc:
        x = pg.parent[x]
    return x


def string_lens(tok, strings):
    return [max(1, len(tok.tokenize(s[0]))) for s in strings]


def windows(tok, question, strings, lens=None):
    """Greedy windows of whole strings that fit MAX_LEN with the question; consecutive windows overlap."""
    q_len = len(tok.tokenize(question)) + 4
    budget = max(32, MAX_LEN - q_len)
    lens = lens if lens is not None else string_lens(tok, strings)
    out, start = [], 0
    while start < len(strings):
        end, used = start, 0
        while end < len(strings) and (used + lens[end] <= budget or end == start):
            used += lens[end]
            end += 1
        out.append((start, end))
        if end >= len(strings):
            break
        nxt = start + max(1, int((end - start) * STRIDE_FRAC))
        start = nxt if nxt > start else end
    return out


def markuplm_scores(tok, model, device, pg, question, strings, cands, batch=8, lens=None):
    """Returns ({node: best span score}, number of windows)."""
    import torch
    best = {i: NEG for i in cands}
    if not strings:
        return best, 0
    wins = windows(tok, question, strings, lens)
    cand = set(cands)
    for b0 in range(0, len(wins), batch):
        chunk = wins[b0:b0 + batch]
        nodes = [[s[0] for s in strings[a:e]] for a, e in chunk]
        xps = [[s[1] for s in strings[a:e]] for a, e in chunk]
        enc = tok([question] * len(chunk), nodes, xpaths=xps, truncation="only_second", max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        word_ids = [enc.word_ids(k) for k in range(len(chunk))]
        seq_ids = [enc.sequence_ids(k) for k in range(len(chunk))]
        with torch.no_grad():
            o = model(**{k: v.to(device) for k, v in enc.items()})
        st, en = o.start_logits.float().cpu().numpy(), o.end_logits.float().cpu().numpy()
        for k, (a, _) in enumerate(chunk):
            ctx = [t for t, sid in enumerate(seq_ids[k]) if sid == 1 and word_ids[k][t] is not None]
            if not ctx:
                continue
            pos = {t: n for n, t in enumerate(ctx)}
            owner = [strings[a + word_ids[k][t]][2] for t in ctx]
            s_ctx, e_ctx = st[k][ctx], en[k][ctx]
            for n in range(len(ctx)):
                hi = min(len(ctx), n + MAX_SPAN)
                m = n + int(e_ctx[n:hi].argmax())
                score = float(s_ctx[n] + e_ctx[m])
                node = owner[n] if owner[n] == owner[m] else lca(pg, owner[n], owner[m])
                while node >= 0 and node not in cand:          # climb to a candidate (tid element with text)
                    node = pg.parent[node]
                if node >= 0 and score > best[node]:
                    best[node] = score
            del pos
    return best, len(wins)


def rrf(*score_dicts, cands):
    fused = {i: 0.0 for i in cands}
    for s in score_dicts:
        order = sorted(cands, key=lambda i: (-s[i], i))
        rank, prev, r = {}, None, 0
        for n, i in enumerate(order):               # tied scores share a rank
            if s[i] != prev:
                r, prev = n + 1, s[i]
            rank[i] = r
        for i in cands:
            fused[i] += 1.0 / (RRF_K + rank[i])
    return {i: round(v, 12) for i, v in fused.items()}


def load_model(name, device):
    import torch
    from transformers import MarkupLMForQuestionAnswering, MarkupLMTokenizerFast
    tok = MarkupLMTokenizerFast.from_pretrained(name)
    model = MarkupLMForQuestionAnswering.from_pretrained(name).to(device).eval()
    if device.startswith("cuda"):
        model = model.half()
    return tok, model


def pick_device(want):
    if want != "auto":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--role", required=True, choices=["dev", "test"], help="only sets MarkupLM was not trained on")
    ap.add_argument("--sites")
    ap.add_argument("--per-site", type=int, default=100)
    ap.add_argument("--max-per-page", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit-pages", type=int, default=0, help="smoke test: stop after N pages")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sites = set(a.sites.split(",")) if a.sites else None
    device = pick_device(a.device)
    tok, model = load_model(a.model, device)
    print(f"[model] {a.model} on {device}")

    rows, _ = S.load_diverse(a.data, sites, a.per_site, a.max_per_page)
    by_page = OrderedDict()
    for r in rows:
        by_page.setdefault(r.page_key, []).append(r)

    per_q, n_windows, n_strings = {}, 0, 0
    t0 = time.time()
    for n, (pk, qs) in enumerate(by_page.items(), 1):
        if a.limit_pages and n > a.limit_pages:
            break
        pg = L.parse_page(qs[0].html_path.read_text(encoding="utf-8", errors="replace"))
        cands = [i for i in range(len(pg.els)) if pg.key[i]]
        if not cands:
            continue
        strings = page_strings(pg)
        lens = string_lens(tok, strings)
        n_strings += len(strings)
        for r in qs:
            g = pg.tid2i.get(r.element_id)
            if g is None or not pg.key[g]:
                continue
            _, aux = C.rank_all(pg, r.question, seed=a.seed)
            sc, cs = aux["sc"], aux["cs"]
            own = {i: round(sc[i] + C.LAMBDA * cs[i], 9) for i in cands}
            ml, nw = markuplm_scores(tok, model, device, pg, r.question, strings, cands, a.batch, lens)
            n_windows += nw
            scores = {"bm25_text": sc, "bm25_own": own, "markuplm": ml,
                      "rrf_markuplm_own": rrf(ml, own, cands=cands), "rrf_markuplm_text": rrf(ml, sc, cands=cands)}
            mm = {"_diag": {"page": pk, "ml_has_span_gold": int(ml[g] > NEG / 2)}}
            for m, s_ in scores.items():
                mm[m] = R.tie_metrics(s_, cands, g)
            per_q[(r.qid, r.site)] = mm
        del pg
        if n % 25 == 0:
            print(f"  pages {n}/{len(by_page)}  q={len(per_q)}  {time.time() - t0:.0f}s")

    vals = list(per_q.values())
    methods = [m for m in OrderedDict.fromkeys(k for v in vals for k in v) if not m.startswith("_")]
    table = {m: {k: round(C.mean(v[m][k] for v in vals), 4) for k in ("rand:hit@1", "rand:hit@10", "rand:hit@30")}
             for m in methods}

    def T(tid, A, B, metric="rand:hit@30", kind=""):
        t = C.paired(per_q, A, B, metric)
        if not t:
            return None
        diffs = [(site, mm["_diag"]["page"], mm[A][metric] - mm[B][metric]) for (qid, site), mm in per_q.items()]
        _, lo2, hi2 = S.two_level_boot(diffs)
        t.update({"id": tid, "type": kind, "page_ci95": [round(lo2, 4), round(hi2, 4)]})
        return t

    tests = [T("L1", "bm25_own", "markuplm", kind="PRIMARY L1: BM25 + own semantic relations vs MarkupLM (R@30)"),
             T("L2", "rrf_markuplm_own", "markuplm", kind="PRIMARY L2: MarkupLM fused with own relations vs MarkupLM (R@30)"),
             T("L3", "markuplm", "bm25_text", kind="MarkupLM vs BM25 text only (R@30)"),
             T("L1h", "bm25_own", "markuplm", metric="rand:hit@1", kind="BM25 + own vs MarkupLM (Hit@1)"),
             T("L2h", "rrf_markuplm_own", "markuplm", metric="rand:hit@1", kind="fusion vs MarkupLM (Hit@1)")]
    tests = [t for t in tests if t]
    fam = [t for t in tests if t["id"] in ("L1", "L2")]
    for t, adj in zip(fam, R.holm([t["sign_p"] for t in fam])):
        t["holm_p"] = float(f"{adj:.3g}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = {"experiment": "C17 MarkupLM node ranking (PREREG_C17.md)", "role": a.role, "model": a.model,
            "device": device, "code_hash": code_hash(), "n": len(per_q),
            "n_pages": len({mm["_diag"]["page"] for mm in vals}),
            "mean_windows_per_question": round(n_windows / max(1, len(per_q)), 2),
            "gold_has_span": round(C.mean(v["_diag"]["ml_has_span_gold"] for v in vals), 4) if vals else None,
            "seconds": round(time.time() - t0), "methods": table, "tests": tests}
    (out / "c17_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== C17 [{a.role}] n={len(per_q)} pages={summ['n_pages']} model={a.model} hash={summ['code_hash']} ===")
    print(f"windows/question={summ['mean_windows_per_question']}  gold node got a MarkupLM span={summ['gold_has_span']}  "
          f"time={summ['seconds']}s")
    print(f"\n   {'method':22s}{'R@30':>8s}{'R@10':>8s}{'Hit@1':>8s}")
    for m, v in table.items():
        print(f"   {m:22s}{v['rand:hit@30']:8.3f}{v['rand:hit@10']:8.3f}{v['rand:hit@1']:8.3f}")
    print("\n--- paired tests (site CI, site->page CI) ---")
    for t in tests:
        print(f"{t['id']:5s} {t['A']} - {t['B']} [{t['metric']}] n={t['n']} diff={t['diff'] * 100:+.1f}%p "
              f"CI[{t['ci95'][0] * 100:+.1f},{t['ci95'][1] * 100:+.1f}] pageCI[{t['page_ci95'][0] * 100:+.1f},{t['page_ci95'][1] * 100:+.1f}] "
              f"sites+/-={t['sites_pos_neg'][0]}/{t['sites_pos_neg'][1]}" + (f" holm={t['holm_p']}" if "holm_p" in t else "")
              + f"  | {t['type']}")
    print(f"[saved] {out.resolve()}")


if __name__ == "__main__":
    main()
