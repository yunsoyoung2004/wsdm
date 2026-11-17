# -*- coding: utf-8 -*-
"""
Identity-Ceiling experiment library (WebSRC, official GT only).

Core idea
---------
Any scorer that looks only at a node's own text gives IDENTICAL scores to
nodes with identical text. If the gold node's text also appears in other,
non-nested DOM positions ("disjoint twins"), a text-only ranker can only pick
the gold node by tie-break luck. This module measures that ceiling, and tests
GT-free ways to (a) remove true duplicates (nested chains) without deleting
distinct evidence (disjoint twins), and (b) break ties using DOM context.

Rule: ranking functions never receive the gold element. Gold is used only in
`evaluate_rank` / taxonomy (evaluation side).
"""
from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

try:  # faster and closer to browser trees when present
    import lxml  # noqa: F401
    PARSER = "lxml"
except Exception:  # pragma: no cover
    PARSER = "html.parser"

KS = (1, 5, 10, 30, 50, 100)
HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")
CTX_MAX_TOKENS = 64
LABEL_MAX_TOKENS = 12

_ws = re.compile(r"\s+")
_tok = re.compile(r"\w+", re.UNICODE)


def norm_ws(s: str) -> str:
    return _ws.sub(" ", s or "").strip()


def norm_key(s: str) -> str:
    """Normalisation used to decide 'identical text'."""
    return norm_ws(s).lower()


def tokenize(s: str) -> list[str]:
    return _tok.findall((s or "").lower())


# ---------------------------------------------------------------- loading
REQ = {"question", "element_id", "answer"}


def _colmap(header):
    low = {h.strip().lower(): h for h in header}
    m = {}
    for want, alts in {
        "question": ["question"],
        "qid": ["id", "qid", "question_id"],
        "element_id": ["element_id", "tid"],
        "answer_start": ["answer_start"],
        "answer": ["answer"],
    }.items():
        for a in alts:
            if a in low:
                m[want] = low[a]
                break
    return m


_site_part = re.compile(r"^([a-z]+?)(\d+)$")


def site_from(qid: str, csv_path: Path) -> str:
    if re.match(r"^[a-z]{2}\d{2}", qid or ""):
        return qid[:4]
    parts = [p.lower() for p in csv_path.parent.parts]
    for p in reversed(parts):
        m = _site_part.match(p)
        if m:
            return m.group(1)[:2] + m.group(2).zfill(2)
    if len(parts) >= 2 and parts[-1].isdigit():
        return parts[-2][:2] + parts[-1].zfill(2)
    return csv_path.parent.name


def _html_index(site_dir: Path) -> dict[str, Path]:
    cands = []
    for sub in ("processed_data", "html", "."):
        d = site_dir / sub
        if d.is_dir():
            cands += list(d.glob("*.html"))
    return {p.stem: p for p in cands}


def resolve_page(qid: str, stems: dict[str, Path]):
    for s in (qid[2:9], qid[4:11], qid[:-5], qid[2:-5], qid[4:-5]):
        if s in stems:
            return s
    best = None
    for st in stems:  # fallback: longest stem embedded in the id
        if len(st) >= 4 and st in qid[2:] and (best is None or len(st) > len(best)):
            best = st
    return best


@dataclass
class QRow:
    qid: str
    site: str
    page_key: str
    html_path: Path
    question: str
    element_id: str
    answer: str
    answer_start: int


def load_rows(data_root: Path, sites=None, split=None, max_per_site=None,
              qids=None, verbose=True):
    data_root = Path(data_root)
    csvs = []
    for p in data_root.rglob("*.csv"):
        try:
            with open(p, encoding="utf-8-sig", newline="") as f:
                header = next(csv.reader(f))
        except Exception:
            continue
        cm = _colmap(header)
        if REQ <= set(cm) and "qid" in cm:
            csvs.append((p, cm))
    if not csvs:
        raise SystemExit(f"[load] no WebSRC-style CSV (question,id,element_id,answer) under {data_root}. "
                         f"Run with --inspect and send the output.")
    if split:  # honour a split directory if the layout has one
        sp = [c for c in csvs if split.lower() in [x.lower() for x in c[0].parts]]
        if sp:
            csvs = sp
    by_site = defaultdict(list)
    skipped = Counter()
    for p, cm in sorted(csvs, key=lambda x: str(x[0])):
        stems = _html_index(p.parent)
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                qid = str(r[cm["qid"]]).strip()
                eid = str(r[cm["element_id"]]).strip()
                try:
                    eid_i = int(float(eid))
                except ValueError:
                    skipped["bad_element_id"] += 1
                    continue
                if eid_i < 0:  # yes/no questions have no evidence node
                    skipped["yes_no_or_no_node"] += 1
                    continue
                site = site_from(qid, p)
                if sites and site not in sites:
                    continue
                pk = resolve_page(qid, stems)
                if pk is None:
                    skipped["html_not_found"] += 1
                    continue
                try:
                    a_st = int(float(r.get(cm.get("answer_start", ""), -1) or -1))
                except ValueError:
                    a_st = -1
                by_site[site].append(QRow(qid, site, f"{site}/{pk}", stems[pk],
                                          r[cm["question"]], str(eid_i),
                                          r[cm["answer"]], a_st))
    rows = []
    for site in sorted(by_site):
        lst = by_site[site]
        if qids is not None:
            lst = [q for q in lst if q.qid in qids]
        if max_per_site:
            lst = lst[:max_per_site]
        rows += lst
    if verbose:
        print(f"[load] csv files={len(csvs)} sites={sorted(by_site)} rows_used={len(rows)} skipped={dict(skipped)}")
    return rows, dict(skipped), sorted(by_site)


# ---------------------------------------------------------------- pages
@dataclass
class Page:
    els: list
    tids: list
    text: list
    key: list
    toks: list
    depth: list
    parent: list
    heading: list
    tid2i: dict
    bm25: "BM25" = None
    _ctx: dict = field(default_factory=dict)
    _ctxc: dict = field(default_factory=dict)
    _chain: list = None


def parse_page(html: str) -> Page:
    soup = BeautifulSoup(html, PARSER)
    els, last_head, heading = [], "", []
    for el in soup.descendants:  # single document-order pass
        if not isinstance(el, Tag):
            continue
        if el.has_attr("tid"):
            els.append(el)
            heading.append(last_head)  # last heading strictly before this node
        if el.name in HEADINGS:
            last_head = norm_ws(el.get_text(" ", strip=True))[:200]
    tids = [str(e.get("tid")).strip() for e in els]
    tid2i = {t: i for i, t in enumerate(tids)}
    elid = {id(e): i for i, e in enumerate(els)}
    parent, depth = [], []
    for i, e in enumerate(els):
        p, par = e.parent, -1
        while p is not None:
            if id(p) in elid:
                par = elid[id(p)]
                break
            p = p.parent
        parent.append(par)
        depth.append(0 if par < 0 else depth[par] + 1)
    text = [norm_ws(e.get_text(" ", strip=True)) for e in els]
    pg = Page(els, tids, text, [t.lower() for t in text], [tokenize(t) for t in text],
              depth, parent, heading, tid2i)
    pg.bm25 = BM25(pg.toks)
    return pg


def is_ancestor(pg: Page, a: int, b: int) -> bool:
    p = pg.parent[b]
    while p >= 0:
        if p == a:
            return True
        p = pg.parent[p]
    return False


def chains(pg: Page) -> list[int]:
    """Union nodes with an identical-text tid-parent -> nested same-text chains."""
    if pg._chain is not None:
        return pg._chain
    root = list(range(len(pg.els)))

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x
    for i, p in enumerate(pg.parent):
        if p >= 0 and pg.key[i] and pg.key[i] == pg.key[p]:
            root[find(i)] = find(p)
    pg._chain = [find(i) for i in range(len(pg.els))]
    return pg._chain


# ---------------------------------------------------------------- BM25
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.N) if self.N else 0.0
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]

    def _score(self, q, tf, dl):
        if not dl or not self.avgdl:
            return 0.0
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if f:
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def scores(self, q):
        return [round(self._score(q, tf, dl), 9) for tf, dl in zip(self.tfs, self.lens)]

    def score_tokens(self, q, toks):
        return round(self._score(q, Counter(toks), len(toks)), 9)


# ---------------------------------------------------------------- context
def _t(el) -> str:
    return norm_ws(el.get_text(" ", strip=True)) if el is not None else ""


def context_text(pg: Page, i: int) -> str:
    """GT-free structural context: row header, column header, preceding labels, last heading."""
    if i in pg._ctx:
        return pg._ctx[i]
    el, parts = pg.els[i], []
    cell = el if el.name in ("td", "th") else el.find_parent(["td", "th"])
    if cell is not None:
        tr = cell.find_parent("tr")
        if tr is not None:
            cells = tr.find_all(["td", "th"], recursive=False)
            if cells and cells[0] is not cell:
                parts.append(_t(cells[0])[:200])
            col = next((k for k, c in enumerate(cells) if c is cell), None)
            table = tr.find_parent("table")
            if table is not None and col is not None:
                first = table.find("tr")
                if first is not None and first is not tr:
                    hc = first.find_all(["td", "th"], recursive=False)
                    if col < len(hc):
                        parts.append(_t(hc[col])[:200])
    # nearest preceding label: first non-empty previous sibling, climbing at most
    # 3 levels, stop at the first one found; labels are short, so cap at 12 tokens
    cur = el
    for _ in range(3):
        sib = cur.find_previous_sibling()
        while sib is not None and not sib.get_text(strip=True):
            sib = sib.find_previous_sibling()
        if sib is not None:
            parts.append(" ".join(_t(sib).split()[:LABEL_MAX_TOKENS]))
            break
        cur = cur.parent
        if cur is None or cur.name in ("body", "html", "[document]"):
            break
    if pg.heading[i]:
        parts.append(pg.heading[i])
    toks = " ".join(parts).split()
    out = " ".join(toks[:CTX_MAX_TOKENS])
    pg._ctx[i] = out
    return out


# ---------------------------------------------------------------- methods
def _candidates(pg):
    return [i for i in range(len(pg.els)) if pg.key[i]]


def _sorted(cands, score, tie):
    return sorted(cands, key=lambda i: (-score[i], tie(i), i))


def regroup_ties(order, score, secondary, limit=100):
    """Reorder ONLY inside groups of exactly equal primary score.
    Cross-group order is untouched (tie-only intervention). Groups starting at or
    after position `limit` are left as is (they cannot affect metrics@<=limit)."""
    out, pos, n = [], 0, len(order)
    while pos < n:
        j = pos
        while j + 1 < n and score[order[j + 1]] == score[order[pos]]:
            j += 1
        grp = order[pos:j + 1]
        if len(grp) > 1 and pos < limit:
            grp = sorted(grp, key=lambda i: (-secondary(i), i))
        out += grp
        pos = j + 1
    return out


def rank_methods(pg: Page, question: str, seed: int = 0, n_rand: int = 10, limit=100, fusion=True):
    """Return {method: ranked list of node indices}. No gold input by construction."""
    q = tokenize(question)
    sc = pg.bm25.scores(q)
    cands = _candidates(pg)
    doc = _sorted(cands, sc, lambda i: 0)
    out = {"bm25_doc": doc}
    for s in range(n_rand):
        rng = random.Random(f"{seed}-{s}")
        perm = {i: rng.random() for i in cands}
        out[f"bm25_rand#{s}"] = _sorted(cands, sc, lambda i, p=perm: p[i])

    seen, ded = set(), []
    for i in doc:  # == earlier text_dedup: first occurrence in ranked order
        if pg.key[i] not in seen:
            seen.add(pg.key[i])
            ded.append(i)
    out["dedup_first"] = ded

    best = {}
    for i in cands:
        k = pg.key[i]
        if k not in best or (pg.depth[i], -i) > (pg.depth[best[k]], -best[k]):
            best[k] = i
    out["dedup_deepest"] = _sorted(list(best.values()), sc, lambda i: 0)

    ch = chains(pg)
    rep = {}
    for i in cands:
        c = ch[i]
        if c not in rep or (pg.depth[i], -i) > (pg.depth[rep[c]], -rep[c]):
            rep[c] = i
    reps = list(rep.values())
    cc = _sorted(reps, sc, lambda i: 0)
    out["chain_doc"] = cc
    for s in range(n_rand):
        rng = random.Random(f"{seed}-c{s}")
        perm = {i: rng.random() for i in reps}
        out[f"chain_rand#{s}"] = _sorted(reps, sc, lambda i, p=perm: p[i])
    out["chain_depth_tie"] = regroup_ties(cc, sc, lambda i: pg.depth[i], limit)
    ctx_s = lambda i: ctx_score(pg, q, i)
    out["chain_ctx_tie"] = regroup_ties(cc, sc, ctx_s, limit)
    rng = random.Random(f"{seed}-shuf")
    pool = list(range(len(pg.els)))
    donor = {i: rng.choice(pool) for i in reps}  # context from a random other node
    shuf_s = lambda i: ctx_score(pg, q, donor[i])
    out["chain_shufctx_tie"] = regroup_ties(cc, sc, shuf_s, limit)
    if fusion:
        # non-tie: text score + LAMBDA * context score (lambda fixed a priori = 1.0)
        fz = {i: round(sc[i] + FUSION_LAMBDA * ctx_s(i), 9) for i in reps}
        out["chain_ctx_fusion"] = sorted(reps, key=lambda i: (-fz[i], i))
        fzs = {i: round(sc[i] + FUSION_LAMBDA * shuf_s(i), 9) for i in reps}
        out["chain_shufctx_fusion"] = sorted(reps, key=lambda i: (-fzs[i], i))
    return out, sc


FUSION_LAMBDA = 1.0


def ctx_score(pg: Page, q: list, i: int) -> float:
    c = pg._ctxc.get(i)
    if c is None:
        toks = tokenize(context_text(pg, i))
        c = pg._ctxc[i] = (Counter(toks), len(toks))
    return round(pg.bm25._score(q, c[0], c[1]), 9)


# ---------------------------------------------------------------- evaluation
def taxonomy(pg: Page, g: int) -> dict:
    same = [i for i in range(len(pg.els)) if pg.key[i] == pg.key[g]]
    ch = chains(pg)
    comps = Counter(ch[i] for i in same)
    m = len(comps)
    if len(same) == 1:
        typ = "unique"
    elif m == 1:
        typ = "nested"
    elif all(v == 1 for v in comps.values()):
        typ = "disjoint"
    else:
        typ = "mixed"
    chain_nodes = [i for i in same if ch[i] == ch[g]]
    gt_deepest = max(chain_nodes, key=lambda i: (pg.depth[i], -i)) == g
    return {
        "collision_type": typ,
        "n_same_text": len(same),
        "n_disjoint_chains": m,
        "gt_is_chain_deepest": int(gt_deepest),
        "ceiling_text_only": 1.0 / len(same),
        "ceiling_text_depth": (1.0 / m) if gt_deepest else 0.0,
        "gt_depth": pg.depth[g],
        "gt_n_tokens": len(pg.toks[g]),
        "page_nodes": len(pg.els),
        "page_dup_text_rate": round(1 - len(set(k for k in pg.key if k)) / max(1, sum(1 for k in pg.key if k)), 4),
    }


def evaluate_rank(pg: Page, lst: list, g: int, score=None) -> dict:
    try:
        r = lst.index(g) + 1
    except ValueError:
        r = None
    d = {"rank": r if r else -1, "list_len": len(lst)}
    for k in KS:
        d[f"hit@{k}"] = int(r is not None and r <= k)
    d["mrr@10"] = (1.0 / r) if (r is not None and r <= 10) else 0.0
    top = lst[0] if lst else None
    d["top1_same_text_wrong_node"] = int(top is not None and top != g and pg.key[top] == pg.key[g])
    d["top1_is_gt_ancestor"] = int(top is not None and is_ancestor(pg, top, g))
    if score is not None and top is not None:
        d["top1_in_tie"] = int(sum(1 for i in lst[:200] if score[i] == score[top]) > 1)
    for k in (10, 30, 100):
        d[f"tokens@{k}"] = sum(len(pg.toks[i]) for i in lst[:k])
    return d
