# -*- coding: utf-8 -*-
"""
C4 library: confirmatory test of the "relational bridge" finding + exploratory probes.

Frozen design values (do not tune on Dev/Select/Confirm results):
  LAMBDA = 1.0  (fusion weight), relation parts = row header, column header,
  nearest preceding label (<=12 tokens, climb <=3 levels), last heading; total <=64 tokens.
Rankers never receive the gold element (see test_c4.py).
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import re
from collections import Counter

from bs4 import Comment, Doctype, NavigableString, Tag

import idc_lib as L

LAMBDA = 1.0
TYPE_BONUS = 1.0
PARTS = ("row", "col", "label", "head")
KS = (1, 5, 10, 30, 100)


# ------------------------------------------------------------------ site split
def split_sites(sites, salt="c4-websrc-2026", frac=(0.6, 0.2, 0.2)):
    """Deterministic hash split of official TRAIN sites -> train / select / confirm."""
    order = sorted(sites, key=lambda s: hashlib.sha256(f"{salt}|{s}".encode()).hexdigest())
    n = len(order)
    a = round(n * frac[0])
    b = a + round(n * frac[1])
    return {"train": sorted(order[:a]), "select": sorted(order[a:b]), "confirm": sorted(order[b:]),
            "salt": salt, "n_sites": n}


# ------------------------------------------------------------------ relations
# Version switch. v1 = everything reported up to C6 (frozen). v2 = extractor fixes developed on
# TRAIN sites only. Select with the environment variable C4_REL=v2 (default v1).
REL_VERSION = os.environ.get("C4_REL", "v1")


def rel_parts(pg, i):
    cache = pg.__dict__.setdefault("_parts_" + REL_VERSION, {})
    if i not in cache:
        cache[i] = _rel_parts_v2(pg, i) if REL_VERSION == "v2" else _rel_parts_v1(pg, i)
    return cache[i]


def _rel_parts_v1(pg, i):
    el = pg.els[i]
    p = {"row": "", "col": "", "label": "", "head": pg.heading[i]}
    cell = el if el.name in ("td", "th") else el.find_parent(["td", "th"])
    if cell is not None:
        tr = cell.find_parent("tr")
        if tr is not None:
            cells = tr.find_all(["td", "th"], recursive=False)
            if cells and cells[0] is not cell:
                p["row"] = L._t(cells[0])[:200]
            col = next((k for k, c in enumerate(cells) if c is cell), None)
            table = tr.find_parent("table")
            if table is not None and col is not None:
                first = table.find("tr")
                if first is not None and first is not tr:
                    hc = first.find_all(["td", "th"], recursive=False)
                    if col < len(hc):
                        p["col"] = L._t(hc[col])[:200]
    cur = el
    for _ in range(3):
        sib = cur.find_previous_sibling()
        while sib is not None and not sib.get_text(strip=True):
            sib = sib.find_previous_sibling()
        if sib is not None:
            p["label"] = " ".join(L._t(sib).split()[:L.LABEL_MAX_TOKENS])
            break
        cur = cur.parent
        if cur is None or cur.name in ("body", "html", "[document]"):
            break
    return p


# ---- v2 ------------------------------------------------------------
_KEY_END = re.compile(r"[:：\-–—]\s*$")
_NUMERICISH = re.compile(r"^[\W\d_]+$")
KEY_TAGS = {"th", "dt", "label"}
V2_SCAN = 3          # previous items inspected per level
V2_WEAK_MAX_TOKENS = 4


def _item_text(x):
    return L.norm_ws(str(x)) if isinstance(x, NavigableString) else L._t(x)


def _prev_items(node, k):
    """Previous siblings INCLUDING bare text nodes (v1 only saw tags), non-empty, nearest first."""
    out = []
    for x in node.previous_siblings:
        if isinstance(x, (Comment, Doctype)):
            continue
        if isinstance(x, NavigableString) or isinstance(x, Tag):
            t = _item_text(x)
            if t:
                out.append((x, t))
                if len(out) >= k:
                    break
    return out


def is_strong_key(x, t):
    toks = t.split()
    return 0 < len(toks) <= 6 and (bool(_KEY_END.search(t)) or (isinstance(x, Tag) and x.name in KEY_TAGS))


def is_weak_key(t):
    toks = t.split()
    return 0 < len(toks) <= V2_WEAK_MAX_TOKENS and not re.search(r"\d", t)


def _rel_parts_v2(pg, i):
    el = pg.els[i]
    p = {"row": "", "col": "", "label": "", "head": pg.heading[i]}
    cell = el if el.name in ("td", "th") else el.find_parent(["td", "th"])
    if cell is not None:
        tr = cell.find_parent("tr")
        if tr is not None:
            cells = tr.find_all(["td", "th"], recursive=False)
            col = next((k for k, c in enumerate(cells) if c is cell), None)
            if col is not None:
                # (C) row identity = first NON-numeric cell left of this one (skip rank columns)
                for c in cells[:col]:
                    t = L._t(c)
                    if t and not _NUMERICISH.match(t):
                        p["row"] = t[:200]
                        break
                table = tr.find_parent("table")
                if table is not None:
                    head = None
                    thead = table.find("thead")
                    if thead is not None and thead.find_all("tr"):
                        head = thead.find_all("tr")[-1]
                    if head is None:
                        head = next((r for r in table.find_all("tr") if r.find("th", recursive=False)), None)
                    if head is None:
                        head = table.find("tr")
                    if head is not None and head is not tr:
                        hc = head.find_all(["td", "th"], recursive=False)
                        if col < len(hc):
                            p["col"] = L._t(hc[col])[:200]
    # (A)+(B) label: look at previous items incl. bare text; take a strong key (ends with ':' / '-'
    # or th/dt/label), else the immediately previous item only if it is a short digit-free phrase.
    # If previous items exist but none is key-like -> stop with NO label (do not borrow a
    # neighbour's value). Climb a level only when this node has no previous items at all.
    cur = el
    for _ in range(3):
        items = _prev_items(cur, V2_SCAN)
        if items:
            strong = next(((x, t) for x, t in items if is_strong_key(x, t)), None)
            if strong is not None:
                p["label"] = " ".join(strong[1].split()[:L.LABEL_MAX_TOKENS])
            elif is_weak_key(items[0][1]):
                p["label"] = items[0][1]
            break
        cur = cur.parent
        if cur is None or cur.name in ("body", "html", "[document]"):
            break
    return p


def uniq_parts(pg, i):
    """Relation parts with identical text removed (e.g. label == row header in tables)."""
    p, seen, out = rel_parts(pg, i), set(), []
    for k in PARTS:
        v = p[k]
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append((k, v))
    return out


def rel_string(pg, i):
    toks = " ".join(v for _, v in uniq_parts(pg, i)).split()
    return " ".join(toks[:L.CTX_MAX_TOKENS])


def _tok_cache(pg, key, i, fn):
    c = pg.__dict__.setdefault(f"_tc_{REL_VERSION}_{key}", {})
    if i not in c:
        t = L.tokenize(fn(i))
        c[i] = (Counter(t), len(t), set(t))
    return c[i]


def rel_tokens(pg, i):
    return _tok_cache(pg, "all", i, lambda j: rel_string(pg, j))


def part_tokens(pg, part, i):
    return _tok_cache(pg, part, i, lambda j: rel_parts(pg, j)[part])


def bm25_counter(pg, q, c):
    return round(pg.bm25._score(q, c[0], c[1]), 9)


# ------------------------------------------------------------------ answer type (exploratory)
_MONEY_Q = re.compile(r"\b(price|cost|costs|how much|pay|fee|salary)\b")
_NUM_Q = re.compile(r"\b(how many|number of|how old|year|when|rating|score|how long|how tall|"
                    r"how heavy|weight|mileage|mpg|age|percent|rank)\b")


def qtype(q):
    ql = q.lower()
    if _MONEY_Q.search(ql):
        return "money"
    if _NUM_Q.search(ql):
        return "number"
    return "other"


def type_ok(text, t):
    if t == "money":
        return bool(re.search(r"[$€£¥₩]|\d+[.,]\d{2}\b", text))
    if t == "number":
        return bool(re.search(r"\d", text))
    return True


# ------------------------------------------------------------------ ranking
def rank_all(pg, question, seed=0, explore=False):
    """All C4 rankers. NO gold argument by construction."""
    q = L.tokenize(question)
    qs = set(q)
    sc = pg.bm25.scores(q)
    n = len(pg.els)
    cands = [i for i in range(n) if pg.key[i]]
    kc = Counter(pg.key[i] for i in cands)
    cs = {i: bm25_counter(pg, q, rel_tokens(pg, i)) for i in cands}
    out = {}
    out["bm25"] = sorted(cands, key=lambda i: (-sc[i], i))
    out["tieonly"] = L.regroup_ties(out["bm25"], sc, lambda i: cs[i], limit=10 ** 9)
    fz = {i: sc[i] + LAMBDA * cs[i] for i in cands}
    out["fusion_all"] = sorted(cands, key=lambda i: (-fz[i], i))
    gate = {i: (sc[i] == 0 or kc[pg.key[i]] > 1) for i in cands}
    out["fusion_need"] = sorted(cands, key=lambda i: (-(sc[i] + LAMBDA * cs[i] * gate[i]), i))

    rng = random.Random(f"{seed}|{question}|{n}")
    shuf_donor = {}
    for i in cands:
        j = rng.choice(cands) if len(cands) > 1 else i
        while j == i and len(cands) > 1:
            j = rng.choice(cands)
        shuf_donor[i] = j
    fs = {i: sc[i] + LAMBDA * cs[shuf_donor[i]] for i in cands}
    out["fusion_shuf"] = sorted(cands, key=lambda i: (-fs[i], i))

    qpool = [j for j in cands if qs & rel_tokens(pg, j)[2]]
    qm_full = int(len(qpool) >= 2)
    qm_donor = {}
    for i in cands:
        opts = [j for j in qpool if j != i]
        qm_donor[i] = rng.choice(opts) if opts else None
    fq = {i: sc[i] + (LAMBDA * cs[qm_donor[i]] if qm_donor[i] is not None else 0.0) for i in cands}
    out["fusion_qm"] = sorted(cands, key=lambda i: (-fq[i], i))

    aux = {"sc": sc, "cs": cs, "qm_full": qm_full, "n_bridge_nodes": len(qpool),
           "shuf_donor": shuf_donor, "qm_donor": qm_donor}
    if explore:
        for part in PARTS:
            ps = {i: sc[i] + LAMBDA * bm25_counter(pg, q, part_tokens(pg, part, i)) for i in cands}
            out[f"x_fusion_{part}"] = sorted(cands, key=lambda i: (-ps[i], i))
        keys = {i: sum(1 for _, v in uniq_parts(pg, i) if qs & set(L.tokenize(v))) for i in cands}
        out["x_keycount"] = sorted(cands, key=lambda i: (-keys[i], -fz[i], i))
        t = qtype(question)
        tok = {i: type_ok(pg.text[i], t) for i in cands}
        out["x_fusion_type"] = sorted(cands, key=lambda i: (-(fz[i] + TYPE_BONUS * tok[i] * (t != "other")), i))
        aux.update({"keys": keys, "qtype": t, "type_ok": tok})
    return out, aux


# ------------------------------------------------------------------ evaluation side
def metrics(lst, g):
    try:
        r = lst.index(g) + 1
    except ValueError:
        r = None
    d = {f"hit@{k}": int(r is not None and r <= k) for k in KS}
    d["mrr@10"] = 1.0 / r if r is not None and r <= 10 else 0.0
    d["rank"] = r or -1
    return d


def diagnostics(pg, g, question, aux):
    """Gold-side descriptive variables (never fed to rankers)."""
    qs = set(L.tokenize(question))
    sc, cs = aux["sc"], aux["cs"]
    cands = [i for i in range(len(pg.els)) if pg.key[i]]
    same_text = [i for i in cands if pg.key[i] == pg.key[g]]
    gr = rel_string(pg, g)
    resid = [i for i in same_text if rel_string(pg, i) == gr]
    d = {
        "gold_bm25_zero": int(sc[g] == 0),
        "gold_dup_text": int(len(same_text) > 1),
        "bridge": int(cs[g] > 0),
        "n_same_text": len(same_text),
        "n_same_text_and_rel": len(resid),
        "ceiling_text": 1.0 / len(same_text),
        "ceiling_text_rel": 1.0 / len(resid),
        "n_bridge_nodes": aux["n_bridge_nodes"],
        "qm_full": aux["qm_full"],
        "n_cands": len(cands),
    }
    for part in PARTS:
        d[f"key_{part}"] = int(bool(qs & part_tokens(pg, part, g)[2]))
        d[f"has_{part}"] = int(bool(rel_parts(pg, g)[part]))
    if "keys" in aux:
        keys = aux["keys"]
        d["gold_keys"] = keys[g]
        d["n_nodes_keys_ge_gold"] = sum(1 for i in cands if keys[i] >= keys[g])
        d["qtype"] = aux["qtype"]
        d["gold_type_ok"] = int(aux["type_ok"][g])
        d["n_type_ok"] = sum(1 for i in cands if aux["type_ok"][i])
    return d


# ------------------------------------------------------------------ stats
def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


def cluster_boot(by_site, B=4000, seed=7):
    sites = sorted(by_site)
    allv = [v for s in sites for v in by_site[s]]
    est = mean(allv)
    if len(sites) < 2:
        return est, float("nan"), float("nan")
    rng = random.Random(seed)
    bs = []
    for _ in range(B):
        pick = [rng.choice(sites) for _ in sites]
        bs.append(mean(v for s in pick for v in by_site[s]))
    bs.sort()
    return est, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p), pos, neg


def paired(rows, A, B, metric, keep=lambda r: True):
    """rows: {(qid,site): {method: metrics..., '_diag': {...}}}"""
    by_site, flat = {}, []
    for (qid, site), mm in rows.items():
        if A in mm and B in mm and keep(mm):
            d = mm[A][metric] - mm[B][metric]
            by_site.setdefault(site, []).append(d)
            flat.append(d)
    if not flat:
        return None
    est, lo, hi = cluster_boot(by_site)
    p, pos, neg = sign_test(flat)
    sp = sum(1 for v in by_site.values() if mean(v) > 0)
    sn = sum(1 for v in by_site.values() if mean(v) < 0)
    return {"A": A, "B": B, "metric": metric, "n": len(flat), "diff": round(est, 4),
            "ci95": [round(lo, 4), round(hi, 4)], "wins": [pos, neg], "sign_p": float(f"{p:.3g}"),
            "sites_pos_neg": [sp, sn], "n_sites": len(by_site)}
