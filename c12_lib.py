# -*- coding: utf-8 -*-
"""
C12 library: grounding and limits of the four relation rules (pre-registered in PREREG_C12.md).

Everything that builds a node context here is GT-free: functions take (page, node index) and
never see the gold node. The bridge-location audit (A) is the only place gold is used, and only
to describe where question words sit relative to the gold node.
"""
from __future__ import annotations

import random
import re
from collections import Counter

import c4_lib as C
import c8_robust as R
import idc_lib as L

STOP = set("""a an the of in on at to for from by with and or is are was were be been being do does did
what which who whom whose where when why how much many this that these those it its as into than
there their they them his her he she you your i me my we our us can could would should will shall
may might must has have had not no yes any some all each per about over under up down out off""".split())

ATTR_KEYS = ("title", "alt", "aria-label", "itemprop", "class", "id", "name")
EXT_ATTR_KEYS = ("title", "alt", "aria-label", "itemprop")   # attributes used in fusion_ext
ANC_LEVELS = 3
ANC_CAP = 64
PREV_WIN = 10
EXT_CAP = 96

AUDIT_CATS = ("row", "col", "label", "head",            # the four rules (v1)
              "parent", "ancestors", "siblings", "same_row", "same_col",
              "caption", "dtdd", "label_for", "figlegend", "attrs", "prev_window", "anywhere")
RULE_CATS = ("row", "col", "label", "head")
EXTRA_CATS = AUDIT_CATS[4:-1]


# ------------------------------------------------------------------ tokens
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def toks(s):
    return L.tokenize(_CAMEL.sub(" ", s or ""))


def content_tokens(question):
    return [t for t in L.tokenize(question) if t not in STOP]


def bridge_tokens(question, gold_text):
    """Question content words that the gold node itself does not contain."""
    g = set(L.tokenize(gold_text))
    return set(t for t in content_tokens(question) if t not in g)


# ------------------------------------------------------------------ page helpers
def soup_root(pg):
    cache = pg.__dict__.get("_c12_root")
    if cache is None:
        r = pg.els[0] if pg.els else None
        while r is not None and r.parent is not None:
            r = r.parent
        pg.__dict__["_c12_root"] = cache = r
    return cache


def _strip_own(text, own):
    return text.replace(own, " ", 1) if own else text


def _cap(s, n):
    return " ".join((s or "").split()[:n])


def _cell(el):
    return el if el.name in ("td", "th") else el.find_parent(["td", "th"])


# ------------------------------------------------------------------ contexts (GT-free)
def ctx_ancestors(pg, i):
    own, out, p = pg.text[i], [], pg.parent[i]
    for _ in range(ANC_LEVELS):
        if p < 0:
            break
        out.append(_cap(_strip_own(pg.text[p], own), ANC_CAP))
        own, p = pg.text[p], pg.parent[p]
    return " ".join(out)


def ctx_same_row(pg, i):
    cell = _cell(pg.els[i])
    tr = cell.find_parent("tr") if cell is not None else None
    if tr is None:
        return ""
    return " ".join(L._t(c) for c in tr.find_all(["td", "th"], recursive=False) if c is not cell)


def ctx_same_col(pg, i):
    cell = _cell(pg.els[i])
    tr = cell.find_parent("tr") if cell is not None else None
    table = tr.find_parent("table") if tr is not None else None
    if table is None:
        return ""
    cells = tr.find_all(["td", "th"], recursive=False)
    col = next((k for k, c in enumerate(cells) if c is cell), None)
    if col is None:
        return ""
    out = []
    for row in table.find_all("tr"):
        if row is tr:
            continue
        rc = row.find_all(["td", "th"], recursive=False)
        if col < len(rc):
            out.append(L._t(rc[col]))
    return _cap(" ".join(out), 200)


def ctx_caption(pg, i):
    table = pg.els[i].find_parent("table") if pg.els[i].name != "table" else pg.els[i]
    cap = table.find("caption") if table is not None else None
    return L._t(cap)


def ctx_dtdd(pg, i):
    el = pg.els[i]
    dd = el if el.name == "dd" else el.find_parent("dd")
    if dd is None:
        return ""
    dt = dd.find_previous_sibling("dt")
    return L._t(dt)


def _label_map(pg):
    m = pg.__dict__.get("_c12_labels")
    if m is None:
        m, root = {}, soup_root(pg)
        if root is not None:
            for lab in root.find_all("label"):
                f = lab.get("for")
                if f:
                    m.setdefault(str(f), []).append(L._t(lab))
        pg.__dict__["_c12_labels"] = m
    return m


def ctx_label_for(pg, i):
    m = _label_map(pg)
    if not m:
        return ""
    cur, out = pg.els[i], []
    for _ in range(ANC_LEVELS + 1):
        if cur is None or not hasattr(cur, "get"):
            break
        idv = cur.get("id")
        if idv and str(idv) in m:
            out += m[str(idv)]
        cur = cur.parent
    return " ".join(out)


def ctx_figlegend(pg, i):
    el, out = pg.els[i], []
    fig = el.find_parent("figure")
    if fig is not None and fig.find("figcaption") is not None:
        out.append(L._t(fig.find("figcaption")))
    fs = el.find_parent("fieldset")
    if fs is not None and fs.find("legend") is not None:
        out.append(L._t(fs.find("legend")))
    return " ".join(out)


def ctx_attrs(pg, i, keys=ATTR_KEYS):
    cur, out = pg.els[i], []
    for _ in range(ANC_LEVELS + 1):
        if cur is None or not hasattr(cur, "get") or cur.name in ("[document]", "html", "body"):
            break
        for k in keys:
            v = cur.get(k)
            if v:
                out.append(" ".join(v) if isinstance(v, list) else str(v))
        cur = cur.parent
    return " ".join(out)


def ctx_prev_window(pg, i):
    out = []
    for j in range(i - 1, -1, -1):
        if len(out) >= PREV_WIN:
            break
        if pg.key[j] and not L.is_ancestor(pg, j, i):
            out.append(pg.text[j])
    return " ".join(reversed(out))


def page_text(pg):
    c = pg.__dict__.get("_c12_pagetext")
    if c is None:
        root = soup_root(pg)
        c = L.norm_ws(root.get_text(" ", strip=True)) if root is not None else ""
        pg.__dict__["_c12_pagetext"] = c
    return c


def field_text(pg, i, cat):
    if cat in RULE_CATS:
        return C.rel_parts(pg, i)[cat]
    return {
        "parent": R.ctx_parent, "ancestors": ctx_ancestors, "siblings": R.ctx_sibling,
        "same_row": ctx_same_row, "same_col": ctx_same_col, "caption": ctx_caption,
        "dtdd": ctx_dtdd, "label_for": ctx_label_for, "figlegend": ctx_figlegend,
        "attrs": ctx_attrs, "prev_window": ctx_prev_window,
        "anywhere": lambda p, j: _strip_own(page_text(p), p.text[j]),
    }[cat](pg, i)


# ------------------------------------------------------------------ A: bridge-location audit (uses gold)
def audit(pg, g, question):
    bt = bridge_tokens(question, pg.text[g])
    row = {"n_bridge_tokens": len(bt)}
    for cat in AUDIT_CATS:
        row[f"in_{cat}"] = int(bool(bt & set(toks(field_text(pg, g, cat))))) if bt else 0
    row["in_rules"] = int(any(row[f"in_{c}"] for c in RULE_CATS))
    row["nowhere"] = int(bool(bt) and not row["in_anywhere"])
    for cat in EXTRA_CATS:
        row[f"adds_{cat}"] = int(row[f"in_{cat}"] and not row["in_rules"])
    row["present"] = {cat: int(bool(field_text(pg, g, cat).strip())) for cat in
                      ("caption", "dtdd", "label_for", "figlegend")}
    row["present"]["attrs_ext"] = int(bool(ctx_attrs(pg, g, EXT_ATTR_KEYS).strip()))
    return row


# ------------------------------------------------------------------ B/C: context strings (GT-free)
def rel_subset(pg, i, parts):
    return _cap(" ".join(v for k, v in C.uniq_parts(pg, i) if k in parts), L.CTX_MAX_TOKENS)


def ext_string(pg, i):
    extra = [ctx_caption(pg, i), ctx_dtdd(pg, i), ctx_label_for(pg, i), ctx_figlegend(pg, i),
             ctx_attrs(pg, i, EXT_ATTR_KEYS)]
    return _cap(" ".join([C.rel_string(pg, i)] + [e for e in extra if e]), EXT_CAP)


def rel_parent_string(pg, i):
    return _cap(C.rel_string(pg, i) + " " + _cap(R.ctx_parent(pg, i), 64), 128)


ABLATIONS = {
    "rel_all": lambda pg, i: C.rel_string(pg, i),
    **{f"minus_{p}": (lambda p: lambda pg, i: rel_subset(pg, i, set(C.PARTS) - {p}))(p) for p in C.PARTS},
    **{f"only_{p}": (lambda p: lambda pg, i: rel_subset(pg, i, {p}))(p) for p in C.PARTS},
    "rel_parent": rel_parent_string,
    "ext": ext_string,
}


def counter_for(pg, name, i, fn):
    cache = pg.__dict__.setdefault(f"_c12_{name}", {})
    if i not in cache:
        t = toks(fn(pg, i))
        cache[i] = (Counter(t), len(t))
    return cache[i]


# ------------------------------------------------------------------ H: one relation type at a time (GT-free)
# Structural relations (DOM Standard, node tree) and semantic relations (HTML Standard, element
# semantics), each attached ALONE, truncated to the same TYPE_CAP tokens. Children/descendants are
# not separate types: a node's text already contains its descendants' text.
TYPE_CAP = 16


def _prev_node(pg, i):
    for j in range(i - 1, -1, -1):
        if pg.key[j] and not L.is_ancestor(pg, j, i):
            return pg.text[j]
    return ""


def _next_node(pg, i):
    for j in range(i + 1, len(pg.els)):
        if pg.key[j] and not L.is_ancestor(pg, i, j):
            return pg.text[j]
    return ""


def _sib(pg, i, prev):
    el = pg.els[i]
    x = el.find_previous_sibling() if prev else el.find_next_sibling()
    while x is not None and not x.get_text(strip=True):
        x = x.find_previous_sibling() if prev else x.find_next_sibling()
    return L._t(x)


def _grandparent(pg, i):
    p = pg.parent[i]
    gp = pg.parent[p] if p >= 0 else -1
    return _strip_own(pg.text[gp], pg.text[p] if p >= 0 else "") if gp >= 0 else ""


def _tag_path(pg, i):
    return " ".join(t for t in R.ctx_xpath(pg, i).split() if t.isalpha() and len(t) <= 10)


def _minus_sem(text, pg, i):
    """Structural context with every token of the node's semantic relations removed."""
    sem = set(toks(C.rel_string(pg, i)))
    return " ".join(w for w in (text or "").split() if not (set(toks(w)) & sem))


TYPES_STRUCT = {
    "s_parent_minus_sem": lambda pg, i: _minus_sem(R.ctx_parent(pg, i), pg, i),     # parent, semantic part removed
    "s_siblings_minus_sem": lambda pg, i: _minus_sem(R.ctx_sibling(pg, i), pg, i),  # siblings, semantic part removed
    "s_parent": R.ctx_parent,                      # DOM: parent
    "s_grandparent": _grandparent,                 # DOM: ancestor (2 levels)
    "s_prev_sibling": lambda pg, i: _sib(pg, i, True),    # DOM: sibling (preceding)
    "s_next_sibling": lambda pg, i: _sib(pg, i, False),   # DOM: sibling (following)
    "s_prev_in_order": _prev_node,                 # DOM: tree order (previous node)
    "s_next_in_order": _next_node,                 # DOM: tree order (next node)
    "s_tag_path": _tag_path,                       # DOM: ancestor chain (element names)
}
TYPES_SEM = {
    "m_row_header": lambda pg, i: C.rel_parts(pg, i)["row"],     # HTML: header cell, scan left in row
    "m_col_header": lambda pg, i: C.rel_parts(pg, i)["col"],     # HTML: header cell, scan up in column
    "m_prev_label": lambda pg, i: C.rel_parts(pg, i)["label"],   # HTML: name-value / caption (positional)
    "m_heading": lambda pg, i: C.rel_parts(pg, i)["head"],       # HTML: heading of the section
    "m_dt_dd": ctx_dtdd,                                          # HTML: dl name-value group (tag)
    "m_label_for": ctx_label_for,                                 # HTML: label element (tag)
    "m_caption": ctx_caption,                                     # HTML: table caption (tag)
    "m_attrs": lambda pg, i: ctx_attrs(pg, i, EXT_ATTR_KEYS),    # HTML: title / alt / aria-label / itemprop
}
TYPES = {**TYPES_STRUCT, **TYPES_SEM}


def type_text(pg, name, i):
    return _cap(TYPES[name](pg, i), TYPE_CAP)


# ------------------------------------------------------------------ G: chunk baseline (GT-free)
def chunks(pg, size=64, stride=32):
    words = page_text(pg).split()
    if not words:
        return []
    out, s = [], 0
    while True:
        out.append(" ".join(words[s:s + size]))
        if s + size >= len(words):
            break
        s += stride
    return out


def chunk_top(pg, question, budget=64, seed="0", size=64, stride=32):
    cs = chunks(pg, size, stride)
    if not cs:
        return ""
    bm = L.BM25([L.tokenize(c) for c in cs])
    sc = bm.scores(L.tokenize(question))
    rng = random.Random(seed)
    perm = [rng.random() for _ in cs]
    order = sorted(range(len(cs)), key=lambda k: (-sc[k], perm[k]))
    out, used = [], 0
    for k in order:
        w = cs[k].split()[:max(0, budget - used)]
        if not w:
            break
        out.append(" ".join(w))
        used += len(w)
    return " | ".join(out)


# ------------------------------------------------------------------ E: GT-free question perturbation
def page_relation_vocab(pg, cands):
    v = pg.__dict__.get("_c12_relvocab")
    if v is None:
        v = set()
        for i in cands:
            v |= set(toks(C.rel_string(pg, i)))
        pg.__dict__["_c12_relvocab"] = v
    return v


def perturb(question, vocab, page_tokens, synonyms):
    """Replace question content words found in the page's relation vocabulary with a synonym
    that does not occur on the page. `synonyms(word) -> list[str]`. Returns (new_q, n_replaced)."""
    out, n = [], 0
    for w in re.findall(r"\w+|\W+", question):
        lw = w.lower()
        if lw.isalpha() and lw not in STOP and lw in vocab:
            rep = next((s for s in synonyms(lw) if s.isalpha() and s != lw and s not in page_tokens), None)
            if rep:
                out.append(rep)
                n += 1
                continue
        out.append(w)
    return "".join(out), n


def wordnet_synonyms():
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("price")
    except Exception as e:  # pragma: no cover - depends on local install
        raise SystemExit("[perturb] needs nltk + wordnet: pip install nltk; "
                         "python -c \"import nltk; nltk.download('wordnet')\"  (" + str(e) + ")")
    cache = {}

    def syn(word):
        if word not in cache:
            seen, out = set(), []
            for s in wn.synsets(word):
                for lem in s.lemmas():
                    n = lem.name().lower()
                    if "_" in n or n == word or n in seen:
                        continue
                    seen.add(n)
                    out.append(n)
            cache[word] = out
        return cache[word]
    return syn
