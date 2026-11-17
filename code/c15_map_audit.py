# -*- coding: utf-8 -*-
"""
C15: how accurately do the positional rules read the HTML-Standard semantic relations off the DOM?
(pre-registered in PREREG_C15.md). No human labels: gold comes from explicit HTML markup, where it exists.

  b1 row header / b2 column header
      gold = header cells assigned to a data cell by explicit table markup: <th> cells on a table grid
      (colspan/rowspan expanded), honouring scope=row|col|rowgroup|colgroup and headers="id ...".
      Simplified HTML-Standard algorithm: column headers = <th> above in the same column that are not
      scope=row; row headers = <th> to the left in the same row that are not scope=col.
      Evaluated only on <td> cells of tables that contain at least one <th> ("marked tables").
  b3 preceding label
      gold = <dt> for a node inside the paired <dd>; <label for=id> text for the node with that id.
  b4 section heading
      gold = first heading (h1-h6) of the nearest explicit sectioning element (section/article/aside/nav)
      that precedes the node inside that element. Evaluated only where such a heading exists.

  Also reports how often the explicit markup exists at all (share of tables with <th>, share of gold
  answer nodes in a marked table) -- the paper says such markup is rare; this measures it.

Rule outputs are C4 REL v1 (c4_lib.rel_parts), unchanged. Gold answers are used only to pick the
"answer nodes" subset.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import c4_lib as C
import c9_common as S
import idc_lib as L

SECTIONING = ("section", "article", "aside", "nav")
HEADS = ("h1", "h2", "h3", "h4", "h5", "h6")


def norm(s):
    s = re.sub(r"\s+", " ", (s or "").lower()).strip()
    return re.sub(r"[\s:：\-–—|·•]+$", "", s).strip()


def toks(s):
    return L.tokenize(s or "")


def tok_f1(pred, gold):
    p, g = Counter(toks(pred)), Counter(toks(gold))
    if not p or not g:
        return float(not p and not g)
    common = sum((p & g).values())
    if not common:
        return 0.0
    pr, rc = common / sum(p.values()), common / sum(g.values())
    return 2 * pr * rc / (pr + rc)


# ------------------------------------------------------------------ table grid (own rows only)
def own_rows(table):
    return [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]


def build_grid(table):
    """cell -> (r, c, rowspan, colspan); slot (r, c) -> cell."""
    pos, slot = {}, {}
    for r, tr in enumerate(own_rows(table)):
        c = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while (r, c) in slot:
                c += 1
            try:
                rs = max(1, min(int(cell.get("rowspan", 1) or 1), 50))
                cs = max(1, min(int(cell.get("colspan", 1) or 1), 50))
            except (TypeError, ValueError):
                rs = cs = 1
            pos[id(cell)] = (r, c, rs, cs, cell)
            for dr in range(rs):
                for dc in range(cs):
                    slot[(r + dr, c + dc)] = cell
            c += cs
    return pos, slot


def gold_headers(table, cell, cache):
    """(row_headers, col_headers) as lists of texts for a data cell, from explicit markup."""
    key = id(table)
    if key not in cache:
        pos, slot = build_grid(table)
        ids = {}
        for _, _, _, _, c in pos.values():
            if c.get("id"):
                ids[str(c.get("id"))] = c
        cache[key] = (pos, slot, ids)
    pos, slot, ids = cache[key]
    if id(cell) not in pos:
        return [], []
    hdr = cell.get("headers")
    if hdr:
        refs = [ids[h] for h in (hdr if isinstance(hdr, list) else str(hdr).split()) if h in ids]
        rows, cols = [], []
        for h in refs:
            hr, hc = pos[id(h)][0], pos[id(h)][1]
            (rows if hr == pos[id(cell)][0] else cols).append(L._t(h))
        return rows, cols
    r, c, rs, cs, _ = pos[id(cell)]
    cols, rows, seen = [], [], set()
    for rr in range(r - 1, -1, -1):                      # upwards in the same column
        h = slot.get((rr, c))
        if h is not None and h.name == "th" and id(h) not in seen and (h.get("scope") or "").lower() not in ("row", "rowgroup"):
            seen.add(id(h))
            cols.append(L._t(h))
    for cc in range(c - 1, -1, -1):                      # leftwards in the same row
        h = slot.get((r, cc))
        if h is not None and h.name == "th" and id(h) not in seen and (h.get("scope") or "").lower() not in ("col", "colgroup"):
            seen.add(id(h))
            rows.append(L._t(h))
    return [t for t in rows if t], [t for t in cols if t]


def gold_label(el):
    dd = el if el.name == "dd" else el.find_parent("dd")
    if dd is not None:
        dt = dd.find_previous_sibling("dt")
        if dt is not None and L._t(dt):
            return L._t(dt), "dt_dd"
    return None, None


def label_for_map(root):
    m = {}
    for lab in root.find_all("label"):
        f = lab.get("for")
        if f and L._t(lab):
            m.setdefault(str(f), L._t(lab))
    return m


def gold_section_heading(el, order):
    """First heading of the nearest explicit sectioning ancestor, if it precedes the node."""
    sec = el.find_parent(SECTIONING)
    if sec is None:
        return None
    h = sec.find(HEADS)
    if h is None or h is el or el in h.parents or order.get(id(h), 1 << 30) >= order.get(id(el), -1):
        return None
    return L._t(h) or None


def score_pair(pred, golds):
    """pred: rule text; golds: list of gold texts (any one counts as an exact hit)."""
    gn = [norm(g) for g in golds if norm(g)]
    pn = norm(pred)
    return {"pred_nonempty": int(bool(pn)), "exact": int(bool(pn) and pn in gn),
            "tok_f1": round(max((tok_f1(pred, g) for g in golds), default=0.0), 4)}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--role", required=True, choices=["dev", "select", "confirm", "test"])
    ap.add_argument("--split-file")
    ap.add_argument("--sites")
    ap.add_argument("--per-site", type=int, default=100)
    ap.add_argument("--max-per-page", type=int, default=3)
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

    recs = []                       # one row per (node, relation) with gold
    tables_total = tables_marked = 0
    ans_total = ans_in_table = ans_in_marked = 0
    for pk, qs in by_page.items():
        pg = L.parse_page(qs[0].html_path.read_text(encoding="utf-8", errors="replace"))
        gold_nodes = {pg.tid2i.get(r.element_id) for r in qs}
        gold_nodes.discard(None)
        root = pg.els[0]
        while root.parent is not None:
            root = root.parent
        tabs = root.find_all("table")
        tables_total += len(tabs)
        marked = {id(t) for t in tabs if any(th.find_parent("table") is t for th in t.find_all("th"))}
        tables_marked += len(marked)
        lfor = label_for_map(root)
        order = {id(t): k for k, t in enumerate(root.descendants)}
        cache = {}
        for i, el in enumerate(pg.els):
            if not pg.key[i]:
                continue
            is_ans = int(i in gold_nodes)
            parts = C.rel_parts(pg, i)
            cell = el if el.name in ("td", "th") else el.find_parent(["td", "th"])
            if is_ans:
                ans_total += 1
                if cell is not None:
                    ans_in_table += 1
                    t = cell.find_parent("table")
                    ans_in_marked += int(t is not None and id(t) in marked)
            # b1/b2 on data cells of marked tables
            if el.name == "td":
                t = el.find_parent("table")
                if t is not None and id(t) in marked:
                    gr, gc = gold_headers(t, el, cache)
                    for rel, pred, golds in (("row", parts["row"], gr), ("col", parts["col"], gc)):
                        recs.append({"page": pk, "rel": rel, "answer": is_ans, "gold_exists": int(bool(golds)),
                                     **score_pair(pred, golds)})
            # b3 label
            g, src = gold_label(el)
            if g is None and el.get("id") and str(el.get("id")) in lfor:
                g, src = lfor[str(el.get("id"))], "label_for"
            if g is not None:
                recs.append({"page": pk, "rel": "label", "src": src, "answer": is_ans, "gold_exists": 1,
                             **score_pair(parts["label"], [g])})
            # b4 heading
            gh = gold_section_heading(el, order)
            if gh:
                recs.append({"page": pk, "rel": "head", "answer": is_ans, "gold_exists": 1,
                             **score_pair(parts["head"], [gh])})
        del pg

    def summ(rs):
        if not rs:
            return {"n": 0}
        has = [r for r in rs if r["gold_exists"]]
        no = [r for r in rs if not r["gold_exists"]]
        out = {"n": len(rs), "n_gold": len(has)}
        if has:
            out.update({"coverage": round(C.mean(r["pred_nonempty"] for r in has), 4),
                        "exact": round(C.mean(r["exact"] for r in has), 4),
                        "exact_if_predicted": round(C.mean(r["exact"] for r in has if r["pred_nonempty"]), 4)
                        if any(r["pred_nonempty"] for r in has) else None,
                        "tok_f1": round(C.mean(r["tok_f1"] for r in has), 4)})
        if no:
            out.update({"n_no_gold": len(no), "rule_without_markup": round(C.mean(r["pred_nonempty"] for r in no), 4)})
        return out

    res = {}
    for rel in ("row", "col", "label", "head"):
        rs = [r for r in recs if r["rel"] == rel]
        res[rel] = {"all_nodes": summ(rs), "answer_nodes": summ([r for r in rs if r["answer"]])}
    markup = {"tables": tables_total, "tables_with_th": tables_marked,
              "share_tables_with_th": round(tables_marked / tables_total, 4) if tables_total else None,
              "answer_nodes": ans_total, "answer_in_table": ans_in_table, "answer_in_th_table": ans_in_marked,
              "share_answers_in_table": round(ans_in_table / ans_total, 4) if ans_total else None,
              "share_answers_in_th_table": round(ans_in_marked / ans_total, 4) if ans_total else None}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": "C15 mapping accuracy (PREREG_C15.md)", "role": a.role, "rel_version": C.REL_VERSION,
               "n_pages": len(by_page), "markup": markup, "accuracy": res}
    (out / "c15_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== C15 [{a.role}] pages={len(by_page)} rel={C.REL_VERSION} ===")
    print("explicit markup:", json.dumps(markup))
    print(f"\n{'relation':8s}{'subset':14s}{'n_gold':>8s}{'cover':>8s}{'exact':>8s}{'ex|pred':>9s}{'tokF1':>8s}{'n_nogold':>10s}{'rule_only':>10s}")
    for rel, d in res.items():
        for sub, s in d.items():
            f = lambda k: (f"{s[k]:.3f}" if isinstance(s.get(k), float) else ("-" if s.get(k) is None else str(s.get(k))))
            print(f"{rel:8s}{sub:14s}{s.get('n_gold', 0):>8}{f('coverage'):>8s}{f('exact'):>8s}{f('exact_if_predicted'):>9s}"
                  f"{f('tok_f1'):>8s}{s.get('n_no_gold', 0):>10}{f('rule_without_markup'):>10s}")
    print(f"[saved] {out.resolve()}")


if __name__ == "__main__":
    main()
