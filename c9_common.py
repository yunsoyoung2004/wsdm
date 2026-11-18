# -*- coding: utf-8 -*-
"""Shared helpers for C8/C9: diverse page sampling and site->page two-level bootstrap."""
from __future__ import annotations

import random
from collections import OrderedDict, defaultdict

import c4_lib as C
import idc_lib as L


def load_diverse(data, sites, per_site=100, max_per_page=3, seed=2026, verbose=True):
    """Random pages per site (fixed seed), at most `max_per_page` questions per page.
    Removes the 'first-N questions in CSV order' bias (few pages, many paraphrases)."""
    from pathlib import Path
    rows, skipped, _ = L.load_rows(Path(data), sites, None, None, verbose=verbose)
    by_site = defaultdict(lambda: OrderedDict())
    for r in rows:
        by_site[r.site].setdefault(r.page_key, []).append(r)
    out = []
    for site in sorted(by_site):
        pages = list(by_site[site].items())
        rng = random.Random(f"{seed}|{site}")
        rng.shuffle(pages)
        picked = []
        for pk, qs in pages:
            qs = sorted(qs, key=lambda q: q.qid)
            rng.shuffle(qs)
            picked += qs[:max_per_page]
            if len(picked) >= per_site:
                break
        out += picked[:per_site]
    if verbose:
        n_pages = len({r.page_key for r in out})
        print(f"[diverse] {len(out)} questions from {n_pages} pages "
              f"(<= {max_per_page}/page, seed {seed})")
    return out, skipped


def two_level_boot(diffs, B=3000, seed=11):
    """diffs: list of (site, page, d). Resample sites, then pages within each sampled site."""
    tree = defaultdict(lambda: defaultdict(list))
    for s, p, d in diffs:
        tree[s][p].append(d)
    sites = sorted(tree)
    flat = [d for _, _, d in diffs]
    est = C.mean(flat)
    if len(sites) < 2:
        return est, float("nan"), float("nan")
    rng = random.Random(seed)
    bs = []
    for _ in range(B):
        vals = []
        for s in (rng.choice(sites) for _ in sites):
            pages = list(tree[s])
            for p in (rng.choice(pages) for _ in pages):
                vals += tree[s][p]
        bs.append(C.mean(vals))
    bs.sort()
    return est, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
