# PREREG C14 — the seven conditions of Figure 1, one run, three retrievers

Written before any C14 output was produced. Code hash is printed by every run; this document is fixed
once the first set is run.

## Why
Table 2 mixes two runs (C8 for structural baselines and dev, C12 for the rest) and has empty cells:
BM25 structural baselines on the official Test, condition (7) "parent − semantic words" at the main
budget, and dense siblings / tag path / (7). C14 fills the whole matrix in one code version and adds
a larger dense retriever (BGE-large, the model HtmlRAG used).

## What is fixed (not tuned)
- Data, sites, sampling: as C12 (WebSRC v1.0; dev / select / confirm / official Test; 100 questions
  per site, ≤ 3 per page, seed 2026; yes/no and node-less questions removed).
- Relation rules: REL_VERSION v1 (unchanged since C4). BM25 k1 = 1.5, b = 0.75, per-page statistics, λ = 1.
- Fields (GT-free), each cut to CAP = 64 tokens:
  (1) text only, (2) own semantic relations, (3) random other node's relations, (3b) question-matched
  other node's relations, (4) parent text (own text removed), (5) two previous + two next siblings,
  (6) tag path = ancestor element names only (no attributes), (7) parent with every token of the
  node's own semantic relations removed.
- BM25 also reports each structural field cut to the length of the node's own relations ("matched").
- Dense: node text[:64] + " | " + field; models BAAI/bge-small-en-v1.5 and BAAI/bge-large-en-v1.5.
  Dense (3) uses a page-level donor (as C12); BM25 (3) uses the C4 per-question donor.
- Metric: Recall@30, expected value under random tie-breaking; Hit@1 reported.

## Tests (per retriever, per site set; Holm within P1–P5 of one retriever and set)
- P1  own − parent
- P2  own − (parent − semantic words)
- P3  parent − (parent − semantic words)
- P4  own − another node's (replication of C8/C12)
- P5  previous node (document order) − the same with the node's semantic-relation tokens removed
  (C12 found the previous node helps on Test, +15.3; P5 asks whether that is again the semantic words inside it)
Secondary: own − siblings, own − tag path, tag path − text, (7) − text, matched versions, own − question-matched.
Pass rule (as before): site CI > 0, site→page CI > 0, more positive than negative sites.

## Predictions
- P2, P3, P4, P5 pass in at least three of the four sets for BM25 and for both dense models.
- P1 is positive but may not pass: in C8 the BM25 own − parent difference was +4.1 to +5.2 points with
  site CIs that included 0. A non-pass of P1 does not count against the claim; P2/P3 carry it.
- (6) tag path − (1) text is near 0 for BM25 (|diff| < 3 points in at least three sets).
- BGE-large text-only beats BGE-small text-only, and own semantic relations still add to BGE-large.

## Status of the official Test
Retrieval levels for (1)–(3) on Test were already seen in C12. Conditions (4)–(7) on Test have never
been run, so their Test cells are first looks. All C14 analyses are reported in full, pass or fail.
