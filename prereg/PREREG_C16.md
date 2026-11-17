# PREREG C16 — reviewer questions: relation count, previous node, structure on top, donor seed

Written before any C16 output. Same data, sites, sampling, rules (REL v1), BM25 settings (k1 1.5,
b 0.75, per-page statistics, lambda 1) and 64-token cap as C14. Test retrieval levels for conditions
(1)-(7) were seen in C14; every C16 condition below is new on every set (post hoc with respect to C14,
fixed here before running).

## Conditions (GT-free)
- struct_all: a1 parent + a2 two previous/next siblings + a3 previous and next node (document order)
  + a4 tag path, word-deduplicated, cut to 64 tokens. BM25 also: cut to own-relation length (matched).
- struct_all_nosem: struct_all with every token of the node's own semantic relations removed.
- all8: own semantic relations followed by struct_all, deduplicated, cut to 64 tokens.
- prev / prev_nosem: previous node in document order, with and without semantic tokens.
- other-node control with donor seeds 13, 17, 23 (BM25).
Retrievers: BM25, BGE-small-en-v1.5, BGE-large-en-v1.5.

## Tests (per retriever and set; Holm within K1-K3)
- K1 own - struct_all       (four semantic vs four structural: relation count)
- K2 struct_all - struct_all_nosem
- K3 prev - prev_nosem
Secondary: all8 - own, struct_all_nosem - text, struct_all - text, BM25 matched versions, own - other by seed.
Pass rule: site CI > 0, site->page CI > 0, more positive than negative sites.

## Predictions
- K1 passes on Test for all three retrievers; on dev/select/confirm it may not (C14 own - parent was
  only 4-5 points there).
- K2 and K3 pass in at least three of four sets for BM25.
- struct_all_nosem - text is <= 0 on Test for BM25 (C14: parent - semantic was below text only).
- all8 - own is small (|diff| < 5 points) and may be negative on Test (C14: + parent was -4.6).
- own - other passes for all three donor seeds on all sets.
All results are reported, pass or fail.
