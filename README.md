# What to Attach to a DOM Node: Supplementary Code

Code for the paper's experiments on WebSRC node retrieval. Every ranking is ground-truth free: a
retriever scores every text-bearing DOM node of a page, and the gold node is used only to compute
metrics.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader wordnet                     # only for the synonym analysis
cd code
python -m unittest test_c4 test_c8 test_c9 test_c12 test_c14 test_c15 test_c16 test_c17
```

The tests build small synthetic pages, so they need no data and no GPU. Two tests are skipped
unless a reader model is cached locally.

## Data

Download WebSRC v1.0 (https://x-lance.github.io/WebSRC/) and unpack it so that each split is a folder
of site directories, each holding `dataset.csv` and `processed_data/*.html`:

```
data/official_train/   # WebSRC train release (train / select / confirm sites)
data/official_data/    # WebSRC dev release (our dev sites)
data/official_test/    # WebSRC test release, prepared with c10_prepare_test.py
```

`c10_prepare_test.py` unzips the official test archive and merges the official answers into each site's
`dataset.csv`.

## Site splits

The splits are by website, so no template is shared between them.

| Split | Sites |
|---|---|
| dev | au01, au08, bo17, ga10, ga12, jo11, mo02, ph04, sp09, sp10 |
| select | bo10, ga09, jo05, jo10, mo07, ph02, sp12, sp15, un11, un12 |
| confirm | au09, au12, bo09, ca02, sp01, sp08, sp14, un03, un04, un07 |
| official Test | au02, au14, ga06, ga08, ga38, ho07, ho08, mo08, sp16, un08 |

The remaining train sites are used only to develop the relation rules. `c4_run.py split` regenerates
the train / select / confirm assignment deterministically from the site names:

```bash
python c4_run.py split --train-data ../data/official_train --out ../results/split.json
```

Each run samples at most 100 questions per site and at most 3 per page, with a fixed seed.

## Reproducing the paper

All commands run from `code/`. Below, `DEV=au01,au08,bo17,ga10,ga12,jo11,mo02,ph04,sp09,sp10`,
`SP=../results/split.json` and `DM=BAAI/bge-small-en-v1.5,BAAI/bge-large-en-v1.5`. Replace `--role test`
with `--role dev --data ../data/official_data --sites $DEV` or with
`--role select|confirm --data ../data/official_train --split-file $SP` for the other splits.

| Paper | Script | Command (official Test) |
|---|---|---|
| Table 3, Table 4, App. A8, A10 (semantic vs structural, semantic words removed, previous node, 8-relation union, donor seeds) | `c16_run.py` | `python c16_run.py --data ../data/official_test --role test --dense $DM --out ../results/c16/test` |
| App. A2 (seven basic conditions, token-matched contexts, question-matched donor) | `c14_run.py` | `python c14_run.py --data ../data/official_test --role test --dense $DM --out ../results/c14/test` |
| Table 5, App. A1 (each relation: own vs another node), App. A4 (leave-one-out), App. A3 and A5 (dense retriever, WordNet synonyms), App. A7 answer inclusion and 64-word chunk baseline | `c12_run.py` | `python c12_run.py --data ../data/official_test --role test --dense BAAI/bge-small-en-v1.5 --perturb --out ../results/c12/test` |
| App. A7 generative F1 (Qwen2.5-1.5B-Instruct) | `c12_gen.py` | `python c12_gen.py --data ../data/official_test --role test --out ../results/c12/gen_test` |
| Extractive reader (RoBERTa, SQuAD 2.0) | `c9_qa.py` | `python c9_qa.py --data ../data/official_test --role test --out ../results/c9/test` |
| Table 6 (rules vs explicit HTML markup) | `c15_map_audit.py` | `python c15_map_audit.py --data ../data/official_test --role test --out ../results/c15/test` |
| Table 7, App. A8-5 (MarkupLM and rank fusion; dev and Test only) | `c17_markuplm.py` | `python c17_markuplm.py --data ../data/official_test --role test --out ../results/c17/test` |
| App. A9 (relation weight λ, train sites) and App. A6 (ties, candidate length) | `c8_robust.py` | `python c8_robust.py --data ../data/official_train --role train --split-file $SP --diverse --out ../results/c8/train` |

`c14_collect.py`, `c16_collect.py` and `c12_collect.py` gather the per-split summaries into one Markdown
report. Every script writes a `*_summary.json` with a hash of the code that produced it.

## Code map

| File | Role |
|---|---|
| `idc_lib.py` | WebSRC loading, page parsing (one node per `tid`), tokenisation, per-page BM25 |
| `c4_lib.py` | The four semantic-relation rules (b1 row header, b2 column header, b3 preceding label, b4 section heading), relation field scoring, site-cluster bootstrap |
| `c12_lib.py` | Structural relations (a1 parent, a2 siblings, a3 previous / next node in document order, a4 tag path), removal of semantic words, chunk baseline |
| `c8_robust.py` | Random-tie metrics (expected Hit@k), Holm correction |
| `c9_common.py` | Page sampling and the site→page two-level bootstrap |
| `c16_run.py` | Union of the four structural relations; a3 uses both the previous and the next node there, and the stand-alone a3 condition uses the previous node only |

## Settings

- BM25: k1 = 1.5, b = 0.75. IDF and average length come from the node texts of the same page, and the
  relation field uses the same statistics.
- Score: S(q, v) = BM25(q, text(v)) + λ · BM25(q, relations(v)), with λ = 1.
- Relation fields are cut at 64 whitespace tokens, or 16 tokens per relation in the single-relation runs.
- Dense retrievers embed `node text[:64] + " | " + relation field`. No retriever is fine-tuned.
- Ties are scored by their expectation under a uniformly random tie order.
- A comparison passes when both the site-cluster and the site→page 95% bootstrap intervals are above
  zero and more sites gain than lose. Holm correction is applied within each primary family.

## Pre-registration

`prereg/` holds the analysis plans, each written before the corresponding run: the predictions, the
primary tests and the pass rule. They are in Korean. The file name gives the experiment ID used in
the code (C14, C16, …).

## License

MIT.
