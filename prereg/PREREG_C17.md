# PREREG C17 — a learned structure-aware model (MarkupLM) as a node ranker

Written before any C17 output.

## Why
The paper argues: do not feed structure as-is; read the value's meaning off the structure and attach it.
The obvious question is whether a model that learned structure end-to-end does better. MarkupLM
(microsoft/markuplm-base-finetuned-websrc) reads text with XPath embeddings and was fine-tuned on WebSRC.

## Fixed
- Sets: dev and official Test only (MarkupLM was fine-tuned on the WebSRC training split, which contains
  our train/select/confirm sites). Sampling as C12/C14/C16.
- Strings: document-order text strings with Hugging-Face-style XPath; script/style/noscript skipped.
  Windows of whole strings that fit 512 tokens with the question, 50% overlap.
- Node score: for each start token the best end within 30 tokens; span score = start + end logit; the span
  goes to the lowest common tid ancestor of its first and last string; node score = best span score;
  nodes without a span tie at the bottom (random tie expectation, as everywhere).
- References on the same questions: BM25 text only, BM25 + own semantic relations (C14 settings),
  reciprocal-rank fusion (k = 60) of MarkupLM with each.

## Tests (Holm over L1, L2)
- L1 BM25 + own semantic relations - MarkupLM (Recall@30)
- L2 RRF(MarkupLM, BM25 + own) - MarkupLM (Recall@30)
Secondary: MarkupLM - BM25 text only; L1 and L2 on Hit@1.

## Expectations (no strong prior)
- MarkupLM beats BM25 text only on Recall@30 (it reads structure).
- Hit@1: MarkupLM is likely better than BM25 + own (it is a trained reader; our Hit@1 is low).
- L2 > 0: adding own semantic relations by fusion helps MarkupLM's candidate list.
- L1 may go either way; both outcomes are reported. This is a node-ranking use of MarkupLM, not the
  span-extraction task it reports in its paper, so numbers are not comparable to MarkupLM's EM/F1.
