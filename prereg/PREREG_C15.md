# PREREG C15 — accuracy of the structure-to-meaning mapping

Written before any C15 output. Descriptive audit; no hypothesis test on retrieval.

## Why
The paper's first contribution is a mapping from DOM positions to HTML-Standard semantic relations
(row header, column header, preceding label, section heading). Its accuracy has never been measured.
Where pages carry explicit markup (<th>/scope/headers, <dt>/<dd>, <label for>, sectioning elements),
that markup gives a gold relation without human labels.

## Fixed
- Rules: C4 REL v1, unchanged. Sampling: as C12/C14 (same sites, 100 q/site, <= 3/page, seed 2026).
- Gold (simplified HTML-Standard header assignment on the expanded table grid):
  column headers = <th> above in the same column, not scope=row/rowgroup;
  row headers = <th> left in the same row, not scope=col/colgroup; headers="..." overrides both.
  Label gold = <dt> of the enclosing <dd>, or <label for=id>. Heading gold = first heading of the nearest
  <section>/<article>/<aside>/<nav>, if it precedes the node.
- Metrics per relation, on all candidate nodes and on gold answer nodes:
  coverage (rule non-empty | gold exists), exact (normalised rule text equals a gold text),
  exact given a prediction, token F1, and rule_only (rule non-empty | no gold in a marked table).
  rule_only is NOT an error rate: unmarked first-column <td> cells are often real row headers.
- Markup prevalence: share of tables with any <th>; share of answer nodes inside a <th> table.

## Expectations (stated before running)
- Fewer than half of the tables carry <th> (the paper says explicit markup is rare).
- Column header: exact >= 0.6 on answer nodes in marked tables (rule = cell of the first row).
- Row header: coverage high, exact lower where tables use scope=row deeper than the first column.
- Label (dt/dd, label[for]) and section heading: few gold cases; reported as is.
All numbers are reported whatever they are.
