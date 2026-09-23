# Graph feature contract

Contract version: **2**, implemented on branch `codex/brief-data`.

## Public interface and handoff

```python
from src.loader import load_inputs
from src.graph_features import build_features, build_graph, percentile_rank

nodes, edges, transactions = load_inputs("data")
features = build_features(nodes, edges, transactions)
```

`build_features` validates copies of its inputs and returns **86 feature columns**
with one row per exact signed-int64 `gid` in an index named `gid`. It preserves
the supplied node row order, including isolates. The supplied dataset produces
2,248 rows. Reusable functions also accept smaller or empty valid datasets;
profile counts are documented separately in [data-quality.md](data-quality.md).

All existing feature names remain available. Version 2 adds availability flags,
complete-window temporal denominators, bounded date-pattern evidence, amount
observations, and individual peer-anomaly components. `relay_2d_ratio` deliberately
changes its denominator to exclude incomplete follow-up windows. Consumers must
honor the validity flags rather than replacing unavailable ratios with observed
zeros. `boundary_censored` remains the feature-layer boundary name; any
`truncated_by_depth`, `retention`, `balance_score`, role or priority fields are
owned by the downstream pipeline/decision engine.

Member 2 should consume the shared `percentile_rank` helper and explicitly define
downstream missing-value behavior. This change does not modify `src/roles.py` or
ranking weights. Member 3 should display the explanations/flags and consume the
auxiliary feature artifact produced by Member 2; this layer writes no CSVs.

## Validation and numeric types

- Input columns: nodes `(gid, depth, is_seed)`; edges
  `(src, dst, sum_kzt, n_tx, depth)`; transactions `(src, dst, date, sum_kzt)`.
  Duplicate column names are rejected; extra columns are omitted in normalized
  copies. Neither validation nor feature construction mutates caller tables.
- Gids/endpoints become signed `int64`. Integer values and signed decimal
  integer strings are parsed without float conversion. Integral floats are
  accepted only when `abs(value) < 2**53`; larger floats might already be rounded
  and are rejected. Booleans, fractions, nulls, nonnumeric and out-of-range values
  are rejected. Node uniqueness is checked after normalization. Negative and
  zero identifiers are allowed. All endpoint references must exist.
- Node depths are integers 0–4; edge discovery depths are integers 1–4. Edge
  depth is sampling metadata, not an assertion that endpoint depths differ by
  one. Seed flags accept booleans or 0/1. No supplied-dataset histogram is imposed.
- Amounts become finite positive `float64`; booleans/complex values are rejected.
  Edge `n_tx` becomes positive int64. Each directed pair occurs only once in
  edges. Transaction pair sums use sorted `math.fsum`; pair counts must match
  exactly. Amounts use `np.isclose(edge_sum, transaction_sum, rtol=1e-6,
  atol=1e-6)`. Tolerances must be finite and nonnegative. Overflowing aggregate
  or node amount totals produce validation errors rather than infinite features.
- Dates are parsed as UTC, normalized to midnight and made timezone-naive.
  Numeric epoch-like inputs and invalid/missing dates are rejected. With an
  offset-bearing timestamp this is the UTC date; the supplied input contains
  only ISO calendar dates. No intraday order is retained or inferred.
- No general function enforces July 2026 or a 5,000 KZT threshold. Those describe
  the supplied extract; tests can use other dates and positive amounts.

Feature counts/degrees/IDs are `int64`; amounts, centralities, ratios and scores
are `float64`; flags are `bool`; text is pandas string dtype; feature dates are
timezone-naive `datetime64[ms]` for stable Parquet round-trips. `NaN` means an
unavailable numeric value and `NaT` an unavailable date. Evidence uses JSON
strings rather than Python containers; gids inside JSON are exact decimal
strings for browser safety. Empty results retain the same typed schema.

## Determinism

The graph inserts numeric gids in ascending order and directed edges in ascending
`(src, dst)` order. Temporal aggregation sorts transactions by
`(src, dst, date, sum_kzt)`. Algorithms use canonical iteration, deterministic
date ties and bounded evidence sorting. Align results by gid before comparing
different input node permutations; output row order intentionally follows the
node input. Tests require exact equality for repeated and shuffled supplied
tables, and exact Parquet round-trip equality.

`build_graph` expects validated int64 tables. It includes isolates and exposes
directed edge attributes `sum_kzt`, `n_tx`, `depth`, and
`distance = 1 / log1p(sum_kzt)`. Amount is therefore converted to a positive
distance before weighted shortest-path centrality.

Reproducibility across environments additionally requires pinned dependencies.
Canonical graph construction does not remove another consumer's responsibility
to preserve ordering when constructing its own projection.

## Base, centrality, reach and cycle fields

| Fields | Type | Meaning / denominator / availability |
|---|---|---|
| `depth`, `is_seed` | int64, bool | Validated node metadata. |
| `boundary_censored` | bool | Exactly `depth == 4`; zero out-degree does not establish terminal behavior. |
| `in_deg`, `out_deg` | int64 | Counts of distinct incoming/outgoing directed neighbors; self-loops count once on each side. |
| `in_kzt`, `out_kzt` | float64 KZT | Sums of observed incoming/outgoing aggregated edge amounts. Zero for isolates. |
| `in_tx`, `out_tx` | int64 | Sums of observed incoming/outgoing edge transaction counts. |
| `total_kzt`, `total_tx` | float64 KZT, int64 | Incoming plus outgoing values. Summing across nodes counts each graph transfer twice. |
| `pagerank` | float64 | Amount-weighted directed PageRank, damping 0.85, uniform teleportation and dangling redistribution; maximum 100 iterations, total-change tolerance `n * 1e-6`. Failure to converge raises. |
| `betweenness` | float64 | Exact normalized directed NetworkX betweenness with `weight="distance"`; endpoints excluded. No interpretation as criminality. |
| `pass_through` | float64 | `out_kzt / in_kzt`, only for nonseed, nonboundary nodes with positive observed inflow. May exceed 1; never a full balance. Otherwise NaN. |
| `pass_through_valid` | bool | Whether the preceding ratio is available. |
| `pass_through_invalid_reason` | string | Empty when valid; otherwise `seed_inbound_incomplete`, `boundary_outbound_incomplete`, or `no_inbound_activity`, in that precedence order. |
| `fanin_share`, `fanout_share` | float64 | `in_deg / (in_deg + out_deg)` and `out_deg / (in_deg + out_deg)` for nonseed, nonboundary nodes with positive total degree. Otherwise NaN. These are neighbor shares, not amount shares. |
| `flow_share_valid` | bool | Availability of both degree-share ratios. |
| `seed_reach_count` | int64 | Number of distinct seeds reaching the node by directed paths of at most four edges, computed once by BFS per seed. A seed reaches itself at distance zero. |
| `seed_reach_fraction` | float64 | Reach count divided by the number of all supplied seeds, including isolates; zero when there are no seeds. |
| `min_seed_distance` | float64 hops | Minimum directed seed distance within cutoff four; NaN if unreachable within that cutoff. |
| `scc_id`, `scc_size` | int64 | Strongly connected component ID and size. IDs are assigned by sorting components by their smallest stringified gid; IDs are labels, not ranks. |
| `in_cycle` | bool | Exactly SCC size >1. A singleton self-loop is deliberately not included in this multi-node-cycle indicator. |
| `reciprocal_relationship_count` | int64 | Number of distinct other nodes with both directed relationships present. Self-loops excluded; each endpoint counts its reciprocal partner once. |

SCC and reciprocity describe structural possibilities, not evidence that the
same funds returned. No exhaustive cycle enumeration is performed.

## Temporal fields and observation limits

Let `I(g)` and `O(g)` be the dates with observed incoming/outgoing activity for
node g. Let `T` be the latest date anywhere in the supplied transactions, not an
assumed month end. All amounts are sums of rows on calendar dates.

The relay denominator is `E(g) = {D in I(g): D + 2 days <= T}`. A match is any
outgoing activity on D, D+1 or D+2. Even a positive observed match on a later
partial-window date is excluded from this ratio, keeping all denominator dates
comparable. Positive pattern evidence below may still show that observed event.
No event after T is inferred. Gaps within the global range describe absence in
this extract, not a guarantee of complete account activity.

| Fields | Type | Meaning / availability |
|---|---|---|
| `active_days` | int64 | Size of `I union O`; zero for isolates. |
| `inbound_active_days`, `outbound_active_days` | int64 | Sizes of I and O. Counts describe observed rows even when ratios are censored. |
| `same_day_flow_ratio` | float64 | Outgoing KZT on dates in I divided by all outgoing KZT. NaN for seeds, depth 4, or no outgoing amount. It is an amount overlap, not matched money. |
| `same_day_flow_valid` | bool | Whether same-day flow ratio is available. |
| `same_day_flow_invalid_reason` | string | Empty, or `seed_inbound_incomplete`, `boundary_outbound_incomplete`, `no_outgoing_activity`, in that order of precedence. |
| `relay_2d_eligible_days` | int64 | Size of E, the complete-follow-up denominator. |
| `relay_2d_matched_days` | int64 | Number of E dates with any outgoing D/D+1/D+2 activity; each inbound date counts at most once. |
| `relay_2d_censored_days` | int64 | Size of I minus size of E: omitted partial-window inbound dates. |
| `relay_2d_ratio` | float64 | Matched / eligible days, only for nonseed, nonboundary nodes with at least one eligible day. Otherwise NaN. No outgoing observations can yield a valid zero only when those conditions hold. |
| `relay_2d_valid` | bool | Whether the relay ratio is available. |
| `relay_2d_invalid_reason` | string | Empty, or `seed_inbound_incomplete`, `boundary_outbound_incomplete`, `no_inbound_activity`, `no_complete_followup_window`, in that precedence order. |
| `temporal_observation_start`, `temporal_observation_end` | datetime64[ms] | Global first/last available transaction dates, repeated per node; NaT for empty transactions. |
| `max_in_senders_day` | int64 | Maximum distinct incoming senders on a calendar date; zero without incoming rows. |
| `max_in_senders_date` | datetime64[ms] | Earliest date attaining that sender maximum; NaT without incoming rows. |
| `peak_activity_kzt` | float64 KZT | Maximum daily sum of incoming plus outgoing observed KZT; zero without activity. |
| `peak_activity_date` | datetime64[ms] | Earliest date attaining the peak amount; NaT without activity. |
| `peak_day_share` | float64 | Peak amount / total observed incoming-plus-outgoing amount. NaN for seeds or inactive nodes. At depth 4 this is the concentration of sampled activity only; it does not imply complete outgoing coverage. |
| `peak_day_share_valid` | bool | Whether this sampled-activity share is available. |

Self-transfers count on both the incoming and outgoing side. Sender maxima and
peak dates provide burst/synchronization observations; no universal AML burst
threshold is claimed. July 30–31 are the omitted follow-up dates for the supplied
July extract; other inputs use their own maximum date.

## Bounded route and return observations

The evidence is attached to intermediate/receiving node **B**:

- A repeated route A→B→C requires three distinct gids and at least **two distinct
  inbound dates D** with A→B on D and B→C on D/D+1/D+2.
- A temporal return candidate A→B→A requires distinct A and B and at least
  **one** such inbound date with B→A on D/D+1/D+2. This is a reciprocal date
  pattern, not tracing the original transfer or searching long cycles.
- For each inbound date choose the earliest matching outgoing date. Same-day
  matches have unknown order. Observed positive matches near the period end
  remain evidence, even though those dates are excluded from the relay ratio.
- Per node and per pattern type examine at most **512 numeric-lexicographic
  counterparty candidates** and **20,000 inbound-date probes**. These bound the
  search itself, not just the output. Truncated counts/support are lower bounds
  among observed patterns; untruncated counts still describe only the extract.
- Evidence is the top **three** detected candidates by support-day count
  descending, then numeric source/destination gid; each retains at most the
  earliest **three** date pairs. Not all qualifying patterns are displayed.

| Fields | Type | Meaning |
|---|---|---|
| `repeated_route_count` | int64 | Number of qualifying A→B→C candidates found within the bounds. |
| `repeated_route_max_support_days` | int64 | Largest qualifying support-day count found; zero if none. |
| `repeated_route_evidence` | string JSON | Bounded evidence records or `[]`. |
| `repeated_route_truncated` | bool | Candidate/date-probe budget omitted part of the route search. |
| `temporal_return_count` | int64 | Number of counterparties A with a qualifying A→B→A date observation. |
| `temporal_return_max_support_days` | int64 | Largest return support-day count found; zero if none. |
| `temporal_return_evidence` | string JSON | Bounded return records or `[]`. |
| `temporal_return_truncated` | bool | Candidate/date-probe budget omitted part of the return search. |

Each JSON record contains string gids `src`, `via`, `dst`, integer
`support_days`, ISO-string `date_pairs`, `in_kzt_on_support_dates`, and
`out_kzt_on_matched_dates`. Inbound totals sum all matched inbound dates once;
outbound totals sum **unique** selected outgoing dates once, even when several
inbound dates match the same outgoing date. They are daily observed volumes,
not matched amounts or evidence that particular funds moved.

These observed pattern counts/evidence can be shown for seeds or boundary nodes
when actual supporting rows exist. An absent pattern never establishes absence
of behavior outside the extract. Inspect sampling and truncation flags.

## Repeated and similar observed amounts

These descriptors support review of observed activity; none establishes
intentional splitting or detects transfers omitted below the extract threshold.

| Fields | Type | Meaning / denominator |
|---|---|---|
| `outgoing_repeated_amount_tx_count` | int64 | All outgoing transaction rows belonging to exact float amount groups with at least two rows anywhere in the observation window. Counts all group members, not only repeats after the first. |
| `outgoing_repeated_amount_tx_share` | float64 | That count / all observed outgoing rows; NaN when there are no outgoing rows. |
| `outgoing_similar_amount_tx_count` | int64 | Rows in qualifying disjoint same-day similar-amount groups defined below. |
| `outgoing_similar_amount_tx_share` | float64 | That count / all observed outgoing rows; NaN without outgoing rows. |
| `similar_amount_group_count` | int64 | Number of qualifying same-day groups. |
| `amount_pattern_evidence` | string JSON | At most three exact/similar groups, ranked by transaction count descending, then kind/date/amount. `[]` means none. |

For each outgoing date, sort amounts ascending, then destination gid. Start a
greedy group at the smallest ungrouped amount a and include subsequent rows
whose amount is at most **1.05a**, inclusive. Move to the next ungrouped row even
if the group does not qualify. A group qualifies when it contains at least
**three rows** sent to **two distinct destinations**. Groups cannot overlap;
this is a reproducible descriptive rule, not a claim to find every possible
close-amount combination. Exact repetition uses actual normalized float values,
not hidden cent rounding. Exact and similar row sets may overlap with each
other, so their shares must not be added.

Exact evidence fields: `kind="exact_repetition"`, `amount_kzt`, `n_tx`.
Similar evidence fields: `kind="same_day_similar_amounts"`, ISO `date`, `n_tx`,
`n_recipients`, `min_kzt`, `max_kzt`, and at most three smallest string
`recipient_gids`. Counts cover all groups even though evidence is limited.
Shares concern observed outgoing rows, so they may be present for seeds; they
make no assumption about seed inflows or missing boundary outgoing activity.

## Depth-peer anomaly decomposition

Within each comparable `depth`, for each metric x compute median m and
MAD = median(|x-m|). With MAD >0:

```text
z = 0.67448975 * abs(x - m) / MAD
component = z / (1 + z)
```

With MAD=0 the component is 0 at the median and 1 elsewhere. A singleton or
constant group therefore has zero deviations. Deviations can represent either
unusually high or unusually low observed activity; scores are not probabilities.

| Fields | Type | Input / meaning |
|---|---|---|
| `peer_anomaly_in_deg` | float64 [0,1] | Component for incoming degree. |
| `peer_anomaly_out_deg` | float64 [0,1] | Component for outgoing degree. |
| `peer_anomaly_log1p_in_kzt` | float64 [0,1] | Component for `log1p(in_kzt)`. |
| `peer_anomaly_log1p_out_kzt` | float64 [0,1] | Component for `log1p(out_kzt)`. |
| `peer_anomaly_in_tx` | float64 [0,1] | Component for incoming transaction count. |
| `peer_anomaly_out_tx` | float64 [0,1] | Component for outgoing transaction count. |
| `peer_anomaly_score` | float64 [0,1] | Arithmetic mean of the six components; each weight is 1/6. |
| `peer_group_size` | int64 | Number of supplied nodes at the same depth, including the node itself. |

## Percentile fields

`percentile_rank(series)` applies numeric coercion, treats nonfinite/unparseable
values as missing, then uses pandas average-tie ranks divided by the valid count.
NaN remains NaN, a singleton receives 1, and an all-missing input stays missing.
For n equal valid values each percentile is `(n+1)/(2n)` (e.g. 2/3 for three
equal values). The lowest valid value need not be zero. Rank describes position
in this sample, not risk or probability.

The following **18 float64 percentile columns** use all valid supplied nodes,
not depth peers:

```text
in_deg_pct, out_deg_pct, in_kzt_pct, out_kzt_pct, in_tx_pct, out_tx_pct,
total_kzt_pct, total_tx_pct, pagerank_pct, betweenness_pct,
seed_reach_count_pct, seed_reach_fraction_pct, active_days_pct,
same_day_flow_ratio_pct, relay_2d_ratio_pct, max_in_senders_day_pct,
peak_day_share_pct, peer_anomaly_score_pct
```

## Verification and limits

The owned tests cover exact directed metrics, closed-form PageRank, seed
convergence/cutoff, SCC/reciprocity, isolates, empty inputs, all temporal offsets,
censored windows, deterministic ties, zero MAD, exact IDs, bounded pattern
search, amount groups, shuffled supplied input and exact Parquet round-trips.

```sh
python -m pytest tests/test_input.py tests/test_features.py -q
```

Verification on 2026-09-23 used the existing local test image with a read-only
repository mount and networking disabled: Python 3.11.16, pandas 3.0.6, NumPy
2.4.6 and NetworkX 3.6.1. The full repository suite passed **238 tests** in
20.33 seconds. A separate load/validate/feature benchmark produced **2,248 × 86**
in **3.61 seconds**. These are measurements on the development machine, not a
hardware-independent runtime guarantee.

The supplied output has 1,632 nodes with a valid complete-window relay ratio,
76 nodes with repeated-route observations, 169 with reciprocal date observations,
and 20 with qualifying similar-amount groups. Five repeated-route searches hit
their bounds; their flags mark the counts as lower bounds. No return search was
truncated. These counts are diagnostics, not hardcoded production expectations.

The implementation uses local pandas/numpy/NetworkX only, no external services.
The supplied graph is small enough for exact weighted betweenness. At million-node
scale, centrality and per-node date/pattern work need a different batch strategy;
the fixed pattern budgets do not make every other operation constant-time.
The latest observed date is the only end-date information available to this API;
it cannot infer coverage beyond that date. All findings remain sampled activity
descriptors for a decision engine, not role assignments or AML conclusions.
