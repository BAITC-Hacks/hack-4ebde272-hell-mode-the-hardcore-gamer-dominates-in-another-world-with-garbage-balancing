# Money Graph — investigation workspace

A local, directed-payment investigation workflow for HackAlem: validate supplied
Parquet, calculate features and heuristic decisions, export three strict CSVs,
then open evidence, neighborhoods and clusters in Streamlit.

## Run locally (Python 3.11)

Run these commands from the repository root containing `main.py`.

```bash
python -m venv .venv
```

Activate on Windows PowerShell: `.\.venv\Scripts\Activate.ps1`.
On POSIX: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
python main.py --data data --out out
python -m streamlit run app.py --browser.gatherUsageStats=false --browser.serverAddress=localhost
```

The first pipeline command generates and validates the complete analytical run.
`starter.py` is the organizer's baseline, not the production entry point.
Dependency versions are pinned in `requirements.txt` and `constraints.txt`.
No API key or paid service is needed for the pipeline or any core investigation
page. The OpenAI SDK is separately optional: `pip install -r requirements-ai.txt`.

## Docker: build, run and regenerate

```bash
docker compose up --build -d
docker compose ps -a
docker compose logs analytics
```

Open [localhost:8501](http://localhost:8501). Analytics must finish successfully
before the app starts; the app has a health check. Source `./data` is mounted
read-only. Generated files are written to the explicit host directory `./out`;
the viewer mounts it read-only. The Python 3.11 base image is fixed by digest.

Regenerate with `docker compose run --rm analytics`. The viewer fingerprints
files on rerun; use **Reload data** to load the newly published run immediately.
During publication, incomplete or mismatched outputs are rejected. Stop services
with `docker compose down`; host output files remain available.

The optional AI build is `docker compose build --build-arg INSTALL_AI=true`;
set `OPENAI_API_KEY` locally and recreate the app. Never commit secrets.
Git and Docker exclusions both cover Streamlit secrets, dotenv files, credential
directories and key/certificate files.

## Tests and final delivery

```bash
docker compose --profile test run --build --rm tests
python main.py --data data --out out --submission submission
```

The test service has **no external network**. Its Chromium browser rehearsal
visits all seven pages, loads the graph canvas, exercises navigation and reload,
and fails on attempted external asset requests. Unit/AppTest tests cover exact
identifiers, isolated/boundary/unknown gids, the histogram regression, filters,
strict artifacts and source/output provenance. Local unit tests use
`pip install -r requirements-dev.txt` then `python -m pytest -q`; install
Chromium to run the real-browser check outside Docker.

`submission/` contains the three required CSVs and a release manifest.
`out/` stays disposable and ignored. The run metadata records SHA-256 input
and output hashes, timestamps/runtime, schema and implementation versions,
Python/platform/package versions and available Git revision. In Docker without
a Git checkout, supply `MONEY_GRAPH_GIT_REVISION` if revision attribution is
required; implementation source hashes are recorded independently.

See [DEMO.md](DEMO.md) for exact examples and the release verification record
under `submission/` for tested commands, runtime and integration limitations.

## Investigation workflow

Start at **Overview** to confirm the observed population and sampling limits.
The **Investigation queue** covers every supplied gid and filters role, cluster,
depth, seed status and priority. Open a **Node card** to inspect role evidence,
score contributions, flow metrics and the next data request. Search accepts any
exact supplied gid, including isolated seeds.

**Network explorer** defaults to a small neighborhood. Arrows show payment
direction, edge width scales logarithmically with KZT, and hover carries amount
and transaction count. Seeds have a star and border; role and cluster coloring
have legends. Isolated accounts/singleton clusters remain visible. Full view is
explicit opt-in and includes all supplied nodes.

**Cluster review** shows membership, seed count, internal turnover and hypothesis.
**Resilience** compares baseline and removal of top 1/5/10/20 candidates using
weak connectivity. This describes structural concentration and sensitivity;
it cannot predict the real-world effect of blocking accounts.

## Scope and observation limits

The supplied sample is **July 1–31, 2026, intrabank transfers at least 5,000 KZT**,
expanded along outgoing transfers from 81 seeds for four hops. It contains
2,248 nodes, 3,119 directed edges, 4,840 transactions and observed edge turnover
365,890,012.01 KZT. There are 19 isolated seeds: zero observed edges is meaningful
as a sampling state, not proof of inactivity outside the sample.

- Incoming seed activity is incomplete; balances and inbound-dependent seed
  ratios are not complete account behavior.
- Hop-4 outbound activity is outside the observation boundary. Zero observed
  out-degree cannot establish retention. Boundary nodes must not be called
  confirmed terminal recipients.
- Transactions have dates, not intraday timestamps. Same-day overlap does not
  establish ordering or tracing of the same money. July-end relay windows lack
  subsequent dates: July 30–31 incoming dates are excluded from the two-day
  relay denominator, and unavailable seed/boundary ratios remain missing.
- SCC/cycle membership is structural reachability, not a dated return of funds.
  Missing external-bank counterparties, earlier/later activity and opening
  balances limit all conclusions.
- Sub-5,000 KZT splitting cannot be detected with these inputs.
- There is no ground truth and no personal-attribute enrichment. Role strength,
  review priority and anomaly scores are heuristics, not calibrated probabilities
  or evidence of guilt. Clusters and roles are investigation hypotheses.
- The supplied dataset is for **hackathon-only use**. Pseudonymous gids and
  financial connections remain sensitive; keep access and copies within that
  scope.

## Input and output contracts

Inputs:

| Artifact | Exact supplied fields |
|---|---|
| `nodes.parquet` | `gid, depth, is_seed` |
| `edges.parquet` | `src, dst, sum_kzt, n_tx, depth` |
| `transactions.parquet` | `src, dst, date, sum_kzt` |

Gids are exact signed-int64 identifiers in data and text in browser payloads.
Edges aggregate individual transactions per ordered payer/payee pair. Displayed
node turnover is incoming + outgoing activity; summing it over nodes would count
an internal payment twice. Overview uses edge turnover counted once.

Required submission headers (no additional columns):

```text
nodes_roles.csv: gid,role,role_score,cluster_id,priority_score,evidence
clusters.csv: cluster_id,n_nodes,n_seed,sum_kzt_internal,top_gids,hypothesis
top_nodes.csv: rank,gid,role,priority_score,why
```

`nodes_roles.csv` has one row per supplied gid; scores are finite in [0,1],
evidence is nonempty with numeric support and at most 200 characters.
`top_nodes.csv` has at least 20 rows (the current pipeline selects 50), sorted
by priority. Cluster counts, seeds, turnover and top-gid membership are checked
against source and decisions.

Auxiliary artifacts:

| Artifact | Contract |
|---|---|
| `node_features.parquet` | one exact, unique `gid` per node; rich graph/temporal metrics, availability flags, exported priority contributions and explanations |
| `resilience.csv` | `scenario,removed_top_n,n_weak_components,largest_weak_component_size,fraction_baseline_largest,seeds_in_largest_component` |
| `run_metadata.json` | completed-run provenance; input/output SHA-256 maps and version/environment information |

The viewer joins features **one-to-one** by gid; the six CSV fields remain
authoritative even if feature files carry duplicate decision columns.
`fraction_baseline_largest` is largest weak-component size divided by its
baseline value, not the fraction of all accounts remaining.

The integrated run contains 180 auxiliary columns, listed exactly below. This
is a recorded schema, not a fixed column-count acceptance target: new optional
features may be additive. Nullable floats/dates represent unavailable evidence;
validity flags and the [feature contract](documentation/feature-contract.md)
define when a number can be interpreted. The [decision contract](documentation/decision-rules.md)
defines candidate diagnostics and contribution semantics.

<details>
<summary>Exact node_features.parquet columns and loaded pandas types</summary>

| Type / group | Columns |
|---|---|
| int64 | `gid, depth, in_deg, out_deg, in_tx, out_tx, total_tx, seed_reach_count, scc_id, scc_size, reciprocal_relationship_count, active_days, max_in_senders_day, inbound_active_days, outbound_active_days, relay_2d_eligible_days, relay_2d_matched_days, relay_2d_censored_days, repeated_route_count, repeated_route_max_support_days, temporal_return_count, temporal_return_max_support_days, outgoing_repeated_amount_tx_count, outgoing_similar_amount_tx_count, similar_amount_group_count, peer_group_size, cluster_id, cross_cluster_in_deg, cross_cluster_out_deg, cross_cluster_degree, cross_cluster_count` |
| bool — observations | `is_seed, boundary_censored, pass_through_valid, flow_share_valid, in_cycle, same_day_flow_valid, relay_2d_valid, peak_day_share_valid, repeated_route_truncated, temporal_return_truncated, truncated_by_depth` |
| bool — role gates | `role_consolidator_gate, role_consolidator_eligible, role_distributor_gate, role_distributor_eligible, role_transit_gate, role_transit_eligible, role_coordinator_gate, role_coordinator_eligible, role_terminal_gate, role_terminal_eligible` |
| bool — decision availability | `decision_pass_through_valid, decision_retention_valid, decision_balance_score_valid, decision_fanout_share_valid, decision_relay_2d_ratio_valid, decision_same_day_flow_ratio_valid` |
| bool — priority availability | `priority_role_strength_available, priority_seed_reach_count_available, priority_betweenness_available, priority_pagerank_available, priority_total_kzt_available, priority_cross_cluster_degree_available, priority_temporal_signal_available, priority_peer_anomaly_score_available` |
| float64 — observations | `in_kzt, out_kzt, total_kzt, pagerank, betweenness, pass_through, fanin_share, fanout_share, seed_reach_fraction, min_seed_distance, same_day_flow_ratio, relay_2d_ratio, peak_day_share, peak_activity_kzt, outgoing_repeated_amount_tx_share, outgoing_similar_amount_tx_share, peer_anomaly_in_deg, peer_anomaly_out_deg, peer_anomaly_log1p_in_kzt, peer_anomaly_log1p_out_kzt, peer_anomaly_in_tx, peer_anomaly_out_tx, peer_anomaly_score, retention, balance_score, internal_out_kzt` |
| float64 — feature/UI percentiles | `in_deg_pct, out_deg_pct, in_kzt_pct, out_kzt_pct, in_tx_pct, out_tx_pct, total_kzt_pct, total_tx_pct, pagerank_pct, betweenness_pct, seed_reach_count_pct, seed_reach_fraction_pct, active_days_pct, same_day_flow_ratio_pct, relay_2d_ratio_pct, max_in_senders_day_pct, peak_day_share_pct, peer_anomaly_score_pct, betweenness_percentile, pagerank_percentile` |
| float64 — role diagnostics | `role_score, role_best_candidate_score, role_consolidator_score, role_consolidator_available_weight, role_distributor_score, role_distributor_available_weight, role_transit_score, role_transit_available_weight, role_coordinator_score, role_coordinator_available_weight, role_terminal_score, role_terminal_available_weight` |
| float64 — decision ratios | `decision_pass_through, decision_retention, decision_balance_score, decision_fanout_share, decision_relay_2d_ratio, decision_same_day_flow_ratio` |
| float64 — decision percentiles | `decision_in_deg_percentile, decision_in_tx_percentile, decision_in_kzt_percentile, decision_out_deg_percentile, decision_out_tx_percentile, decision_out_kzt_percentile, decision_seed_reach_count_percentile, decision_betweenness_percentile, decision_pagerank_percentile, decision_cross_cluster_out_deg_percentile, decision_cross_cluster_degree_percentile, decision_total_kzt_percentile` |
| float64 — priority inputs/contributions | `priority_role_strength_value, priority_role_strength_contribution, priority_seed_reach_count_value, priority_seed_reach_count_contribution, priority_betweenness_value, priority_betweenness_contribution, priority_pagerank_value, priority_pagerank_contribution, priority_total_kzt_value, priority_total_kzt_contribution, priority_cross_cluster_degree_value, priority_cross_cluster_degree_contribution, priority_temporal_signal_value, priority_temporal_signal_contribution, priority_peer_anomaly_score_value, priority_peer_anomaly_score_contribution, priority_score` |
| string — feature text/JSON | `pass_through_invalid_reason, same_day_flow_invalid_reason, relay_2d_invalid_reason, repeated_route_evidence, temporal_return_evidence, amount_pattern_evidence` |
| str — decision text | `role, role_rule, role_rule_details, role_best_candidate, role_consolidator_gate_reason, role_distributor_gate_reason, role_transit_gate_reason, role_coordinator_gate_reason, role_terminal_gate_reason, priority_explanation, evidence` |
| datetime64[ms] | `temporal_observation_start, temporal_observation_end, peak_activity_date, max_in_senders_date` |

</details>

Metadata version 1 requires `schema_version=1`, `status="complete"`, a nonempty
`run_id`, `started_at`, `completed_at`, `runtime_seconds`, `command`,
`invocation`, `inputs`, `outputs`, `schema_versions`, `rule_versions`,
`environment`, `git`, and `validation`. Input/output maps use exact filenames
with `sha256`, `size_bytes`, and `rows`. Schema versions include the strict
submission version and feature-schema hash. Rule versions map implementation
paths to SHA-256 hashes. Environment contains Python implementation/version,
platform and package versions. Git records revision, dirty state and attribution
source (unavailable values remain explicit).

Submission `release_metadata.json` additionally records
`package_type="submission"` and `analytics_run_metadata_sha256`; its output map
contains just the three submission CSVs. The rich feature artifact remains in
the reproducible analysis run, so a bare three-CSV package is not a substitute
for the complete viewer input.

## Rules and integrated contracts

This branch integrates Member 1's post-audit feature commit `bf4859c`, followed
by Member 2's decision/export commit `bf5776e`, on shared testing commit
`b601280`. The viewer consumes their completed contracts:

- [Feature contract v2](documentation/feature-contract.md): exact metrics,
  units, observation flags, temporal denominators and bounded pattern evidence.
- [Data-quality profile](documentation/data-quality.md): input hashes, counts,
  reconciliation and the supplied sample's observation limits.
- [Decision contract brief-decisions-v1](documentation/decision-rules.md):
  full formulas, gates, tie rules, explanations and strict export validation.

The [architecture](documentation/architecture.md) shows how these layers join.
The [audit](documentation/technical-brief-audit.md) is the historical starting
assessment; final integration checks and remaining limits are recorded with
the release under `submission/`. Verification counts in the member documents
describe their handoff snapshots; the 180-column schema above and the release
record describe the combined implementation.

All roles use the shared P(x): average tied rank divided by the number of
available values. A singleton receives 1, a constant group of N values receives
(N+1)/(2N), and missing/nonfinite values stay missing. Role scores divide the
weighted sum by the sum of available component weights; an observed zero retains
its weight, while a missing component does not. Missing values cannot pass a
numeric gate. Every structural candidate must pass its gate and score at least
0.55; the thresholds are transparent heuristics, not learned probabilities.

| Role | Weighted score components | Gate |
|---|---|---|
| Consolidator | .30 P(in degree), .20 P(in tx), .20 P(in KZT), .20 P(seed reach), .10 retention | in degree ≥2 and (seed reach ≥2 or P(in degree) ≥.80) |
| Distributor | .40 P(out degree), .20 P(out tx), .20 P(out KZT), .10 fanout share, .10 P(cross-cluster out degree) | out degree ≥2 |
| Transit | .25 balance, .20 two-day relay, .20 P(betweenness), .15 P(in degree), .15 P(out degree), .05 same-day ratio | both degrees >0 and (pass-through in [.5,1.5] or relay ≥.5) |
| Coordinator | .25 P(seed reach), .25 P(betweenness), .15 P(PageRank), .15 P(cross-cluster degree), .10 P(in degree), .10 P(out degree) | at least two of seed reach/betweenness/PageRank/cross-cluster percentiles ≥.90, and at least one observed relationship |
| Terminal | .50 no-outflow signal, .20 retention, .15 P(in KZT), .10 P(in tx), .05 P(in degree) | non-seed, depth <4, uncensored flags, positive incoming degree/KZT, 0 ≤ outgoing KZT ≤.10 × incoming KZT, and (out degree=0 or valid pass-through ≤.10) |
| Peripheral | strongest structurally gated candidate below .55; zero if no structural gate passes | no eligible winning candidate |

Pass-through = observed out/in when valid; retention = clip(1−pass-through,0,1);
balance = clip(1−abs(1−pass-through),0,1). Fanout share is out degree divided by
in degree + out degree when available. No-outflow signal is 1 when both observed
out degree and outgoing KZT are zero and incoming KZT is positive; otherwise it
is clip(1−pass-through/.1,0,1) where the ratio is valid. Seed, hop-4 and explicit
invalidity flags suppress dependent ratios and fallback derivations. Highest
eligible score wins; equal scores break in the order coordinator, consolidator,
distributor, transit, terminal. A zero-activity isolate has peripheral strength
zero. The exported `role_rule_details` and candidate gate diagnostics explain
the actual winning rule and available-weight denominator.

Priority = .25 role strength + .20 P(seed reach) + .15 P(betweenness)
+ .10 P(PageRank) + .10 P(total KZT) + .10 P(cross-cluster degree)
+ .05 temporal signal + .05 peer anomaly. Temporal signal is the bounded maximum
of available valid relay and same-day ratios. Priority keeps fixed weights:
missing terms contribute zero without renormalization, with separate availability
flags. All eight exported contributions sum to priority. `priority_explanation`
and `top_nodes.why` identify the three leading contributions; `evidence` explains
the assigned role independently. Scores rank review effort, not guilt.

Louvain uses an amount-weighted undirected projection, resolution 1, seed 42.
Exact numeric node/edge ordering and canonical `math.fsum` aggregation stabilize
equivalent input permutations; other flow metrics retain directed edges. All
isolates are included, and no cluster count is prescribed. Cluster labels sort
by seed count, internal turnover and minimum gid as documented in the decision
contract. Reproducibility is tied to the pinned dependency versions.

## Temporal and pattern evidence

Two-day relay is the fraction of eligible incoming-active dates D that have any
outgoing activity on D, D+1 or D+2. Eligible dates satisfy D+2 ≤ the latest
observed transaction date; the exported eligible/matched/censored day counts
make that denominator visible. Seeds, boundary nodes and nodes without a full
follow-up denominator have an unavailable ratio, not measured zero. Same-day
flow is outgoing KZT on incoming-active dates divided by all outgoing KZT,
subject to its availability flags. Neither ratio traces particular funds.

The feature contract also specifies sender maxima and their dates, sampled
peak-date KZT/share, six depth-peer anomaly components, repeated A→B→C routes,
reciprocal A→B→A date observations and repeated/similar outgoing amount groups.
Route searches have per-node candidate/probe limits and explicit truncation
flags; evidence stores only the strongest bounded examples with exact string
gids, observed dates and KZT. These optional descriptors do not become new role
or priority inputs merely by being exported. Date patterns do not prove intraday
order or returned funds, and observed amount groups cannot reveal omitted
sub-threshold transfers.

## Optional AI and privacy

The assistant uses deterministic node/top/cluster/counterparty/path/comparison
tools. Returned factual claims and node links are validated against cited tool
results; arbitrary prose is not accepted as verified evidence. Missing nodes,
tool failures and unsupported claims are tested with mocks. Live API responses
remain unverified without an approved API key. Enabling AI sends the question
and bounded tool results to the configured provider; the core viewer stays local.

## Toward one million nodes

Keep raw files partitioned and queryable through DuckDB/Polars or a warehouse.
Precompute ranks, role evidence, communities and bounded adjacency indexes, then
serve paginated queues and capped neighborhoods. Do not render the whole graph.

Exact all-node betweenness must become sampled/approximate (with fixed seeds,
reported approximation error and convergence checks), and centrality/path work
must have explicit iteration/time/work limits. Use sparse PageRank, incremental
or partitioned community detection, bounded motif enumeration, and distributed
seed-reach/frontier computation. Replace per-node scans of all transactions with
vectorized daily aggregation and partitioned window joins for D/D+1/D+2 relay;
carry observation-window flags through those computations. Benchmark CPU/memory,
skewed high-degree nodes and approximation stability before claiming this scale.

## Structure

```text
data/                         supplied, immutable Parquets
main.py, pipeline.py          one-command integration/provenance/packaging
src/                          member-owned features and decisions; optional AI
app.py                        Streamlit investigation workflow
out/                          ignored generated run with auxiliary metrics
submission/                   three tracked strict CSVs and release records
tests/                        graph, decision, UI, AI and offline integration tests
documentation/architecture.md data flow and ownership
documentation/feature-contract.md exact feature/availability semantics
documentation/data-quality.md supplied input profile and limitations
documentation/decision-rules.md formulas, gates and export contract
DEMO.md                       numerical five-minute walkthrough
```
