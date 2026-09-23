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
python -m streamlit run app.py --browser.gatherUsageStats=false
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
  subsequent dates; current feature limitations must accompany interpretation.
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

Exact auxiliary schema for this integrated baseline (69 columns; nullable floats
represent unavailable evidence). These are the actual Parquet columns, grouped
by storage type:

| Type | Columns |
|---|---|
| int64 | `gid, depth, in_deg, out_deg, in_tx, out_tx, total_tx, seed_reach_count, scc_id, scc_size, reciprocal_relationship_count, active_days, max_in_senders_day, cluster_id, cross_cluster_in_deg, cross_cluster_out_deg, cross_cluster_degree` |
| bool | `is_seed, boundary_censored, in_cycle, truncated_by_depth` |
| float64 — observed metrics | `in_kzt, out_kzt, total_kzt, pagerank, betweenness, pass_through, fanin_share, fanout_share, seed_reach_fraction, min_seed_distance, same_day_flow_ratio, relay_2d_ratio, peak_day_share, peer_anomaly_score, retention, balance_score` |
| float64 — percentiles | `in_deg_pct, out_deg_pct, in_kzt_pct, out_kzt_pct, in_tx_pct, out_tx_pct, total_kzt_pct, total_tx_pct, pagerank_pct, betweenness_pct, seed_reach_count_pct, seed_reach_fraction_pct, active_days_pct, same_day_flow_ratio_pct, relay_2d_ratio_pct, max_in_senders_day_pct, peak_day_share_pct, peer_anomaly_score_pct, betweenness_percentile, pagerank_percentile` |
| float64 — decisions/contributions | `role_score, priority_score, priority_role_strength_contribution, priority_seed_reach_count_contribution, priority_betweenness_contribution, priority_pagerank_contribution, priority_total_kzt_contribution, priority_cross_cluster_degree_contribution, priority_temporal_signal_contribution, priority_peer_anomaly_score_contribution` |
| string | `role, evidence` |

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

## Rules and integration status

The shared baseline is `origin/testing@b601280`, containing Member 1 graph
commit `329e432` and Member 2 decision commit `34d1c34`. Their post-audit
fixes and contract documents were not yet published when this integration began.
The UI branch changes consumer/integration code and packages strict schemas;
it does not rewrite teammates' graph or decision engines.

Member-owned contract handoffs are
`documentation/feature-contract.md`, `documentation/data-quality.md`, and
`documentation/decision-rules.md`. Until those commits arrive, the current
authoritative implementations are [graph features](src/graph_features.py),
[temporal features](src/temporal.py), [roles](src/roles.py),
[priority](src/priority.py) and [clustering](src/clustering.py).
The [audit](documentation/technical-brief-audit.md) lists unresolved analytical
findings; the release record states which are still pending.

Current role formulas use P(x) = average-tie percentile over available values
and normalized weighted component means. Missing components do not contribute
to the mean's denominator; missing percentile inputs become zero. A candidate
must pass its gate and score at least 0.55:

| Role | Weighted score components | Gate |
|---|---|---|
| Consolidator | .30 P(in degree), .20 P(in tx), .20 P(in KZT), .20 P(seed reach), .10 retention | in degree ≥2 and (seed reach ≥2 or P(in degree) ≥.80) |
| Distributor | .40 P(out degree), .20 P(out tx), .20 P(out KZT), .10 fanout share, .10 P(cross-cluster out degree) | out degree ≥2 |
| Transit | .25 balance, .20 two-day relay, .20 P(betweenness), .15 P(in degree), .15 P(out degree), .05 same-day ratio | both degrees >0 and (pass-through in [.5,1.5] or relay ≥.5) |
| Coordinator | .25 P(seed reach), .25 P(betweenness), .15 P(PageRank), .15 P(cross-cluster degree), .10 P(in degree), .10 P(out degree) | at least two of seed reach/betweenness/PageRank/cross-cluster percentiles ≥.90 |
| Terminal | .50 no-outflow signal, .20 retention, .15 P(in KZT), .10 P(in tx), .05 P(in degree) | non-seed, depth <4, in degree >0, out degree=0 or pass-through ≤.10 |
| Peripheral | min(max candidate score, .549999) | no eligible winning candidate |

Pass-through = observed out/in when valid; retention = clip(1−pass-through,0,1);
balance = clip(1−abs(1−pass-through),0,1). No-outflow signal is 1 when out degree
is zero, otherwise clip(1−pass-through/.1,0,1) when available. Invalid seed ratios
stay missing. Highest eligible score wins; equal scores break in the order
coordinator, consolidator, distributor, transit, terminal. Peripheral strength
is a fallback, not positive confidence. The old decision percentile singleton
special case differs from the feature helper; this audit finding awaits Member 2.

Priority = .25 role strength + .20 P(seed reach) + .15 P(betweenness)
+ .10 P(PageRank) + .10 P(total KZT) + .10 P(cross-cluster degree)
+ .05 temporal signal + .05 peer anomaly. Temporal signal is the bounded
maximum of relay and same-day ratios (missing becomes zero). Each weighted term
is exported and displayed. This score ranks review effort separately from role.

Louvain uses a sum-KZT undirected projection, resolution 1, seed 42. All other
flow metrics retain directed edges. Fixed randomness alone does not guarantee
row-permutation invariance in the baseline; the graph/cluster canonicalization
fix awaits the members' commits. Baseline temporal features include active dates,
same-day outflow overlap, D/D+1/D+2 outbound-date overlap, max same-day distinct
senders and peak-day activity share. Date overlap is not money tracing.

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
DEMO.md                       numerical five-minute walkthrough
```
