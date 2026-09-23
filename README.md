# Money Graph — AML investigation workspace

Money Graph turns a sampled, directed graph of intra-bank payments into an
analyst workflow. The pipeline produces explainable candidate roles, clusters and
review priorities; the Streamlit app makes those outputs searchable, auditable and
easy to investigate. A role or priority is a **hypothesis for review**, never a
finding that a client has done anything wrong.

## Problem and solution

The supplied data contains 2,248 observed clients, 3,119 directed payer→payee
relationships, 4,840 transactions and 81 seed clients. Reading this as a static
network loses the operational question: which candidate should an analyst open
first, what numeric evidence supports that choice, and what data should be
requested next?

The solution keeps the graph analytics pipeline separate from the interface:

- The pipeline validates Parquet, builds the directed weighted graph, calculates
  graph/temporal/seed features, clusters it, assigns the agreed role and priority
  formulas, and exports CSVs.
- `app.py` only consumes those CSVs plus the source Parquet for context and
  small network views. It never recalculates or changes role/priority formulas.
- Every node card shows exported evidence, a priority explanation, sampling
  boundaries, and a concrete next data request.

See [the architecture diagram](documentation/architecture.md).

## Installation

Use Python 3.10+ and a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install streamlit pyvis
```

`pyvis` is optional but enables directed interactive network diagrams. The optional
AI panel additionally needs `pip install openai` and `OPENAI_API_KEY`; the core
app and CSV pipeline do not require either.

## Run

The exact analytics pipeline command, from the repository root, is:

```bash
python main.py --data data --out out
```

The current starter snapshot can produce the prescribed CSV skeleton with:

```bash
python starter.py --data data --out out
```

After the analytics team’s `main.py` integration, use the first command. Open the
analyst interface with:

```bash
streamlit run app.py
```

The sidebar accepts other source/export paths, so a separate pipeline workspace
can be inspected without copying data.

## Analyst workflow

1. **Overview** confirms the data scope, role counts, score distribution and depth
   boundary before conclusions are drawn.
2. **Investigation queue** filters a score-sorted list by role, cluster, depth,
   seed status and minimum priority. Opening a selected row leads to its card.
3. **Node card** presents scores, directed flow, evidence, counterparties and
   what to request next. At depth 4 it displays a prominent truncation warning.
4. **Network explorer** defaults to a 1- or 2-hop neighborhood or a cluster;
   it does not default to an unreadable 2,248-node hairball. Arrows show flow,
   colour is exported role, size is priority, and ★ marks a seed.
5. **Cluster review** gives composition, internal turnover, hypothesis and top
   candidate gids. **Resilience** shows supplied structural-concentration results.

## Input schema

`data/edges.parquet` is one aggregated directed relationship per row:

| Field | Meaning |
|---|---|
| `src`, `dst` | payer and payee gid |
| `sum_kzt` | observed turnover on the relationship |
| `n_tx` | number of transactions |
| `depth` | discovery hop (1–4) |

`data/nodes.parquet` provides `gid`, discovery `depth`, and `is_seed`.
`data/transactions.parquet` provides individual `src`, `dst`, `date`, and
`sum_kzt` rows. The UI uses source data only for counts, observed turnover,
counterparties and visualization; export values remain authoritative for results.

## Output schema

The UI expects the analytics-owned outputs below in `out/`.

| File | Required fields | Used for |
|---|---|---|
| `nodes_roles.csv` | `gid`, `role`, `role_score`, `cluster_id`, `priority_score`, `evidence` | cards, filtering, role and priority context |
| `clusters.csv` | `cluster_id`, `n_nodes`, `n_seed`, `sum_kzt_internal`, `top_gids`, `hypothesis` | cluster review |
| `top_nodes.csv` | `rank`, `gid`, `role`, `priority_score`, `why` | investigation queue |
| `resilience.csv` (optional) | `scenario`, `largest_component_size`, `n_components`, `fraction_remaining` | structural concentration page |

Extra exported features are displayed automatically when named `seed_reach`,
`pagerank_percentile`, `betweenness_percentile`, `temporal_relay`,
`cross_cluster_degree`, `anomaly_score`, `priority_decomposition`, or close
aliases. This preserves forward compatibility without inventing values.

## Analytics contract: roles, priorities, clusters and time

The final role and priority formulas are owned by the analytics pipeline. The UI
does **not** duplicate them, so a dashboard release cannot silently diverge from
the scored CSV. Its role contract is the six exported labels:
`consolidator`, `transit`, `distributor`, `terminal`, `coordinator`, and
`peripheral`; `role_score` is the confidence emitted by that role engine.

Likewise, `priority_score` and any `priority_decomposition`/`why` are rendered
verbatim from the priority engine. The role and priority formula documentation
should be versioned alongside the analytics source and included in each export’s
`evidence`; this gives analysts exact numeric reasoning while retaining a single
source of truth.

Louvain clustering is run by the pipeline on an explicitly documented undirected
projection of the directed payment graph. The app only shows its cluster IDs,
members, hypothesis, and internal supplied turnover. Temporal analysis uses
`transactions.parquet` dates in the pipeline; a supplied `temporal_relay` feature
is displayed but never reconstructed by the UI.

## Sampling limitations and safe interpretation

- The graph is an **outgoing** four-hop expansion from 81 seeds. Seed `in_kzt`
  can be incomplete, so a high outflow/inflow ratio for a seed is not by itself a
  behavioral signal.
- At depth 4, downstream outgoing payments are outside the supplied sample. The
  app explicitly warns that `out_deg=0` is not confirmed retention and suggests
  requesting extended outbound history.
- Observed turnover is only turnover inside the supplied graph and period.
- Network position, a cluster, a score, or a pattern establishes a review lead,
  not illicit intent or guilt. Corroborating operational, legal and customer
  context is required.

## Privacy and access

Gids are pseudonymous identifiers, but payment-graph data remains sensitive.
Run the app in an approved environment; enforce least-privilege access, audit
exports, encrypt data at rest/in transit, and do not paste raw records into public
tools. The optional AI assistant sends only the user’s question and deterministic
tool results to the configured OpenAI account; enable it only under your approved
data-processing policy.

## Optional grounded AI assistant

With `OPENAI_API_KEY`, the assistant can call only deterministic functions:
`get_node`, `get_top_nodes`, `get_cluster`, `find_common_descendants`,
`get_counterparties`, `find_paths`, and `compare_nodes`. The system instructions
require source metrics, real gids/numbers, appropriate sampling caveats and
candidate-review language. Without a key the complete core app still works.

## Scaling toward ~1M nodes

Do not ship the full graph to the browser. Precompute exports/features in a batch
engine, store Parquet partitioned by gid/cluster/date, and query with DuckDB,
Polars or a warehouse. Serve paginated queue results and bounded neighborhoods
from an API/cache; build cluster summaries and rank indexes offline. Use a graph
database or adjacency store for path/counterparty queries, background resilience
jobs, authorization-aware row filtering and aggregated/de-identified UI defaults.

## Project structure

```text
data/                       # source Parquet supplied for the case
out/                        # analytics-owned CSV exports (created by pipeline)
main.py                     # analytics pipeline integration (owned by analytics)
starter.py                  # supplied baseline loader/export skeleton
app.py                      # Streamlit analyst UI
src/ai_assistant.py         # optional grounded tool-calling assistant
documentation/architecture.md
DEMO.md                     # five-minute analyst walkthrough
```

## Demo

Follow [DEMO.md](DEMO.md) after producing the final output files. It selects a
real exported high-priority coordinator/consolidator, transit/distributor and a
hop-4 boundary example, then records their exact values directly from the UI.
