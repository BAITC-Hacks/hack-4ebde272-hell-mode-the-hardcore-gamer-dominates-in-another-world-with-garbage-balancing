# Supplied dataset profile and observation limits

Profile recomputed directly from the three supplied Parquet files on
2026-09-23. These counts describe this release of the data, not hardcoded
acceptance criteria for reusable feature functions. `validate_inputs` and
`build_features` also support smaller synthetic and other valid datasets.

## File identity and storage

| File | Rows | SHA-256 |
|---|---:|---|
| `data/nodes.parquet` | 2,248 | `d2a45b0df6e9352832d5fb09839d10b9e23f898156c3bab263b051b31cc0296d` |
| `data/edges.parquet` | 3,119 | `4e71dde5cd3115bcb26e91202665532ee6581cf8233a9fc8059ea59fb7358a38` |
| `data/transactions.parquet` | 4,840 | `c30c5317b5439591dde86f2058dc47a3d19b2900c055ded994fe547f6fb7e7da` |

Identifiers are stored as signed `int64` in all three files. Node depths are
`int64`, edge depths are `int8`, and seed flags are boolean. Amounts are
`float64`, transaction counts are `int64`, and source transaction dates are
ISO date strings. Validation normalizes dates and preserves identifiers as
exact signed `int64` values.

The minimum and maximum gids are `100000000011452100` and
`100000008782800100`. Both exceed JavaScript's safe integer range. Preserve
integer types in Python/Parquet; serialize gids as decimal strings for browser
payloads. Never convert gids to floating point or reconstruct an identifier
after such a conversion. There are no null input cells, duplicate node gids or
duplicate aggregated `(src, dst)` pairs. All edge and transaction endpoints
belong to the node table.

## Population, dates and amounts

| Measure | Observed value |
|---|---:|
| Seed clients | 81 |
| First / last available date | 2026-07-01 / 2026-07-31 |
| Distinct dates with transactions | 31 |
| Minimum observed transaction | 5,000.00 KZT |
| Maximum observed transaction | 3,000,000.00 KZT |
| Total transaction amount | 365,890,012.01 KZT |
| Total aggregated-edge amount | 365,890,012.01 KZT |
| Sum of edge transaction counts | 4,840 |
| Self-loops | 0 |

Amounts above are displayed to two decimal places. Per-pair transaction sums
and counts reconcile with the edge table under the validator's documented
floating tolerance; the brief rounds total turnover to 365,890,012 KZT.

| Sampling depth | Nodes | Aggregated edges with this depth |
|---|---:|---:|
| 0 | 81 | 0 |
| 1 | 472 | 520 |
| 2 | 462 | 640 |
| 3 | 789 | 1,200 |
| 4 | 444 | 759 |

Edge depth is supplied sampling metadata, not a requirement that source and
destination depths differ by one. General validation accepts integral node
depths 0–4 and edge sampling depths 1–4; it does not impose this dataset's
histogram.

## Components and seed coverage

The directed graph includes every supplied node, including nodes absent from
the edge table. Weak connectivity ignores direction solely for component
membership; directed features retain transfer direction.

- There are **35 weak components**: 16 contain edges and 19 are singleton
  isolates. All 19 isolates are seeds.
- The largest component has 1,877 nodes and 46 seeds. The next has 270 nodes
  and one seed. The other 14 components containing edges have sizes
  17, 13, 6, 6, 6, 6, 5, 5, 4, 4, 3, 3, 2 and 2.
- There are **371 nodes outside the largest component**, including isolates.
  Excluding the 19 isolates gives 352, explaining the brief's figure.
- Another **12 seeds have observed incoming edges but no outgoing edges**.
  Together with the 19 isolated seeds, this gives 31 seeds without observed
  outgoing activity. All 81 seeds remain in node outputs and seed-reach
  denominators. A seed reaches itself at distance zero.
- All **444 depth-4 nodes have zero observed outgoing edges**. They are
  sampling-boundary observations, not evidence of terminal economic behavior.

Do not delete isolates, add artificial edges, force eight total clusters, or
alter data to match a differently defined denominator in the brief. Cluster
counts are algorithm outputs, not input-validation constraints.

## Observed inflow and outflow

`in_kzt` and `out_kzt` are sums of the corresponding directed edge amounts.
Comparisons use observed data only:

| Condition | Nodes |
|---|---:|
| Positive observed inflow and outflow greater than inflow | 354 |
| Zero observed inflow and positive observed outflow | 23 |
| Outflow greater than inflow, combining both cases | 377 |

The first row matches the brief's 354. The 23 nodes in the second row have no
valid inflow denominator. These conditions do not establish negative balances,
unexplained wealth or a complete account-level sources-and-uses statement.
The total of node `in_kzt + out_kzt` double-counts each transfer across its two
endpoints; use the transaction or edge total for graph-wide turnover.

## Interpretation limits

1. **Sampling:** this is a four-hop outgoing sample originating from known
   seeds. Seed incoming data is incomplete. Depth-4 outgoing data is censored.
   Missing observations cannot be interpreted as absence of financial activity.
   Seed-dependent and boundary-dependent ratios remain unavailable according
   to [the feature contract](feature-contract.md).
2. **Bank and amount scope:** the brief limits the sample to intrabank
   transfers of at least 5,000 KZT. Other-bank activity and smaller transfers
   are unobserved. Similar or repeated observed amounts may support a review
   hypothesis, but cannot establish splitting into omitted sub-5,000 transfers.
3. **Time:** only dates are supplied. Same-day overlap cannot establish which
   transfer came first or that outgoing money is the incoming money. July 30
   and July 31 inbound dates lack a complete subsequent D/D+1/D+2 window in
   this extract. Use complete-window denominators and expose omitted dates;
   do not count unobserved August activity as a failed relay.
4. **Completeness:** an available date is not proof that every relevant transfer
   was captured. The data establishes observed patterns within the supplied
   extract, not a full balance, full history or future behavior.
5. **Identity and labels:** gids are pseudonymous. There are no personal
   attributes or ground-truth role labels. Do not infer identities or claim
   measured classification accuracy. Features and downstream roles are
   explainable investigation hypotheses, not findings of guilt.
6. **Use:** the supplied brief limits the dataset to the hackathon. The local
   feature layer does not contact external services or enrich client identities.

## Reproduce the profile

Run this read-only command from the repository root after installing the
project dependencies. It uses no cached exports and writes no input files.
Generic structural/reconciliation validation runs via `load_inputs`; the
following profile calculations deliberately remain separate from validation.

```sh
python - <<'PY'
from pathlib import Path
import hashlib
import networkx as nx
from src.loader import load_inputs

nodes, edges, transactions = load_inputs("data")
graph = nx.DiGraph()
graph.add_nodes_from(nodes["gid"].tolist())
graph.add_edges_from(edges[["src", "dst"]].itertuples(index=False, name=None))
seeds = set(nodes.loc[nodes["is_seed"], "gid"])
components = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
inflow = edges.groupby("dst")["sum_kzt"].sum().reindex(nodes["gid"], fill_value=0)
outflow = edges.groupby("src")["sum_kzt"].sum().reindex(nodes["gid"], fill_value=0)

print("Rows:", len(nodes), len(edges), len(transactions), "seeds:", len(seeds))
print("Node depths:", nodes["depth"].value_counts().sort_index().to_dict())
print("Edge depths:", edges["depth"].value_counts().sort_index().to_dict())
print("Dates:", transactions["date"].min(), transactions["date"].max(),
      "distinct:", transactions["date"].nunique())
print("Amounts min/max:", transactions["sum_kzt"].min(), transactions["sum_kzt"].max())
print("Turnover:", round(transactions["sum_kzt"].sum(), 2),
      round(edges["sum_kzt"].sum(), 2), "count:", int(edges["n_tx"].sum()))
print("Isolated seeds:", sum(graph.degree(gid) == 0 for gid in seeds))
print("Incoming-only seeds:", sum(graph.in_degree(gid) > 0 and
      graph.out_degree(gid) == 0 for gid in seeds))
print("Weak components (size, seeds):", [(len(c), len(c & seeds)) for c in components])
print("Out > in with positive in:", int(((outflow > inflow) & (inflow > 0)).sum()))
print("Positive out with zero in:", int(((outflow > 0) & (inflow == 0)).sum()))
print("Boundary nodes with outgoing edges:", sum(graph.out_degree(gid) > 0
      for gid in nodes.loc[nodes["depth"] == 4, "gid"]))
for filename in ("nodes.parquet", "edges.parquet", "transactions.parquet"):
    path = Path("data") / filename
    print(filename, hashlib.sha256(path.read_bytes()).hexdigest())
PY
```
