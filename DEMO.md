# Five-minute analyst demo

This runbook deliberately selects from the **integrated analytics output**, not
from a hard-coded gid. That keeps the demonstration real when the role/priority
engine is updated. At the time this UI branch was created, `out/` was not present
and the supplied starter exports empty role/priority fields; do not invent sample
numbers. Run the pipeline first, then use the three real candidates selected below
and paste the displayed figures into the presenter notes.

## Before the demo (30 seconds)

```bash
python main.py --data data --out out
streamlit run app.py
```

In the Streamlit sidebar, keep **Exports directory** as `out` and **Source data
directory** as `data`. On Overview, confirm the real scope: 2,248 nodes, 3,119
edges, 4,840 transactions, 81 seeds, observed turnover and the actual role/cluster
counts. State: “This is a four-hop outgoing sample, so these are investigation
candidates, not conclusions.”

## Candidate selection (run once after export)

Use this read-only command from the repository root to print the three actual
gids and their exported evidence. It deliberately does not recalculate roles or
priorities.

```bash
python -c "import pandas as p; d=p.read_csv('out/nodes_roles.csv'); t=p.read_csv('out/top_nodes.csv'); x=t.merge(d,on='gid',how='left',suffixes=('_top','')); role=lambda z: x[x['role'].astype(str).str.lower().isin(z)].sort_values('priority_score_top' if 'priority_score_top' in x else 'priority_score',ascending=False).head(1); b=d[p.to_numeric(d['depth'],errors='coerce').eq(4)].sort_values('priority_score',ascending=False).head(1); print('COORD/CONSOLIDATOR\n',role({'coordinator','consolidator'}).T.to_string()); print('\nTRANSIT/DISTRIBUTOR\n',role({'transit','distributor'}).T.to_string()); print('\nHOP-4 BOUNDARY\n',b.T.to_string())"
```

Record the printed values in the three boxes below before presenting. This is the
exact evidence the node card will render (plus any extra exported feature columns).

| Case | gid | Role / confidence | Priority score | Numeric evidence / `why` |
|---|---:|---|---:|---|
| A — coordinator or consolidator | _(output)_ | _(output)_ | _(output)_ | _(copy exported `evidence` and `why`)_ |
| B — transit or distributor | _(output)_ | _(output)_ | _(output)_ | _(copy exported `evidence` and `why`)_ |
| C — depth-4 boundary | _(output)_ | _(output)_ | _(output)_ | `depth=4`; copy in/out KZT, degrees and evidence |

## 0:30–1:20 — Queue: case A, a coordinator/consolidator

1. Open **Investigation queue**; filter Role to the selected role from case A.
2. Point to its rank, gid, exported priority, cluster, seed reach, turnover and
   `why`; select the row and open the node card.
3. Read the real numbers recorded above: role confidence, in/out degree, in/out
   KZT, in/out transaction counts, PageRank/betweenness percentiles, seed reach
   and the exact `evidence`/priority decomposition.
4. Open its 1-hop graph. Explain only what the arrows show: observed funds flow
   toward/from this gid. Say: “The role is a candidate for review supported by
   these supplied figures, not an allegation.”

## 1:20–2:20 — Flow: case B, a transit/distributor

1. Return to the queue and filter to the real role from case B; open its card.
2. Contrast the exact incoming/outgoing KZT, transaction counts, pass-through or
   temporal relay feature (if exported), and its `why` with case A.
3. Switch to **Network explorer → selected gid, 2 hops**. Highlight arrows,
   counterparties and any cross-cluster degree exported by the pipeline.
4. Close with the investigation question: “Which adjacent high-value flows and
   transaction dates should we request to corroborate this routing hypothesis?”

## 2:20–3:20 — Boundary: case C at depth 4

1. Search the actual case-C gid in **Node card**.
2. Show the amber boundary warning and read it verbatim: “Outgoing transfers
   beyond hop 4 are not present in the supplied sample. Do not interpret
   out_deg=0 as confirmed retention.”
3. State the actual depth, in/out KZT, transaction counts and evidence recorded
   above. Point out the suggested next request: extended outbound history plus
   counterparties/dates.

## 3:20–4:20 — Cluster and resilience context

1. Open **Cluster review** for case A’s real cluster. Show node/seed count,
   internal turnover, top gids, role mix and the exported hypothesis.
2. If `out/resilience.csv` is present, open **Resilience**. Compare baseline,
   remove-top-1, top-5, top-10 and top-20 on largest component, components and
   fraction remaining.
3. Use the exact framing: this is *structural concentration analysis*, not proof
   that blocking accounts would destroy a criminal network.

## 4:20–5:00 — Safe close and optional AI

Show the optional AI page only if a configured `OPENAI_API_KEY` is approved for
the environment. Ask it to compare cases A and B; it can use only deterministic
lookups and must cite gids/numbers. End with: “The workflow preserves the evidence,
the sample boundary and the next data request—so an analyst can make a defensible
decision about what to review next.”
