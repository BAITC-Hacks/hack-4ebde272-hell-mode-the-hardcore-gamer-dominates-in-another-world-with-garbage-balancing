# Five-minute Money Graph demo

These examples were selected from the integrated Python 3.11 output, not injected
into production decisions. The package has 2,248 node rows, 91 clusters and 50
ranked candidates. Regenerate with:

```bash
python main.py --data data --out out --submission submission
python main.py --data data --out out --validate-only
python -m streamlit run app.py --browser.gatherUsageStats=false --browser.serverAddress=localhost
```

## 0:00–0:30 — establish scope

Overview: 2,248 nodes, 3,119 directed edges, 4,840 transactions, 81 seeds and
365,890,012.01 KZT observed turnover (the metric card rounds to whole KZT).
Say “July 2026 intrabank transfers ≥5,000 KZT, outgoing expansion to four hops.”
Scores are review heuristics and hypotheses. No ground truth establishes
classification accuracy or guilt.

## 0:30–1:30 — collection candidate

In the queue select **consolidator**, then open gid **100000003115284100**.

| Card field | Actual exported value |
|---|---:|
| Role strength | 0.970420887893 (card: 0.970) |
| Review priority | 0.974189866614 (card: 0.974) |
| Cluster / depth | 12 / 1 |
| In / out degree | 8 / 2 |
| In / out KZT | 2,160,500 / 517,000 |
| In / out transactions | 15 / 4 |
| Seed reach | 9 |
| PageRank / betweenness percentile | 97.5th / 94.5th |
| Temporal relay | 0.600 |
| Cross-cluster degree / anomaly | 5 / 0.950 |

Exact role evidence: `in_deg=8, in=2,160,500 KZT, seed_reach=9, retention=0.76`.
Explain “eight observed payers and nine upstream seeds meet the consolidator gate;
the weighted role strength is 0.9704.” The displayed retention is a sampled
out/in-derived measure, not a verified account balance.

The three largest priority contributions are role strength **0.242605222**,
seed reach **0.199688612**, and betweenness **0.141792705**. Show the contribution
table, then open the 1-hop network and its role legend.
Cluster 12 contains **59 nodes**, **1 seed**, and **6,772,097 KZT** internal turnover;
the exported hypothesis is “collection-oriented structure.”

## 1:30–2:30 — distribution candidate

Search **100000000331309100** and open its 2-hop neighborhood.

| Card field | Actual exported value |
|---|---:|
| Role / strength | distributor / 0.994747467835 (card: 0.995) |
| Review priority | 0.974321587005 (card: 0.974) |
| Cluster / depth | 48 / 2 |
| In / out degree | 5 / 99 |
| In / out KZT | 984,635 / 23,001,375 |
| In / out transactions | 8 / 126 |
| Seed reach | 4 |
| PageRank / betweenness percentile | 93.9th / 99.8th |
| Temporal relay | 1.000 |
| Cross-cluster degree / anomaly | 29 / 0.944 |

Exact evidence: `out_deg=99, out=23,001,375 KZT, out_tx=126, cross_out=25`.
Show 99 recipients and 126 observed outgoing transfers; 25 outgoing relationships
cross the assigned cluster boundary. Toggle cluster colors/highlighting.
The three largest priority contributions are role strength **0.248686867**,
seed reach **0.184919929**, and betweenness **0.149666370**.

Outflow greatly exceeds observed inflow; missing balances and external activity
prevent a source-of-funds conclusion. Relay=1 describes date overlap, not proof
that received money was forwarded in a particular order. Request opening balance,
a longer period and timestamped adjacent transfers.

## 2:30–3:30 — boundary case

Search **100000003037476100**.

| Card field | Actual exported value |
|---|---:|
| Role / strength | consolidator / 0.852461447212 (card: 0.852) |
| Review priority | 0.658151585039 (card: 0.658) |
| Cluster / depth | 9 / 4 |
| In / out degree | 3 / 0 |
| In / out KZT | 555,000 / 0 |
| In / out transactions | 3 / 0 |
| Seed reach | 2 |
| PageRank / betweenness percentile | 64.5th / 37.0th |
| Cross-cluster degree / anomaly | 1 / 0.439 |

Read the card warning: “Outgoing transfers beyond hop 4 are not present in the
supplied sample. Do not interpret out_deg=0 as confirmed retention.”

Three observed payers and two upstream seeds satisfy the consolidator gate.
The baseline evidence string currently says
`depth=4, in_deg=3, out_deg=0; boundary-censored beyond hop 4`;
it omits that winning-role reasoning, a Member 2 post-audit fix still pending.
Baseline raw `relay_2d_ratio=0` is censored here and must not be interpreted as
observed non-relay; the card explains that limitation.
Largest priority contributions are role strength **0.213115362**, seed reach
**0.128336299**, and turnover **0.087655694**.
Request outgoing payments beyond hop 4 and the following observation window.

## 3:30–4:20 — structure and isolated clients

Open Resilience:

| Removal | Largest weak component | Components | Fraction of baseline largest |
|---|---:|---:|---:|
| Baseline | 1,877 | 35 | 1.000000 |
| Top 1 | 1,782 | 93 | 0.949387 |
| Top 5 | 1,695 | 149 | 0.903037 |
| Top 10 | 1,604 | 205 | 0.854555 |
| Top 20 | 1,342 | 353 | 0.714971 |

Call this structural concentration analysis. It does not simulate operational
intervention or prove a criminal network would collapse.

Optional quick navigation check: **100000000456947100** is a known isolated seed,
depth 0, in/out degrees 0/0, in/out KZT 0/0, cluster 29. Its graph shows a single
seed node. Search **-99** to contrast the explicit unknown-gid state.

## 4:20–5:00 — explanation challenge and close

Ask a teammate to choose three gids from the supplied node file. For each:

1. Search the exact gid in Node card.
2. Read role strength, degrees/KZT, seed reach and the exported evidence.
3. Explain the largest priority contributions and any seed/hop/date limitation.
4. State the next data request.

The automated Chromium rehearsal searches a connected node, a boundary node and
an isolated seed, checks their evidence cards and enforces a combined **<60s**
search limit. It also visits all seven pages, opens graph canvases with external
network access disabled, navigates queue→card and cluster→network, regenerates
outputs and reloads the app. This verifies interaction, not a human narrator's
ability to deliver the five-minute explanation; rehearse the spoken timing above.

Without a key, show the fully working core and the optional AI disabled state.
With approved configuration, ask the assistant for the first candidate's seed
reach: it selects cited exported fields, local validation rejects unsupported
claims, and an **Open gid** button opens only a known node. Live API behavior
has not been verified in this delivery.

The baseline top CSV still repeats role evidence in `why`; use the node card's
separate contribution table for numerical priority reasoning. A completed
Member 2 commit must improve the exported reasons before claiming all audit
requirements closed.
