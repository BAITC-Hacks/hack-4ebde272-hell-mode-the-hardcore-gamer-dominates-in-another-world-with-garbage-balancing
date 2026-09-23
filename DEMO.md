# Five-minute Money Graph demo

**Decision to demonstrate:** whom should an AML analyst review first, what observed
pattern supports that choice, and which missing data should be requested next?
The examples below come from the supplied July 2026 data and committed outputs;
they are presentation cues, never inputs to the scoring algorithm.

## Before the clock starts

1. Follow [README](README.md) to install or build the pinned Python 3.11 runtime.
   Dependency installation is preparation, not the measured analytical run.
2. Open the viewer and a terminal in the repository root. Keep the three supplied
   Parquet files ready for the sidebar upload controls. Keep the local core
   independent of an AI key. Prepare downloads in a local folder.
3. Check the source/run shown by the viewer. Regenerating different data can change
   the roles, clusters and numerical examples below; refresh the cue cards first.
4. Assign one driver, one narrator and one person to watch time and handle the jury's
   arbitrary-gid challenge. Rehearse aloud; automated browser timing does not verify
   a human five-minute presentation.

## Timed live sequence

| Time | Driver action | Narrator's point |
|---|---|---|
| 0:00–0:35 | Expand **Upload case dataset**. Supply **Nodes Parquet**, **Edges Parquet** and **Transactions Parquet**, then click **Validate and run uploaded case** **live**. Show the successful run and visible run ID. | “Three source files become roles for every supplied account, communities and an explained priority list. This run uses observed July intrabank transfers of at least 5,000 KZT.” |
| 0:35–1:25 | In **Investigation queue**, filter consolidators and open **100000003115284100**. Show role evidence, priority contributions and its one-hop graph. | “Eight payers and reachability from nine seeds support collection. Its review priority is 0.974; role strength, seed convergence and path centrality explain that position.” |
| 1:25–2:10 | Search **100000000331309100**. Open its two-hop graph and switch between role and cluster colors. | “This account sends to 99 observed recipients through 126 transfers. Its 23 million KZT outgoing volume is incomplete account context, not proof of unexplained money.” |
| 2:10–2:50 | Open **100000003037476100** and point to the hop-4 warning and unavailable relay/retention values. | “Three payers and two upstream seeds support collection. Zero outgoing edges here cannot establish a final recipient: outgoing activity was not sampled.” |
| 2:50–3:20 | Show **Cluster review** for cluster 14, then the baseline and top-20 **Resilience** rows. | “The community summarizes 54 accounts, one seed and 6.245 million KZT of internal transfers. Removing the priority top 20 reduces the largest observed component from 1,877 to 1,342: structural concentration, not a prediction of criminal-network collapse.” |
| 3:20–4:20 | Let the jury name three supplied gids. Follow the four-step challenge below, about 20 seconds each. | Explain the displayed decision with actual metrics, including an observation limit when relevant. The jury chooses the identifiers. |
| 4:20–5:00 | Return to **Investigation queue**, clear filters, select the three discussed gids in **Review shortlist**, and click **Download review shortlist**. Show its `why`, `limitations` and `next_request`. Click **Download submission bundle** in the sidebar. | “The analyst now has a selected review list with specific follow-up requests, plus the three fixed-schema jury exports. These are reproducible investigation hypotheses, not calibrated probabilities or findings of guilt.” |

The upload action validates and runs the same production pipeline in a private
temporary workspace. It does not overwrite supplied `data/` or the configured
`out/`. Download the handoff before resetting that uploaded case; **Use configured
dataset** restores the original read-only source/export selection. Shortlist
selection resets when the run changes, so records from different cases do not mix.

For a command-line live demonstration instead, restore the configured dataset
and run this command from the prepared local Python 3.11 environment:

```bash
python main.py --data data --out out --submission submission
```

For the Docker command-line path, run `docker compose run --rm analytics`; the
viewer reads the regenerated host `out/` after **Reload data**. Packaging remains
the explicit command above. Never present pre-existing outputs as a fresh live run.
Show the elapsed time that actually appears, rather than promising a historical
benchmark. The requirement is source Parquets to exports in under 300 seconds.

## Two distinct handoffs

- **Review shortlist CSV:** analyst-selected exact gids with role, scores, evidence,
  priority `why`, observation `limitations`, `next_request` and `run_id`. It records
  a review decision, not a finding or an automatically sent law-enforcement request.
- **Submission bundle ZIP:** the three unchanged fixed-schema CSVs plus
  `release_metadata.json` for that analytical run. Selecting a shortlist does not alter
  the official ranking, roles or cluster exports.

Open the downloaded review file during the last segment to show that the boundary
case asks for beyond-hop-4 outgoing history. Keep the exported run ID with the
review record so the recommendation can be traced to its source run.

## Three case cue cards

Use these values to check that the correct node is open. In the timed presentation,
explain the pattern and the largest priority contributors; open the full weighted
formula only if the jury asks how the score was calculated.

| Observed field | Collection candidate | Distribution candidate | Boundary collection candidate |
|---|---:|---:|---:|
| Exact gid | 100000003115284100 | 100000000331309100 | 100000003037476100 |
| Role | consolidator | distributor | consolidator |
| Role strength | 0.970421 | 0.994747 | 0.852461 |
| Review priority | 0.974279 | 0.974322 | 0.658396 |
| Cluster / depth | 14 / 1 | 49 / 2 | 8 / 4 |
| Incoming / outgoing counterparties | 8 / 2 | 5 / 99 | 3 / 0 |
| Incoming / outgoing KZT | 2,160,500 / 517,000 | 984,635 / 23,001,375 | 555,000 / 0 |
| Incoming / outgoing transactions | 15 / 4 | 8 / 126 | 3 / 0 |
| Distinct upstream seeds within four hops | 9 | 4 | 2 |
| Two-day relay | 3 / 5 eligible dates = 0.600 | 4 / 4 eligible dates = 1.000 | Unavailable: outgoing boundary |

### 1. Collection candidate

Exact evidence: `Received 2,160,500 KZT from 8 senders; a possible collection point. Reachable from 9 initial case accounts.`

The structural gate requires at least two incoming counterparties and either at
least two upstream seeds or incoming-degree percentile at least 0.80. The observed
8 and 9 satisfy it; the highest eligible weighted role score is 0.970421, above
0.55. The card exposes every component, weight and competing candidate.

The three largest priority contributions are role strength **+0.242605**, seed
reach **+0.199689**, and path centrality **+0.141793**. The observed retained share
is 0.760704; it describes the sample, not a verified balance. Cluster 14's exact
internal turnover is **6,244,622 KZT**. Its hypothesis is: “Possible collection
group: its 54 accounts have 91 incoming and 71 outgoing payment links. Internal
transfers total 6,244,622 KZT; full account balances remain unknown.” Request a longer history and timestamped adjacent flows to assess
whether the collection pattern persists.

### 2. Distribution candidate

Exact evidence: `Sent 23,001,375 KZT to 99 recipients in 126 transfers, suggesting a distribution role.`

The distribution gate requires at least two outgoing counterparties; 99 qualifies,
and weighted strength 0.994747 clears 0.55. Of its observed outgoing relationships,
25 cross community boundaries. The largest priority contributions are role strength
**+0.248687**, seed reach **+0.184920**, and path centrality **+0.149666**.

Outgoing KZT exceeds observed incoming KZT. Request opening balance, a longer
period and timestamped adjacent transfers before interpreting the source of funds.
Relay 1 means outgoing activity overlapped four eligible incoming-date windows;
it does not establish intraday order or forwarding of the same money.

### 3. Boundary collection candidate

Exact evidence: `Received 555,000 KZT from 3 senders; a possible collection point. Reachable from 2 initial case accounts. Outgoing transfers beyond hop 4 are unobserved.`

Three incoming counterparties and two upstream seeds satisfy the collection gate.
Retained share is unavailable, so the score uses available weight **0.90**; the
missing 0.10 retention component is excluded. Relay is unavailable with reason
`boundary_outbound_incomplete`, even though one inbound date has a complete calendar
window and another is month-end censored. Its temporal priority contribution is
zero because the signal is unavailable, not because non-relay was observed.

The largest priority contributions are role strength **+0.213115**, seed reach
**+0.128336**, and observed turnover **+0.087656**. Request outgoing transfers
beyond hop 4 and at least two days beyond the observed period.

## Jury challenge: any three supplied gids within one minute

Use **Node card**, without running the model or recomputing the graph. Allow roughly
20 seconds per gid:

1. **Decision:** read role and strength, then the winning structural gate under
   **Role evidence**. For peripheral nodes, say whether no gate passed or the
   strongest eligible structure fell below 0.55.
2. **Evidence:** state the two most useful actual counts/amounts and the gate
   threshold they satisfy. **Evaluated role candidates** explains why another
   plausible role did not win; the highest eligible score wins with documented ties.
3. **Priority:** name the leading contribution in the exported explanation.
4. **Limit/action:** identify a seed, hop-4 or date-window limitation if present,
   then say what the analyst should request next. Open the graph if asked for links.

Do not infer a label from the gid or use the prepared examples as a substitute for
this challenge. Exact large gids are identifiers, not numbers to round. The linked
[decision rules](documentation/decision-rules.md) and
[feature contract](documentation/feature-contract.md) are the formula reference.

## Short backup demonstrations

- **Isolated known node:** `100000000456947100` is a seed at depth 0, with zero
  observed in/out relationships and KZT, cluster 30, peripheral strength 0.
  Its graph contains one node. Priority **0.164754** includes relative tied ranks
  and the seed's reachability from itself; it does not establish suspicious activity.
  Request completeness confirmation and longer history. Search `-99` to show the
  distinct unknown-gid state. All 19 supplied isolates remain represented.
- **Temporal/originality evidence:** open the existing node-card daily activity,
  peer-anomaly and bounded route/amount evidence. State observed dates and support
  counts from the selected record; do not imply exact fund tracing, exhaustive
  route enumeration or visibility of transfers below 5,000 KZT.
- **Optional AI:** without a key, show the local workflow and disabled AI state.
  With configured access, ask for a candidate's seed reach and show the validated
  tool citation and **Open gid** navigation. A live API response is not guaranteed
  by mocked/local tests; never make it a dependency of the five-minute demo.

## Evidence and acceptance record

The committed package has **2,248 node rows, 91 clusters and 50 ranked candidates**.
All six role labels occur. All 19 isolated seeds have peripheral strength zero;
no hop-4 node is assigned terminal. The top-list `why` explains priority separately
from the role's short `evidence`. For the collection candidate it is:

> It matches the collection review rule, with observed payments from 8 accounts. It is reachable from 9 starting case accounts within four directed hops and lies on shortest directed routes between other accounts in the observed graph.

The automated browser rehearsal exercises connected, boundary and isolated gids,
role evidence, graph canvases, page navigation and regeneration/reload with external
networking disabled. Its one-minute check measures interaction, not human reasoning
or spoken delivery. See [the release record](submission/RELEASE.md) and the latest
verification report for measured commands, environment and remaining limits.
