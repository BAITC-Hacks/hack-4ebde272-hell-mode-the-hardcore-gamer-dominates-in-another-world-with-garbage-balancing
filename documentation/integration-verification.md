# Integrated technical-brief verification — 2026-09-23

## Result and revisions

The integrated software passes the checks below against the supplied HackAlem
technical brief. No remaining mandatory software blocker was identified. A human
five-minute presentation and live optional LLM calls have not been verified by
this run; automated UI checks do not substitute for the team's spoken demo.

The fetched `origin/testing` and `origin/codex/brief-ui` both pointed to
`fca96990b897dfa28e807ce369016dc753f9d946`. That history already contained all
three members' work:

- Data/features: `bf4859c6a93e3b7f66915dacf3126b75a0028396`, merged by `ca109c6`.
- Decisions/exports: `bf5776e`, merged by `d1ca5ab`.
- UI/integration: `e075b64`, followed by release record/artifacts in `fca9699`.

Local `testing` was fast-forwarded to that shared integrated commit. Both the
data and UI branch tips are ancestors of the resulting branch. No duplicate
merge or history rewrite was needed; the initial working tree was clean.

Verification found and fixed one contract mismatch in commit
`59ffba3ed27eb65b1b49eb934639762c0426aa1e`: decision scoring and explanations
previously checked generic ratio flag names, missing the feature producer's
`relay_2d_valid`, `same_day_flow_valid` and `flow_share_valid`. They now honor
these flags and the compatible legacy aliases. Every present validity flag must
be true; missing or false values suppress the ratio. Seed/depth-4 protections
remain in place. Regression tests cover scoring, priority and explanations.
The supplied data already encoded unavailable ratios as NaN, so this correction
does not change its three submission CSVs.

## Mandatory brief coverage

| Requirement | Verified result |
|---|---|
| One command, raw Parquet to three CSVs, under 300 seconds | Fresh Docker analytics completed successfully in **4.69 seconds** including publication; recorded computation/validation time **4.681239 seconds**. Installation/image build time is separate. |
| Role, score, cluster and evidence for every supplied node | **2,248 unique gids**; all required fields populated; allowed roles and finite scores in [0,1]; longest evidence **142 characters**, below 200. |
| Exact submission schemas | `nodes_roles.csv`: 6 columns; `clusters.csv`: 6; `top_nodes.csv`: 5. Strict validation passed, including optimized Python with assertions disabled. |
| Formal, explainable role criteria | Versioned rules document gates, weights, ties, missing values and fallback; node cards expose the actual winning rule and observed metrics. Three selected arbitrary-gid cards loaded with evidence in **0.89 seconds**. Spoken explanation quality still requires rehearsal. |
| Clustering and summaries | **91 clusters**, covering every gid. Counts, seeds, internal turnover, member references and top gids reconcile with source data and node outputs. |
| Explained ranking of at least 20 nodes | **50 rows**, correctly ordered and cross-file consistent. Priority reasons use ranking contributions and differ from role evidence. |
| Directed visualization and arbitrary-gid lookup | Full graph contains exactly **2,248 nodes and 3,119 edges**, including **19 isolates**; exact string identifiers in browser payloads. Connected, boundary, isolated, unknown and large gids are covered by tests. |
| Local operation without paid services | Entire suite ran with Docker networking disabled. Core graph HTML contains zero external script/link references; browser checks observed no external asset requests. Core image has no OpenAI SDK requirement. |
| Reproducibility and exact identifiers | Equivalent row permutations preserve features and integrated decisions. Identical regeneration preserves artifact hashes. Int64 boundary/large identifiers survive validation, exports, joins and browser/AI serialization. |
| Source, instructions, diagram and final files | Repository includes setup, pinned Python 3.11 dependencies, complete criteria, limitations, 1M-node scaling discussion, architecture diagram, tracked CSVs and a release manifest. `DEMO.md` contains actual examples. |
| Privacy and cautious interpretation | Reviewed production paths calculate from supplied pseudonymous data, with no hardcoded answer lists or invented personal attributes. Scores and roles are hypotheses, not calibrated probabilities or claims of guilt. |

All required CSV values **and bytes** from this fresh run match the tracked
`submission/` package. Its historical release manifest remains associated with
the code/environment that produced that package. The new run's independent
provenance is in ignored `out/run_metadata.json`; the old manifest was not
rewritten to claim it generated the new run.

## Audit closure and optional evidence

The earlier [technical-brief audit](technical-brief-audit.md) is a historical
completion plan. Its implementation findings were checked again after merging:

- **F01–F03:** strict CSVs are separate from the **2,248-row, 180-column** rich
  Parquet artifact; rules and independent role/priority reasons are available.
- **F04–F05:** isolates and cluster views render; role/cluster legends and
  directed edges work offline, including an executing browser canvas.
- **F06:** setup, reproducible outputs, dependency pins, privacy/limitations,
  actual demo examples and architecture are present. Human demo delivery is
  the remaining presentation check.
- **Q01–Q03:** identifier/depth/input validation, independent raw-edge export
  reconciliation and canonical graph/clustering construction are covered.
- **Q04–Q07:** seed and hop-4 ratio invalidity, incomplete two-day windows,
  fingerprinted loading, shared percentile semantics and exact-value fixtures
  are covered. The newly fixed consumer flag mismatch has regression coverage.

Additional implemented evidence includes date-level activity/bursts and sender
counts, bounded repeated routes, structural return paths and observed dated
return sequences, observed similar amounts, individual depth-peer deviations,
removal-of-top-N scenarios, node cards and suggested missing-data requests.
The assistant's tools, claim validation and node links are tested locally/mocked.
Every one of the 2,248 rich node records also serialized with exact gids and
strict JSON (`allow_nan=False`).

The dataset checks retain the actual population: **81 seeds**, **July 2026**,
minimum observed transaction **5,000 KZT**, **19 isolated seeds**, **12 incoming-only
seeds**, and **16 components with edges plus 19 singleton components**. There
are **354** nodes with outflow greater than positive inflow and another **23**
with positive outflow but zero observed inflow. Nodes were not removed to force
agreement with ambiguous brief wording. Final role counts are terminal 1,100;
peripheral 602; distributor 303; consolidator 108; transit 85; coordinator 50.

## Executed verification

Environment: CPython **3.11.16**, Linux aarch64, Docker Desktop. Dependencies
include pandas **3.0.6**, NumPy **2.4.6**, NetworkX **3.6.1**, PyArrow **25.0.1**,
Streamlit **1.64.0**, pytest **9.1.1**, Playwright **1.55.0**, and Chromium
**153.0.8010.52**.

```bash
docker compose --profile test build analytics tests
docker run --rm --network none \
  --mount type=bind,source="$PWD",target=/app,readonly \
  money-graph:test python -m pytest -q -p no:cacheprovider --durations=5 -rP
MONEY_GRAPH_GIT_REVISION=59ffba3 docker compose up -d --no-build
docker compose exec -T app python main.py --data data --out out --validate-only
docker compose exec -T app python -O validate_submission.py --data data --out out
docker compose ps -a
```

- **461 tests passed in 78.24 seconds**, with no failures or skips. Source was
  mounted read-only and runtime networking disabled. The tested implementation
  is the code committed as `59ffba3`; images were rebuilt with that correction.
- Real Chromium visited Overview, Investigation queue, Node card, Network
  explorer, Cluster review, Resilience and AI analyst. It exercised queue/card
  and cluster/graph navigation, a singleton canvas, regeneration/reload and the
  visible new run ID. The original Altair histogram failure has regression
  coverage and Overview rendered without an exception.
- Tests also cover filters/empty states, feature semantics, score ties,
  malformed inputs/exports, incomplete publication, concurrent writers, source
  replacement, stale caches and unsupported AI claims.
- Both fresh-output validators returned success. Three required CSVs were
  independently compared with tracked releases for exact values and bytes.
- Compose analytics exited **0** and the app reported **healthy**, bound to
  **127.0.0.1:8501**. Sources and the viewer's outputs are mounted read-only.
- Fresh run ID: `f70d410f-8878-45d5-a177-d69badc4acb4`.

## Remaining limits

- Rehearse the team's actual five-minute narration and explanation of 2–3 nodes;
  automated navigation timing establishes UI readiness, not human readiness.
- Live optional external LLM calls were not made. Core use and deterministic/
  mocked assistant behavior are verified; API/network/model behavior is not.
- GUI dataset upload and selected-review-list download are optional conveniences
  not implemented. The supported dataset/CSV workflow uses the documented CLI.
- There are no labeled roles to establish accuracy. Observed date overlap does
  not prove intraday order or trace the same money. Seed inflows and depth-4
  outflows are incomplete, end dates censor relay windows, and sub-5,000 KZT
  splitting is invisible. Bounded route searches report their search limits.
- Runtime was measured on the supplied dataset; the million-node approach is
  documented, not implemented or benchmarked. No test suite proves absence of
  every possible defect.
