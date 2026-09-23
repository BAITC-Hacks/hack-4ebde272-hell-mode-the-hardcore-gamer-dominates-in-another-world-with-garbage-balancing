# Technical brief audit and three-person completion plan

Audit date: 2026-09-23. Branch: `testing`.

**Historical audit:** the fixes below have since been integrated. See the
[integrated verification report](integration-verification.md) for current
requirement coverage, 461 passing tests, the final contract fix and remaining
verification limits.

Scope: the supplied Russian HackAlem technical brief, current source, current
local exports, Docker configuration, documentation, and existing tests. This
report distinguishes required submission fixes from optional scoring features.
It is a completion plan; the application fixes listed here have not been made
as part of this audit. Existing uncommitted integration work is included in the
review and must be included in the eventual handoff.

## 1. Assessment

The integrated application already implements most of the analytical workflow.
It is not yet ready to claim full compliance with the brief. The main gaps are
the fixed export schema, documented decision rules, understandable explanations,
visualization of isolated clients, external graph assets, and final packaging.

Earlier integration verification in this task passed 41 tests under Docker's
Python 3.11 runtime; the pipeline took approximately 2.45 seconds. Those results
support basic functionality and performance, but do not prove compliance with
every requirement. This audit performed focused data/source/HTML checks rather
than rerunning that entire suite.

### Current data and outputs

| Item | Observed |
|---|---|
| Inputs | 2,248 nodes; 3,119 directed edges; 4,840 transactions |
| Seeds and depths | 81 seeds; depth counts 81 / 472 / 462 / 789 / 444 |
| Transaction dates | 2026-07-01 through 2026-07-31 |
| Smallest observed transaction | 5,000 KZT |
| Observed turnover | 365,890,012.01 KZT; the brief rounds to whole KZT |
| Local `out/nodes_roles.csv` | 2,248 rows, **69 columns**, all six role labels represented |
| Local `out/clusters.csv` | 91 clusters, including 8 with more than one seed |
| Local `out/top_nodes.csv` | 50 rows |
| Local `out/resilience.csv` | Baseline plus removal of top 1 / 5 / 10 / 20 |
| Seed ratio protection | Seed pass-through and inbound-dependent temporal ratios are missing |
| Boundary protection | No depth-4 node is classified as `terminal` |

Current roles: 1,100 terminal, 601 peripheral, 303 distributor, 108 consolidator,
85 transit, and 51 coordinator. These are model outputs, not validated labels.
There is no ground truth with which to claim classification accuracy.

### Reconcile the brief before changing the data

- There are **16 components containing edges, plus 19 isolated seed nodes**.
  Including every supplied node correctly gives 35 weak components. Of all
  2,248 nodes, 371 are outside the largest component of 1,877; excluding the
  19 isolates gives the brief's 352. Keep isolates in all node outputs.
- The brief's **354 nodes with outflow greater than inflow** matches the subset
  with positive observed inflow. Another 23 have positive outflow and zero
  observed inflow, giving 377 if that condition is counted without a denominator
  restriction. Describe the denominator explicitly.
- The reference's eight communities refers to communities with multiple seeds,
  not an acceptance requirement that the entire graph have exactly eight clusters.
  The current result has eight such communities. Do not tune results to a
  prescribed cluster count or hardcoded gids.

## 2. Mandatory requirement coverage

| Brief requirement | Status | Evidence and remaining work |
|---|---|---|
| M1: One-command raw data to three CSVs in under five minutes | Implemented; release verification remains | `main.py`, `pipeline.py`, and Compose integrate the stages. Pin the tested environment and reproduce from the final clean checkout. |
| M2: Role, score, cluster and evidence for all 2,248 nodes | Data present; schema and explanation gaps | All rows exist, but `nodes_roles.csv` has 69 rather than the specified six columns. Some evidence does not explain the assigned role. See F01 and F03. |
| M3: Formal rule or threshold for every role; explain three arbitrary gids within a minute | Incomplete | Rules exist in `src/roles.py`, but README describes an ownership contract instead of documenting formulas, thresholds, missing-value behavior and tie resolution. See F02. |
| M4: Cluster membership and populated summaries | Implemented | All nodes assigned; summaries contain the six required fields. Improve consistency validation and reproducibility checks. |
| M5: Explained top list of at least 20 plus directed graph and arbitrary-gid search | Partially complete | 50 ranked nodes and directed graph exist. `why` repeats role evidence, and 19 isolated nodes cannot be drawn. See F03 and F04. |
| Exact three submission schemas | Incomplete | Cluster and top headers match. The node header must be separated from the richer UI feature artifact. |
| Local operation; no required paid service | Core implemented; asset gap | LLM is optional. Generated graph HTML still references two external Bootstrap assets. See F05. |
| Sampling limits, privacy and cautious language | Partially documented | Seed and depth-4 protections exist. Threshold, date window, intrabank scope, score interpretation and date-only limits need clearer documentation. |
| README with setup, rules, outputs, limitations and 1M-node discussion | Partial | Setup and scaling exist; role rules and some limitations are missing, and several sections are stale. |
| Source repository and three final exports | Present locally; handoff incomplete | Exports exist in ignored `out/`, not a tracked or documented submission package; integration files also remain uncommitted. |
| One solution diagram | Implemented | `documentation/architecture.md` provides the required flow diagram. Update if the export contract changes. |
| Five-minute live run and substantive explanation of 2–3 nodes | Prepared outline only | `DEMO.md` has a selection command and placeholders; final examples and a timed rehearsal are still needed. |
| No hardcoded answers or invented personal attributes | No violation identified in reviewed production paths | Preserve calculated results and pseudonymous gids. Test fixtures and choosing real demo examples are not hardcoded production answers. |

## 3. Required fixes before submission

### F01 — Export the exact fixed schemas

**Owner: Member 2; consumer changes: Member 3.**

Evidence: `src/exports.py:7` adds three fields to the required node schema, and
`:49` appends every remaining feature. The current file has 69 columns. Both
`validate_submission.py:17` and `tests/test_outputs.py:21` allow extra fields.

Keep `nodes_roles.csv` exactly:

```text
gid,role,role_score,cluster_id,priority_score,evidence
```

Keep `clusters.csv` exactly:

```text
cluster_id,n_nodes,n_seed,sum_kzt_internal,top_gids,hypothesis
```

Keep `top_nodes.csv` exactly:

```text
rank,gid,role,priority_score,why
```

Export the detailed node metrics and score contributions separately, for example
as `node_features.parquet`, indexed or keyed by exact `gid`. Update the UI to join
that artifact one-to-one. Simply removing the extra columns would lose the node
card's analytical detail. Preserve int64 gids in data and strings in browser
payloads; never pass large identifiers through floating-point conversions.

**Acceptance:** exact headers and types, 2,248 unique input gids, populated
required fields, valid roles, scores in [0,1], evidence at most 200 characters;
all current node-card metrics remain available through the auxiliary artifact.

### F02 — Publish the actual role and priority rules

**Owner: Member 2; README integration: Member 3.**

Evidence: `README.md:153` says the rules should be documented, whereas
`src/roles.py:59` contains weights and `:73` contains gates. Current rules include
a 0.55 candidate-score threshold, coordinator gates at the 90th percentile,
transit pass-through in [0.5,1.5] or relay at least 0.5, and explicit terminal
exclusion for seeds and depth 4. These choices need a written rationale.

Document all six roles, component weights, eligibility gates, percentile/tie
semantics, missing-value renormalization, the winning-role tie order, and the
peripheral fallback. Also document `src/priority.py:9` weights and the distinction
between role strength and review priority. The implementation calls role scores
evidence strength, while README and DEMO still call them confidence; explain
that these are heuristic scores, not calibrated probabilities.

**Acceptance:** README links to complete versioned rules; a node lookup exposes
the actual winning rule, metrics and thresholds, allowing three arbitrary gids
to be explained within the brief's one-minute check.

### F03 — Make role evidence and ranking reasons useful to an analyst

**Owner: Member 2.**

Evidence: `src/explanations.py:26` replaces all role evidence for a depth-4 node
without outgoing edges with a boundary statement. The current data has **10
depth-4 consolidators** whose evidence therefore omits why they are consolidators.
Other evidence strings use internal abbreviations such as `seed_reach`,
`retention`, and `100p` without plain-language explanations. In
`src/exports.py:56`, `top_nodes.why` is copied directly from `evidence`; all 50
current top reasons equal role evidence, even though ranking uses other signals.

Generate concise, readable role explanations with supporting values, plus the
relevant limitation. Generate priority explanations separately from the largest
exported score contributions. The node card already has contribution values;
reuse the analytical output rather than implementing another scoring formula.
Explain peripheral/no-observed-activity cases without implying strong positive
evidence: all 19 isolates currently receive the peripheral fallback score
0.549999, so calling that a confidence of membership would be misleading.

**Acceptance:** every role has a substantive reason, including boundary nodes;
`evidence` stays within 200 characters; every top reason explains the ranking
using actual metrics/contributions; no claims of guilt or complete balances.

### F04 — Display any supplied gid, including isolated nodes

**Owner: Member 3.**

Evidence: `app.py:452` returns no graph when edges are empty and `:458` constructs
the node set only from edge endpoints. `network_page` also exits for an empty
edge selection at `:523`. As a result the 19 known isolated seeds, isolated
clusters, and the full set of 2,248 nodes cannot all be represented visually.

Build the displayed node set from the selected node/cluster records, retain the
focus node even when it has zero edges, and distinguish a known isolated gid
from an unknown gid. Preserve the focus node when limiting large neighborhoods
to 500 edges. Add a role legend and a cluster coloring/highlighting mode; today
clusters can be filtered, but colors only represent roles.

**Acceptance:** search a connected node, boundary node, isolated seed, unknown
gid and large identifier; show the correct state and exact identifier for each.
The full view includes 2,248 nodes; singleton cluster view displays its member.

### F05 — Remove external assets from the core viewer

**Owner: Member 3.**

Although `pyvis_html` uses `cdn_resources="in_line"`, the generated HTML includes
Bootstrap CSS and JavaScript references under `https://cdn.jsdelivr.net/`.
The current test checks only that vis-network is not fetched remotely. This is
a runtime network dependency/request left in a product described as local.

Remove unused remote assets or bundle the necessary files locally. Inspect actual
generated HTML and exercise the core UI with external network access disabled.
Installation/build dependency downloads are separate from runtime operation.

**Acceptance:** every core page, including the network canvas, works without
external network access after installation. The optional LLM is the only
intentional runtime exception.

### F06 — Finish the README, final export package and demo

**Owner: Member 3; analytical content from Members 1 and 2.**

- Add POSIX activation alongside the existing Windows virtualenv command; use
  the tested Python 3.11 environment for the submission.
- Remove stale starter-snapshot language. Keep the organizer starter clearly
  separate from the production command so judges do not run skeleton exports.
- Correct the documented resilience columns to match the actual export.
- Document July 2026, intrabank-only transfers, the 5,000 KZT inclusion threshold,
  incomplete observed balances, date-only timing, no labels or personal
  attributes, and the supplied data's hackathon-only use restriction.
- Make the 1M-node discussion specific: exact betweenness and the current
  per-node temporal scans must be replaced by bounded/approximate centrality and
  vectorized or partitioned aggregation; storage and browser pagination alone
  do not address computation.
- Pin or constrain the dependency versions that passed under Python 3.11. Keep
  a release record of versions, input hashes, command and runtime. Broad minimum
  versions currently allow future changes to results or APIs.
- Create a documented final `submission/` package containing the **three strict
  CSVs** after F01–F03. Keep normal intermediate outputs in ignored `out/`.
  Alternatively attach an explicit release bundle; do not rely on a developer's
  ignored folder or inaccessible Docker named volume as the handoff.
- Select 2–3 real nodes from final outputs, fill DEMO's placeholders, include a
  boundary example, and rehearse the five-minute live flow plus the arbitrary-gid
  explanation challenge. Record real metrics, not invented examples.
- Review and include the current modified/untracked implementation and Docker
  files in the final team commit/handoff. Keep credentials out of both Git and
  the Docker context; the present `.dockerignore` does not cover Streamlit
  secrets or all secret paths covered by the expanded `.gitignore`.

**Acceptance:** a teammate starting from the final clean checkout can follow
README, generate and validate the three deliverables in under 300 seconds, open
the viewer, and run the prepared demo. The architecture diagram reflects the
final auxiliary-feature artifact.

## 4. Correctness and integration improvements

These are concrete weaknesses or verification gaps, distinct from missing bonus
features. Complete them before making a broad claim that every function works.

| ID | Finding and evidence | Action / acceptance | Owner |
|---|---|---|---|
| Q01 | `src/schema.py` checks gid nulls/uniqueness, but not integer/int64 types for all identifiers. Edge depth is cast with `int()` later rather than validated as an integral depth. | Reject malformed, fractional, boolean or out-of-range identifiers and invalid edge depths with useful errors; preserve valid int64 values exactly. Keep supplied-dataset profile checks separate from reusable small-fixture validation. | 1 |
| Q02 | `validate_submission.py` allows extra columns, uses removable Python `assert` checks, and checks total cluster population without validating every reference/count/turnover. It does not fully reconcile top roles/scores/gids against node outputs. | Explicit validation errors; exact schemas/types; nonempty text; cluster membership, seed counts, internal turnover and top-gid membership; top rows match the node table and sorted scores. Avoid treating the implementation's 50-row choice as a maximum required by the brief, which only requires at least 20. | 2 |
| Q03 | Fixed Louvain seed does not remove input-order sensitivity. With the same graph's rows shuffled using random state 7, a focused check produced 93 clusters instead of 91, with different partitions. | Canonicalize node/edge insertion and aggregation order; preserve original gid output ordering separately. Check repeat runs and equivalent row permutations, with tolerances for numeric metrics. Pin dependencies for reproducibility across installs. | 1 graph construction; 2 clustering |
| Q04 | All 444 boundary nodes have `relay_2d_ratio=0`, even though their outgoing window is censored. Dates at July's end also lack the full following two days. | Mark unavailable relay evidence or explicitly qualify its observation window/denominator; prevent missing activity from being presented as observed non-relay. Preserve the distinction between date overlap and proof that the same money moved onward. | 1; 2 consumes flags |
| Q05 | `app.py:101` caches data by path strings; replacing exports at the same paths does not itself invalidate the cache. Independent source/export paths are not checked for matching datasets. | Add file fingerprints/run metadata or an explicit reliable reload flow, and reject mismatched source/exports. Do not mix old scores with new edge context. | 3; 2 emits agreed metadata |
| Q06 | There are two percentile helpers. `src/roles.py:14` returns the clipped raw value for one valid input; `src/graph_features.py:35` specifies singleton rank 1. | Adopt one documented percentile policy and test ties, missing values, constants, singleton and zero MAD. This is a reusable-function defect, not evidence that the 2,248-row run failed. | 1 and 2, each in owned files |
| Q07 | Current tests check feature presence/bounds but have limited exact-value coverage for temporal offsets, SCC/reciprocity, seed-cutoff distances and degenerate peer groups. There is no strict full submission-contract check. | Add focused fixtures for D/D+1/D+2 versus D+3, month-end censoring, two-seed convergence, 4-versus-5-hop reach, cycles, isolates, zero MAD, score tie resolution and winning-rule evidence; then an end-to-end acceptance suite. | 1 features; 2 decisions/exports; 3 integration |

The current date-overlap metrics do not establish intraday order. SCC membership
establishes a possible directed return path, not temporal proof that particular
funds returned. These distinctions belong in both documentation and analyst text.

## 5. Optional features: current coverage and remaining work

The brief explicitly says these do not block submission. Complete required fixes
first; these tasks can improve originality and practical value afterward.

| Optional feature | Current state | Work to complete it | Owner |
|---|---|---|---|
| Hop-4 artifact handling | Implemented for pass-through and terminal roles | Preserve protections; address temporal censoring under Q04. | 1 + 2 |
| Temporal relay, bursts, synchronized senders | Date-level relay, peak-day share and sender-count features exist; UI exposes relay only | Add activity charts, readable burst/same-day sender evidence, and date-granularity caveats. | 1 computes; 3 displays |
| Repeated routes A→B→C and return flows | SCC/reciprocal structure exists; no repeated-route detector or dated return-flow evidence | Bounded motif/route aggregation with support counts, dates and amounts; separate structural possibilities from observed timed sequences. Do not enumerate every cycle. | 1 |
| Amount splitting / peer anomalies | Depth-based median/MAD peer score exists; no explicit observed splitting detector or per-signal explanation | Expose the contributing peer deviations; optionally flag repeated/similar observed amounts using documented rules. **Sub-5,000 KZT splitting cannot be detected from this sample.** | 1 features; 2 explanations |
| Removal of top-N | Implemented with baseline and multiple scenarios | Keep it labeled structural concentration analysis; explain components versus communities. | 2, display 3 |
| Natural-language AI with links to nodes | Deterministic tools and optional model flow exist; responses are rendered as Markdown with no implemented node deep-link contract | Add validated node links/navigation and citation checks; handle tool errors. Mocked tests do not establish that every final model claim is supported. Live API behavior remains unverified without configuration. | 3 |
| Autogenerated node card | Implemented | Add winning-rule explanation and better presentation of temporal/cycle evidence. | 2 evidence; 3 display |
| Completeness and next data request | Seed/boundary warnings and generic requests exist | Add threshold, period, missing external counterparties and per-node reasons for requesting further data. Avoid presenting missing observations as zero behavior. | 1 flags; 3 display |
| Import/review-list convenience | CLI/path-based loading works; no upload/recompute or selected-case download flow | If time allows, add an input validation/run flow and analyst-selected review-list export. These improve the scenario; the brief does not require a specific upload widget. | 3 |

## 6. Work split for three members

### Member 1 — Data contracts, features and observation limits

**Own:** `src/schema.py`, `src/loader.py`, `src/graph_features.py`,
`src/temporal.py`, `tests/test_input.py`, `tests/test_features.py`.
Create `documentation/feature-contract.md` and `documentation/data-quality.md`.

1. Write a supplied-data profile with the counts and interpretation distinctions
   above; document every feature, units, denominator and invalidity rules.
2. Finish Q01, the graph portion of Q03, Q04, and the feature portions of
   Q06–Q07. Preserve all 2,248 nodes and exact gids.
3. Hand Member 2 a deterministic gid-keyed feature table plus documented
   availability flags. Hand Member 3 readable observation-limit descriptions.
4. After mandatory work, add bounded repeated-route/return evidence and observed
   amount-pattern signals, with matching fixture tests and clear limitations.

**Done when:** graph/temporal expectations are numerically tested, seed ratios
remain invalid, depth-4 observations do not imply terminal behavior, cutoff and
isolates work, and repeat/permutation checks meet the agreed contract.

### Member 2 — Decisions, explanations, clustering and submission contracts

**Own:** `src/roles.py`, `src/priority.py`, `src/clustering.py`,
`src/explanations.py`, `src/exports.py`, `src/resilience.py`,
`validate_submission.py`, `tests/test_roles.py`, `tests/test_outputs.py`.
Create `documentation/decision-rules.md`.

1. Publish the exact output/auxiliary-artifact contract immediately (F01), so the
   UI can be updated in parallel without guessing column names.
2. Complete F02–F03: documented formulas, threshold rationale, score meaning,
   winning-rule evidence, and independent ranking reasons.
3. Complete Q02 and the decision/clustering portions of Q03 and Q06–Q07. Keep
   deterministic ties, all-node membership and cutoff/seed protections.
4. Emit strict submission CSVs plus the rich feature artifact and agreed run
   metadata. Verify internal turnover, cluster references and contribution sums.
5. Supply final numerical examples and interpretation notes for the demo.

**Done when:** the three files pass a strict validator, every role and top
position is explainable, and a teammate can explain three arbitrary gids using
the written rules and exported evidence.

### Member 3 — Viewer, integration, Docker and delivery

**Own:** `app.py`, `src/ai_assistant.py`, `main.py`, `pipeline.py`, `Dockerfile`,
`compose.yaml`, `.dockerignore`, `.gitignore`, dependency files,
`tests/test_app.py`, `tests/test_ai_assistant.py`, `README.md`, `DEMO.md`,
`documentation/architecture.md`, and the final `submission/` package.

1. Update the loader for Member 2's auxiliary feature artifact; complete F04–F05
   and Q05. Keep score computation in the decision engine.
2. Add cluster highlighting/legend, isolated-node rendering, exact-gid
   navigation and useful distinction between unavailable data and zero.
3. Complete F06: pinned environment, secret exclusions, correct README, final
   output package, architecture update, filled demo and clean-checkout run.
4. Run the final end-to-end suite and browser checks across all seven pages;
   verify regeneration/reload, offline operation, boundary/isolate searches and
   all required exports under Docker Python 3.11.
5. After required work, expose temporal/route evidence and implement validated
   AI node links. Keep the core workflow available without an API key.

**Done when:** another teammate can reproduce the complete result from the
handoff, find any supplied gid, understand roles/clusters, and perform the live
demo without hidden local state or required external assets.

### Coordination and integration order

1. Agree on F01's exact CSVs, the rich feature artifact, availability flags and
   metadata before further edits. Member 3 owns shared pipeline/README edits;
   Members 1 and 2 supply their own contract documents.
2. Work in parallel within the ownership boundaries above. Integrate feature
   changes first, decision/export changes second, then the viewer consumer.
3. Finish all required fixes before adding bonus detectors. Do not tune role
   labels to organizer examples or promise supervised accuracy without labels.
4. Freeze decision rules and dependencies, regenerate outputs, validate and
   rehearse. A final integration window should be reserved rather than using the
   entire remaining time on new features.

## 7. Final acceptance checklist

- [ ] Clean checkout, documented setup, one pipeline command, under 300 seconds.
- [ ] Exactly the required column sets and types in each of the three CSVs.
- [ ] 2,248 unique gids matching input, all required values populated.
- [ ] Valid role dictionary, bounded finite scores, readable evidence <=200 chars.
- [ ] All clusters reconcile to member nodes, seeds, internal edges and top gids.
- [ ] At least 20 ranked nodes with matching node scores and useful priority reasons.
- [ ] Written formulas and thresholds; three arbitrary gids explained within a minute.
- [ ] Directed viewer with roles/clusters; connected, isolated and boundary gids work.
- [ ] Large gids remain exact in CSV, Python, browser nodes, edges and navigation.
- [ ] Seeds and depth-4 nodes never produce unsupported ratio/terminal claims.
- [ ] Date-only observations and missing windows are described honestly.
- [ ] Core viewer works offline after installation; LLM remains optional.
- [ ] Repeatability, schema and regression checks pass in the final Python 3.11 image.
- [ ] Final exports, source, diagram, documentation and rehearsed demo are delivered.

## 8. Repository hygiene completed in this audit

`.gitignore` now covers Python environments/build output, bytecode, test and
analysis caches, coverage, notebook checkpoints, local credentials and Streamlit
secrets, generated `out/` and `exports/`, logs, temporary files, personal Compose
overrides, editor state and OS metadata. Sanitized `.env.example` and templates
remain eligible for tracking.

The rules intentionally preserve source, supplied `data/*.parquet`, dependency
manifests, shared Docker configuration, documentation and final `submission/`
CSVs. Ignoring all CSV/Parquet files would conceal required deliverables. Git
ignore rules do not remove previously tracked files or filter a Docker build
context; final Docker exclusions remain part of Member 3's work above.
