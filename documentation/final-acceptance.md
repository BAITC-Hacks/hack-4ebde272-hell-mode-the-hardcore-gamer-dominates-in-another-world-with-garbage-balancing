# Final brief acceptance — 2026-09-23

Reviewed against the supplied English HackAlem Money Graph brief and its five
mandatory checks, full analyst scenario, constraints, deliverables and judging
criteria. The earlier [integration record](integration-verification.md) documents
the merged baseline; this record covers the final workflow improvements.

## What was completed

- The viewer now accepts three Parquet uploads, validates and runs the production
  pipeline locally, and activates the case only after its outputs can be loaded.
  Source data and configured exports remain untouched; failed replacements keep
  the previous case. Temporary case files are removed on explicit reset.
- Analysts can select exact gids, keep their shortlist across pages and filters,
  and download existing decisions with priority reasons, observation limits,
  next data requests and a run ID. No request is sent to another party.
- A download packages the three strict submission CSVs and their release
  manifest. It is checked against the displayed run, including when publication
  occurs between page loading and download creation. Selections and AI answers
  reset on a new or changed run.
- Invalid pass-through flags now invalidate precomputed retention and flow
  balance too. The viewer uses the shared percentile policy for fallback values.
- Optional AI graph tools use deterministic tie ordering, and transfer-amount
  citations identify the exact directed edge. Claims still require matching
  scalar values from actual tool results.
- The queue's **Why** column uses the exact published `top_nodes.csv.why`
  wording, with a full-text view for its selected account. Other accounts use
  the full-run priority reason. Role evidence and group hypotheses now explain
  measured observations in everyday language, retaining the 200-character node
  limit and all observation warnings. Numerical decisions and ranks are unchanged.
- The AI panel reads a private local `.env` and stays usable as a setup form
  while the key is blank. Compose includes the SDK; tests exercise the actual
  SDK using an in-memory HTTP transport, including authentication/quota/service
  errors, without external requests or a real credential.
- Chart categories remain horizontal in overview, community and resilience
  views; daily dates use short horizontal labels. Node hover details use wrapped
  plain text, preserving exact gids without showing formatting tags. Real
  Chromium checks both narrow tooltips and labels containing hostile HTML.
- README, architecture and the five-minute demo now follow the complete analyst
  path. The demo starts with an actual live calculation and includes three real
  cases, an arbitrary-gid challenge, and the final review/export handoff.

These changes preserve the supplied dataset's decisions; no ranking weights were
changed. The final release manifest records the generating implementation and
source hashes, including the viewer, workspace helpers and AI implementation.

## Mandatory requirements

| Brief requirement | Acceptance evidence |
|---|---|
| M1: one command from raw Parquet to exports in less than 300 seconds | `python main.py --data data --out out --submission submission`; repeated complete runs, including a real browser upload, are exercised offline. Final clean release timing is recorded in [submission/RELEASE.md](../submission/RELEASE.md). |
| M2: every node has role, scores, cluster and nonempty evidence | 2,248 exact unique gids, allowed role labels, finite scores in [0,1], exact six-column node schema and evidence at most 200 characters. |
| M3: documented, explainable criteria and three arbitrary gids | Shared percentile semantics, explicit eligibility gates/weights/ties/missing-value rules, actual winning-rule diagnostics, and numeric contribution displays. Browser challenge opens connected, boundary and isolated cases within one minute. |
| M4: clustering and populated summaries | 91 communities cover all supplied nodes; six-column summaries reconcile members, seeds, internal turnover and top-gid membership. |
| M5: explained top list and directed searchable viewer | 50 ranked accounts, five-column schema, independent priority reasons, directed graph/role/cluster views, and exact arbitrary-gid search including isolates. |

The five checks have automated coverage. The final package also includes source,
README setup and rules, strict CSVs, an architecture diagram, and a substantive
demo script. The team still needs to perform the actual spoken demonstration.

## Judging criteria: inspectable evidence

| Criterion in the brief | What to demonstrate |
|---|---|
| Functionality / 25 points | Upload → validated local run → every supplied node → explained queue → selected review CSV and submission ZIP. |
| Technical implementation / 25 points | Canonical directed graph and communities; exact int64 IDs; censoring and missing-value contracts; bounded dated evidence; source/output hashes; race-safe publication/downloads; offline browser tests. |
| README and reproducibility / 25 points | Explicit Python 3.11 setup or one-command Docker startup, pinned core dependencies, strict independent validator, tracked outputs/manifest, diagram, and repeat/permutation tests. |
| Practical value / 15 points | Show why an account deserves review, why an apparently terminal boundary account is uncertain, and the specific data request included in the shortlist. |
| Originality and development / 10 points | Seed convergence, bounded recurring/return patterns, observed similar-amount evidence, individual peer anomalies, removal scenarios, grounded optional AI and a concrete million-node design discussion. |

This maps evidence to the rubric; it does not predict the jury's score.

## Verification commands

**618 tests passed in 107.46 seconds**, with no failures or skips, under the pinned
Python 3.11 Docker test environment with runtime networking disabled. The final
browser test completed in 28.67 seconds. Sources were mounted read-only. The
installed OpenAI SDK was exercised through in-memory HTTP responses; no live
API request was made. All 2,248 node explanations fit in 183 characters or fewer.

```bash
docker compose --profile test build analytics tests
docker run --rm --network none \
  --mount type=bind,source="$PWD",target=/app,readonly \
  money-graph:test python -m pytest -q -p no:cacheprovider --durations=8
docker compose exec -T app python main.py --data data --out out --validate-only
docker compose exec -T app python -O validate_submission.py --data data --out out
```

Final full-suite results and clean-release timing are recorded in
[submission/RELEASE.md](../submission/RELEASE.md). The expanded real Chromium
rehearsal passed with all seven pages, executing graph canvases, regeneration,
review selection/download, upload/run, submission download, malformed replacement
preservation and return to configured data. The earlier baseline upload-to-verified-viewer
step was **6.10 seconds**; three arbitrary-gid cards took **0.99 seconds** in that
focused rehearsal. The current full workflow is reverified above; no external
asset requests were observed.

Exact-value fixtures cover seed and hop-4 masks, month-end temporal windows,
multiple-seed reachability, four-versus-five hops, isolates, SCCs/reciprocity,
ties/singletons/zero MAD, row permutations, invalid inputs, strict cross-file
exports, publication failures, stale state and assistant grounding.

## Explicit limits

- Run the [five-minute demo](../DEMO.md) aloud as a team; software navigation
  timing cannot verify a human explanation or presentation delivery.
- Live optional LLM calls were not made. Local/mocked tools, claim grounding and
  node navigation are tested; core use has no API-key or paid-service dependency.
- Production upload is scoped to 2,248 nodes, July 2026 and transfers at least
  5,000 KZT. Resource limits and temporary-file lifetime are documented in README.
  Generic feature/input functions remain usable with small synthetic datasets.
- No labeled roles exist, so classification accuracy cannot be established.
  Date overlap does not trace the same money; incomplete seed inflows, hop-4
  outflows and end-of-period windows stay explicit. Below-threshold structuring
  is invisible. Bounded route evidence reports its search limits.
- Performance is measured on the supplied dataset. Million-node scaling is a
  documented design, not a claimed implementation or benchmark.
