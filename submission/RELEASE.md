# Delivery verification — 2026-09-23

The final package was generated from the integrated feature, decision and analyst
workflow implementation. Required CSVs passed strict validation against supplied
source data. This verifies contracts and behavior, not the truth of hypotheses.

## Revision and integration

- Branch: `codex/brief-ui`, created from shared `origin/testing@b601280`.
- Member 1: `bf4859c` (validation, canonical features and temporal evidence),
  integrated first by merge `ca109c6`.
- Member 2: `bf5776e` (roles, priority, clustering, explanations and native
  exports), integrated second by merge `d1ca5ab`.
- Final implementation: `e075b643a1c9272905316b522f4cf426274047f3`.
- Artifacts and this record are committed after that implementation revision;
  the manifest identifies the code that generated them.

No teammate-owned graph or decision module was directly edited by the UI owner.
The temporary CSV serialization compatibility layer was removed in favor of the
native Member 2 exporter. Member contracts are linked from README.

Changed owned files: `app.py`, `src/ai_assistant.py`, `pipeline.py`, `Dockerfile`,
`compose.yaml`, `.dockerignore`, `requirements.txt`, `constraints.txt`,
`requirements-ai.txt`, `requirements-dev.txt`, `tests/test_app.py`,
`tests/test_ai_assistant.py`, `tests/test_integration.py`,
`tests/test_browser_integration.py`, `README.md`, `DEMO.md`,
`documentation/architecture.md`, and `submission/`.
`main.py` already delegates to the pipeline; `.gitignore` already keeps normal
`out/` disposable while preserving the submission package.

## Final artifacts and timing

| Artifact | Rows | Columns |
|---|---:|---:|
| `submission/nodes_roles.csv` | 2,248 | exactly 6 |
| `submission/clusters.csv` | 91 | exactly 6 |
| `submission/top_nodes.csv` | 50 | exactly 5 |
| `out/node_features.parquet` | 2,248 | 180 |
| `out/resilience.csv` | 5 | 6 |

The three CSVs and `release_metadata.json` are tracked. Rich auxiliary data and
`run_metadata.json` are reproducibly generated in ignored `out/`.
`submission/.gitattributes` preserves CSV/JSON bytes across checkouts.

Final run ID: `f4c16f10-bf89-4b9d-8c58-59bea5f85c83`.
The clean runtime image ran:

```bash
python main.py --data data --out out --submission submission
```

Measured pipeline time through publication: **8.24 seconds**.
Manifest time through validation: **8.115235 seconds**.
Host elapsed time including Docker process startup: **10.4136 seconds**.
All are below the required 300 seconds; installation/build time is excluded.

Source data was mounted read-only from a clean detached checkout of `e075b64`.
Output and submission directories were explicit separate writable mounts;
runtime external networking was disabled. `MONEY_GRAPH_GIT_REVISION` attributed
the image to that commit because the Docker context excludes `.git`.

The manifest records input/output hashes, source/rule hashes, schema identities,
command, environment and timing. Package CSVs were checked byte-for-byte against
strictly validated `out/` files and their SHA-256 values. The release manifest's
analytics-manifest digest and run ID match final `out/run_metadata.json`.

Final roles: terminal 1,100; peripheral 602; distributor 303; consolidator 108;
transit 85; coordinator 50. All 19 isolated seeds remain present with peripheral
role strength zero. No depth-4 node is labeled terminal. Unavailable seed/boundary
relay ratios remain missing with explicit validity reasons. Demo examples and
cluster IDs were refreshed from these results.

## Verification performed

Runtime: CPython **3.11.16**, Linux amd64 / Docker Desktop WSL2.
Dependencies: pinned manifests and constraints, including pytest **9.1.1** and
Playwright **1.55.0**. Browser: Chromium **153.0.8010.52**.

- **434 tests passed in 144.32 seconds**, no failures or skips, using the final
  clean checkout mounted read-only into the rebuilt Python 3.11 test image.
  The container used `--network none`.
- All seven actual browser pages rendered; embedded graph canvases executed;
  the browser route guard observed no external asset request.
- Queue → card and cluster → graph navigation, regeneration/reload and visible
  run information passed. Three supplied connected/boundary/isolated gids opened
  with evidence in **1.58 seconds**, below the one-minute interaction limit.
- AppTest/unit coverage includes all queue filters and empty states, unknown and
  exact large gids, all 2,248 full-graph nodes, singleton clusters, focus retention
  after graph truncation, cluster coloring and the Altair histogram regression.
- Feature joins, independent role/priority explanations, score contributions,
  censoring flags, route/amount/anomaly evidence and mixed/offset date normalization
  are covered. Row permutations preserve the integrated 180-column feature
  artifact and required CSV decisions.
- Hash mutations, source/export mismatch, failed publication, source replacement,
  concurrent regeneration and stale-cache/stale-AI-response cases are covered.
- Optional AI: **37 mocked/local tests** cover failed tools, missing nodes,
  unsupported claims, exact citations/navigation and sampling/search caveats.
  Core runtime has neither an OpenAI SDK nor an API-key requirement.
- Compose analytics completed with exit code 0 before the new app started.
  App health became **healthy** at `127.0.0.1:8501`. Logs advertise localhost;
  the explicit browser address avoids Streamlit external-IP discovery.
- Compose regeneration took **10.13 seconds**, then strict artifact/provenance
  validation passed. The final package was regenerated afterward as above.
- Strict standalone validation passed under optimized Python (`python -O`),
  using raw edges for independent internal-turnover reconciliation.

Representative verification commands, from the repository root:

```bash
docker build --target runtime -t money-graph:local .
docker build --target test -t money-graph:brief-browser .
docker run --rm --network none --mount type=bind,source=ABSOLUTE_CLEAN_CHECKOUT,target=/app,readonly money-graph:brief-browser python -m pytest -q -p no:cacheprovider --durations=5 -rP
docker compose -p money-graph-brief-check up -d --no-build
docker compose -p money-graph-brief-check run --rm analytics
docker compose -p money-graph-brief-check exec -T app python main.py --data data --out out --validate-only
docker compose -p money-graph-brief-check exec -T app python -O validate_submission.py --out out --data data
```

Replace `ABSOLUTE_CLEAN_CHECKOUT` with the absolute clean source path. Normal
reproduction uses `docker compose up --build -d` and
`docker compose --profile test run --build --rm tests` as documented in README.
The standalone validator targets complete `out/`, including the auxiliary
feature artifact; bare three CSVs are not a complete viewer/validator input.

## Remaining limitations

- Live external AI calls were not tested; only mocked/local grounding behavior
  is verified. Enabling AI sends questions and selected tool results externally.
- The five-minute demo has real final examples and an automated browser rehearsal;
  a human narrator's spoken five-minute delivery remains unverified.
- Optional import/run UI and analyst review-list download were not implemented.
  The validated CLI remains the supported run workflow.
- One-million-node execution is a documented scaling design, not a benchmark.
  Native Windows dependencies were not the release test platform; verification
  used Docker Linux/Python 3.11 on the Windows host.
- July/date-only/intrabank/threshold sampling, incomplete seed inflows, hop-4
  truncation and bounded route searches limit interpretation. Scores are
  heuristics; no ground truth establishes accuracy or guilt. Dataset use remains
  hackathon-only.
