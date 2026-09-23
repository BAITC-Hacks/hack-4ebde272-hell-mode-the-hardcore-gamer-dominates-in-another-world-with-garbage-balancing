# Final delivery verification — 2026-09-23

## Implementation and artifacts

Generated from clean implementation commit
`c0a137a635b982edea69aa25e7c8aa18648c57f6` on `testing`, which includes all three
members' branches and the final analyst-workflow/correctness changes.

| Artifact | Rows | Columns |
|---|---:|---:|
| `nodes_roles.csv` | 2,248 | exactly 6 |
| `clusters.csv` | 91 | exactly 6 |
| `top_nodes.csv` | 50 | exactly 5 |
| auxiliary `out/node_features.parquet` | 2,248 | 180 |
| auxiliary `out/resilience.csv` | 5 | 6 |

The three required CSVs remain byte-identical to the previous verified release;
no ranking weights or supplied-data decisions changed. The release manifest was
regenerated for this implementation. Normal rich outputs remain in ignored `out/`.

Final release run: `f35dd311-b92a-473c-82b7-dfc62a7160e4`.
Pipeline time including publication: **4.19 seconds**.
Manifest computation/validation time: **4.170597 seconds**.
The 300-second requirement excludes dependency installation and image building.

## Clean release procedure

The final runtime image was rebuilt from the committed implementation. The
release ran with Docker `--network none`, source data mounted read-only, and
explicit writable mounts for host `out/` and `submission/`:

```bash
python main.py --data data --out out --submission submission
```

`MONEY_GRAPH_GIT_REVISION` was set to the full implementation commit above,
since the image excludes `.git`. Actual implementation hashes are also recorded,
including `app.py`, `src/workspace.py` and `src/ai_assistant.py`.
The manifest identifies its code/environment, not this later documentation commit.

Both validators passed against the fresh outputs:

```bash
python main.py --data data --out out --validate-only
python -O validate_submission.py --data data --out out
```

All CSV bytes and SHA-256 hashes were checked against `out/` and the release
manifest. The release and analytics manifests share the final run ID, and
`analytics_run_metadata_sha256` matches the exact final analytics manifest.
Input data remains unchanged. The package preserves the 19 isolated seeds and
all hop-4/seed observation limits.

## Tests and application

- **510 tests passed in 93.31 seconds**, no failures or skips, with source mounted
  read-only into the pinned Python 3.11 test image and external networking disabled.
- Real Chromium tested all seven pages, graph canvases, exact-gid lookup,
  navigation, regeneration/reload, review-list selection/download, private
  upload/run, submission ZIP download, malformed-upload preservation and reset.
  It observed no external asset requests; final browser test time was 26.41 seconds.
- A focused browser rehearsal opened three arbitrary-gid cards in 0.99 seconds
  and completed upload-to-verified-viewer in 6.10 seconds. These are measured
  software interactions, not a timed human narration.
- Regression coverage includes strict schemas/reconciliation, exact numerical
  features, input permutations, censored derived ratios, source/output races,
  downloaded-run identity, stale selections and grounded AI edge citations.
- Compose analytics exited successfully; the rebuilt app is healthy at
  `127.0.0.1:8501`. Source data and viewer outputs remain read-only mounts.

Environment: CPython **3.11.16**, Linux aarch64 / Docker Desktop; pandas **3.0.6**,
NumPy **2.4.6**, NetworkX **3.6.1**, PyArrow **25.0.1**, Streamlit **1.64.0**,
Altair **6.3.0**, PyVis **0.3.2**, pytest **9.1.1**, Playwright **1.55.0** and
Chromium **153.0.8010.52**. Core runtime has no OpenAI SDK or API-key requirement.

## Remaining verification limits

The team must rehearse the actual [five-minute demo](../DEMO.md) and spoken
explanations. Live optional external LLM calls were not made; deterministic and
mocked behavior is tested. There are no ground-truth role labels, so this verifies
contracts and behavior rather than analytical accuracy or guilt.

See the [final brief acceptance record](../documentation/final-acceptance.md) for
all mandatory checks and judging-criteria evidence. The earlier
[integration record](../documentation/integration-verification.md) is historical;
it predates the completed upload/review/download workflow.
