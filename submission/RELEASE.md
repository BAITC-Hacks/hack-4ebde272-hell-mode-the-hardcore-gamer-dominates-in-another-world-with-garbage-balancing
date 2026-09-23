# Final delivery verification — 2026-09-23

## Implementation and artifacts

Generated from clean implementation commit
`1d80035259e1efddd90efd3c95f627a8dfcd1369` on `testing`, which includes all three
members' branches, the complete analyst workflow, readable explanations and
optional-AI configuration, plus responsive horizontal chart labels and readable
node hover cards.

| Artifact | Rows | Columns |
|---|---:|---:|
| `nodes_roles.csv` | 2,248 | exactly 6 |
| `clusters.csv` | 91 | exactly 6 |
| `top_nodes.csv` | 50 | exactly 5 |
| auxiliary `out/node_features.parquet` | 2,248 | 180 |
| auxiliary `out/resilience.csv` | 5 | 6 |

The three required CSVs are byte-identical to release `939a4c0`, retaining its
2,248 readable node explanations, 91 group hypotheses and 50 priority reasons.
This UI update preserves all gids, roles, scores, ranks, memberships, counts,
turnover and top members. Node evidence has at most **183 characters** (limit 200).
The queue uses the published `top_nodes.csv.why` text and shows the selected
account's full reason. Rich outputs remain in ignored `out/`.

Final release run: `359ea5d7-1505-4a40-a8bd-de345bfd0b12`.
Pipeline time including publication: **4.34 seconds**.
Manifest computation/validation time: **4.325137 seconds**.
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
including `app.py`, `src/workspace.py`, `src/ai_assistant.py`, `src/config.py`
and `requirements-ai.txt`.
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

- **618 tests passed in 107.46 seconds**, no failures or skips, with source mounted
  read-only into the pinned Python 3.11 test image and external networking disabled.
- Real Chromium tested all seven pages, graph canvases, exact-gid lookup,
  navigation, regeneration/reload, review-list selection/download, private
  upload/run, submission ZIP download, malformed-upload preservation and reset.
  It observed no external asset requests; final browser test time was 28.67 seconds.
  Upload replacement waits for the server to acknowledge the new selection.
- Regression coverage includes strict schemas/reconciliation, exact numerical
  features, input permutations, censored derived ratios, source/output races,
  downloaded-run identity, stale selections and grounded AI edge citations.
- Chart tests preserve category order and exact values while keeping labels
  horizontal. Real Chromium hover tests verify exact gids, multiline text and
  wrapping at 320px, including hostile HTML labels without script execution or
  external requests. The rebuilt overview was also checked visually.
- Compose analytics exited successfully; the rebuilt app is healthy at
  `127.0.0.1:8501`. Source data and viewer outputs remain read-only mounts.
- The actual OpenAI SDK completed a tool-call/citation round trip using an
  in-memory HTTP transport. Simulated 401/429/500 responses were handled without
  revealing credentials or provider response text. No live API request was made.

Environment: CPython **3.11.16**, Linux aarch64 / Docker Desktop; pandas **3.0.6**,
NumPy **2.4.6**, NetworkX **3.6.1**, PyArrow **25.0.1**, Streamlit **1.64.0**,
Altair **6.3.0**, PyVis **0.3.2**, pytest **9.1.1**, Playwright **1.55.0** and
Chromium **153.0.8010.52**. Compose now includes OpenAI SDK **3.19.0**; core
analytics and investigation pages still require no API key.

## Optional AI setup

Local `.env` exists with `OPENAI_API_KEY=` left blank and
`OPENAI_MODEL=gpt-4o-mini`. Its permissions are `0600`; Git ignores it and the
Docker image excludes it. Only the empty `.env.example` template is tracked.
After adding a key locally, run:

```bash
docker compose up -d --no-deps --force-recreate app
```

Native Streamlit rereads `.env` on rerun. Nonempty process settings take precedence.
The panel explains setup and disables its Ask action while the key is blank.
Questions and bounded tool results go to OpenAI only after the user clicks Ask.

## Remaining verification limits

The team must rehearse the actual [five-minute demo](../DEMO.md) and spoken
explanations. Live optional external LLM calls were not made; deterministic and
mocked behavior is tested. There are no ground-truth role labels, so this verifies
contracts and behavior rather than analytical accuracy or guilt.

See the [final brief acceptance record](../documentation/final-acceptance.md) for
all mandatory checks and judging-criteria evidence. The earlier
[integration record](../documentation/integration-verification.md) is historical;
it predates the completed upload/review/download workflow.
