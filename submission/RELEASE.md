# Delivery verification — 2026-09-23

This package is generated from the current integrated baseline and the completed
UI/delivery changes. It passes strict structural submission validation. It is
**not a claim that every analytical audit finding has been closed**.

## Revision and integration

- Branch: `codex/brief-ui`, created from shared `origin/testing@b601280`.
- Existing Member 1 graph commit: `329e432`; Member 2 decision commit: `34d1c34`.
  Both were already ancestors of the shared testing commit.
- UI/pipeline/tests implementation: `e478f8b347f6998bb927fb699e3808e54dfa8262`.
- Final runtime configuration: `d05b9f8ee756f1dd1ad2d974e8bbd9f0b4841721`.
- No newer Member 1/2 audit-fix branches were published when remote refs were
  checked. Their modules were not edited or copied into this branch.

Changed owned code: `app.py`, `src/ai_assistant.py`, `pipeline.py`,
`Dockerfile`, `compose.yaml`, `.dockerignore`, dependency manifests/constraints,
`tests/test_app.py`, `tests/test_ai_assistant.py`, new integration/browser tests,
`README.md`, `DEMO.md`, and `documentation/architecture.md`.
`main.py` already delegates to the shared pipeline; `.gitignore` already
preserves submission files while ignoring disposable outputs.

## Final package

| File | Rows | Columns |
|---|---:|---:|
| `nodes_roles.csv` | 2,248 | exactly 6 |
| `clusters.csv` | 91 | exactly 6 |
| `top_nodes.csv` | 50 | exactly 5 |

`release_metadata.json` records input and output SHA-256 hashes, rule/source
hashes, environment, command, run ID and timings. `.gitattributes` preserves
CSV/JSON bytes on Windows and POSIX checkouts so those hashes remain meaningful.
The auxiliary `node_features.parquet`, `resilience.csv` and full run manifest
are reproducibly generated in ignored `out/`.

Final run: `52a88928-1997-4664-97fd-33f7fe96515d`.
Measured complete pipeline: **4.36 seconds**; metadata records **4.209944
seconds** through validation before package publication. Both are below 300s.

## Verification performed

The implementation was checked out separately with a clean Git status, built
with the pinned Python 3.11 base image and installed from pinned constraints.
Runtime: CPython **3.11.16**, Linux amd64 / Docker Desktop WSL2.
Browser: Chromium **153.0.8010.52**, Playwright **1.55.0**.

- Full suite using the clean checkout as a read-only source mount, external
  network disabled: **117 passed in 90.80s**, no skips or failures.
- Seven actual browser pages and graph canvases work with external access
  disabled. The browser route guard observed no external asset request.
- Connected, isolated, boundary and unknown gids; singleton/full graph;
  >2^53 identifiers; filter/navigation; histogram; artifact mutation;
  failed publication; concurrent regeneration and reload are covered.
- Three arbitrary input gids opened with evidence in **1.59 seconds** in the
  focused browser rehearsal. This is an interaction measurement, not a claim
  about a human narrator's spoken presentation.
- AI: **29 mocked/local tests**, no live API request; core runtime has neither
  the OpenAI SDK nor an API key requirement.
- Compose analytics completed with exit code 0 before the app started.
  App health became **healthy** at `127.0.0.1:8501`.
- Compose regeneration completed in **4.24s**, followed by successful strict
  artifact/provenance validation.
- Final startup fix (`d05b9f8`) explicitly sets browser.serverAddress, preventing
  Streamlit's external-IP discovery. Rebuilt clean runtime, restarted Compose,
  checked localhost-only welcome URL and healthy service, then regenerated the
  final package. Application modules/tests are identical to the full-suite commit.

Commands used (replace local mount paths with your checkout):

```bash
docker build --target test -t money-graph:brief-browser .
docker run --rm --network none --mount type=bind,source=ABSOLUTE_CLEAN_CHECKOUT,target=/app,readonly money-graph:brief-browser python -m pytest -q -p no:cacheprovider --durations=5
docker build --target runtime -t money-graph:local .
docker compose -p money-graph-brief-check up -d --no-build
docker compose -p money-graph-brief-check run --rm analytics
docker compose -p money-graph-brief-check exec -T app python main.py --data data --out out --validate-only
```

Final packaging was run in the clean runtime image with `--network none`,
`data/` mounted read-only and explicit writable mounts for `out/` and
`submission/`; command: `python main.py --data data --out out --submission submission`.
The final metadata's Git revision was supplied via `MONEY_GRAPH_GIT_REVISION`
because the image intentionally excludes `.git`.

Normal reproduction:

```bash
docker compose up --build -d
docker compose --profile test run --build --rm tests
python main.py --data data --out out --submission submission
python main.py --data data --out out --validate-only
```

The last two commands use the documented local Python 3.11 environment; for a
Docker-only export, mount the host submission directory to `/app/submission`
and pass `--submission /app/submission` to the analytics service.

## Remaining analytical handoffs

The compatibility split produces the mandatory strict schemas without changing
the existing decisions. Full post-audit acceptance still requires the members'
completed commits:

1. Member 1: robust raw-data contracts, canonical/permutation-stable construction,
   temporal month-end/hop-4 availability, exact feature fixtures and the promised
   feature/data-quality documentation.
2. Member 2: canonical clustering and percentile policy, winning-rule/boundary
   explanations, ranking reasons independent of role evidence, full decision-rule
   documentation, and the finalized native export/validator contract.

Specifically, current `top_nodes.why` repeats role evidence. Some boundary
consolidators' evidence is only a boundary message. The viewer shows the real
metric/contribution tables and observation caveats, but does not invent missing
analytical reasoning. Current feature relay zeroes at the hop boundary are
explicitly qualified as censored; the feature-engine fix remains pending.

Member-owned contract documents are absent, so README links to the existing
source implementations and lists the expected handoff paths rather than claiming
those documents exist. Optional repeated-route detectors and import/review-list
convenience workflows were not added. Live AI testing and a human five-minute
spoken rehearsal remain unverified.

After those commits arrive: integrate Member 1, then Member 2; remove any obsolete
serialization compatibility seam; rerun the complete suite, regenerate this
package and refresh the three real examples in DEMO. Preserve seed and hop-4
protections throughout.
