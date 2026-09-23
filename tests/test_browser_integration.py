"""Real Chromium rehearsal with runtime egress disabled.

Docker's test target installs Chromium; Compose additionally uses network_mode:
none. The browser route guard makes attempted external asset requests a failure.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd
import pytest

from pipeline import run

ROOT = Path(__file__).resolve().parents[1]


def test_offline_browser_rehearsal(tmp_path, monkeypatch):
    api = pytest.importorskip("playwright.sync_api")
    chromium = shutil.which("chromium")
    if not chromium:
        pytest.skip("Run Docker test target for the Chromium offline acceptance test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "out"
    run(ROOT / "data", output)
    source = pd.read_parquet(ROOT / "data/nodes.parquet")
    edges = pd.read_parquet(ROOT / "data/edges.parquet")
    known_edges = set(edges.src) | set(edges.dst)
    isolated = str(source.loc[~source.gid.isin(known_edges), "gid"].iloc[0])
    connected = str(edges.src.iloc[0])
    boundary = str(source.loc[source.depth.eq(4), "gid"].iloc[0])
    env = {**os.environ, "MONEY_GRAPH_OUTPUT_DIR": str(output),
           "MONEY_GRAPH_DATA_DIR": str(ROOT / "data"),
           "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
    log_path = tmp_path / "streamlit.log"
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
             "--server.address=127.0.0.1", "--server.port=8597",
             "--server.headless=true", "--browser.gatherUsageStats=false"],
            env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with urlopen("http://127.0.0.1:8597/_stcore/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if time.monotonic() > deadline:
                        pytest.fail(log_path.read_text(encoding="utf-8"))
                    time.sleep(0.1)
            with api.sync_playwright() as p:
                browser = p.chromium.launch(executable_path=chromium, args=["--no-sandbox"])
                context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
                external = []

                def guard(route):
                    if urlparse(route.request.url).hostname not in {"127.0.0.1", "localhost", None}:
                        external.append(route.request.url)
                        route.abort()
                    else:
                        route.continue_()

                context.route("**/*", guard)
                page = context.new_page()
                page.goto("http://127.0.0.1:8597")
                api.expect(page.get_by_role("heading", name="Money Graph · Analyst workspace")).to_be_visible(timeout=30000)

                def workspace(name, title=None):
                    page.get_by_test_id("stSidebar").get_by_text(name, exact=True).click()
                    api.expect(page.get_by_role("heading", name=title or name, exact=True)).to_be_visible(timeout=15000)
                    api.expect(page.get_by_test_id("stException")).to_have_count(0)

                workspace("Investigation queue")
                page.get_by_role("button", name="Open node card", exact=True).click()
                api.expect(page.get_by_role("heading", name="Node card", exact=True)).to_be_visible()
                search = page.get_by_role("textbox", name="Search arbitrary gid")
                challenge_started = time.monotonic()
                for gid in (connected, boundary, isolated):
                    search.fill(gid)
                    search.press("Enter")
                    api.expect(page.get_by_role("heading", name=f"gid {gid}", exact=False)).to_be_visible()
                    api.expect(page.get_by_text("Role evidence", exact=True)).to_be_visible()
                challenge_seconds = time.monotonic() - challenge_started
                assert challenge_seconds < 60, "Three arbitrary-gid cards exceeded the one-minute challenge"
                print(f"three arbitrary-gid cards: {challenge_seconds:.2f}s")
                search.fill("-99")
                search.press("Enter")
                api.expect(page.get_by_text("That gid is not", exact=False)).to_be_visible()
                workspace("Network explorer")
                graph_search = page.get_by_role("textbox", name="Selected gid", exact=True)
                graph_search.fill(isolated)
                graph_search.press("Enter")
                # A rendered canvas verifies that embedded JS actually runs offline.
                api.expect(page.locator("iframe")).to_have_count(1, timeout=20000)
                api.expect(page.frame_locator("iframe").locator("canvas")).to_be_visible(timeout=20000)
                workspace("Cluster review")
                page.get_by_role("button", name="Open this cluster in Network explorer").click()
                api.expect(page.get_by_role("heading", name="Network explorer")).to_be_visible()
                api.expect(page.frame_locator("iframe").locator("canvas")).to_be_visible(timeout=20000)
                workspace("Resilience", "Resilience · structural concentration")
                workspace("AI analyst", "Optional AI analyst")
                api.expect(page.get_by_text("Core investigation features work without an API key.", exact=False)).to_be_visible()
                workspace("Overview", "Money Graph · Analyst workspace")
                # Re-publish at the same paths and exercise the explicit reload.
                run(ROOT / "data", output)
                new_run = json.loads((output / "run_metadata.json").read_text())["run_id"]
                page.get_by_role("button", name="Reload data", exact=True).click()
                api.expect(page.get_by_role("heading", name="Money Graph · Analyst workspace")).to_be_visible()
                api.expect(page.get_by_text(f"Verified run: {new_run}", exact=True)).to_be_visible()
                api.expect(page.get_by_test_id("stException")).to_have_count(0)

                # Complete the brief's analyst handoff: choose exact accounts,
                # download their evidence, then upload/run a fresh private case.
                workspace("Investigation queue")
                shortlist = page.get_by_role("combobox", name="Review shortlist", exact=True)
                for gid in (boundary, isolated):
                    shortlist.fill(gid)
                    page.get_by_role("option", name=gid, exact=True).click()
                api.expect(page.get_by_test_id("stException")).to_have_count(0)
                review_download = page.get_by_test_id("stDownloadButton").filter(has_text="Download review shortlist").locator("a, button")
                with page.expect_download() as download:
                    review_download.click()
                review = pd.read_csv(download.value.path(), dtype={"gid": "string"})
                assert set(review.gid) == {boundary, isolated}
                assert review.run_id.eq(new_run).all()
                assert review.loc[review.gid.eq(boundary), "next_request"].str.contains("beyond hop 4").all()
                assert review.loc[review.gid.eq(isolated), "next_request"].str.contains("isolated gid").all()

                page.get_by_text("Upload case dataset", exact=True).click()
                for name, label in (("nodes", "Nodes Parquet"), ("edges", "Edges Parquet"),
                                    ("transactions", "Transactions Parquet")):
                    page.get_by_label(label, exact=True).locator('input[type="file"]').set_input_files(str(ROOT / f"data/{name}.parquet"))
                upload_run = page.get_by_role("button", name="Validate and run uploaded case", exact=True)
                api.expect(upload_run).to_be_enabled()
                upload_started = time.monotonic()
                upload_run.click()
                api.expect(page.get_by_text("Uploaded case is ready.", exact=False)).to_be_visible(timeout=60000)
                print(f"uploaded case to verified viewer: {time.monotonic() - upload_started:.2f}s")
                api.expect(review_download).to_be_disabled()
                active_run_text = page.get_by_test_id("stSidebar").get_by_text("Verified run:", exact=False).inner_text()
                assert new_run not in active_run_text
                assert json.loads((output / "run_metadata.json").read_text())["run_id"] == new_run
                with page.expect_download() as download:
                    page.get_by_test_id("stDownloadButton").filter(has_text="Download submission bundle").locator("a, button").click()
                with zipfile.ZipFile(download.value.path()) as archive:
                    assert set(archive.namelist()) == {"nodes_roles.csv", "clusters.csv", "top_nodes.csv", "release_metadata.json"}
                    assert len(pd.read_csv(archive.open("nodes_roles.csv"))) == 2248
                    uploaded_run = json.loads(archive.read("release_metadata.json"))["run_id"]
                    assert uploaded_run in active_run_text
                # Bad replacements never evict a valid active case or overwrite
                # the configured source/output mounts.
                page.get_by_label("Nodes Parquet", exact=True).locator('input[type="file"]').set_input_files({
                    "name": "nodes.parquet", "mimeType": "application/octet-stream", "buffer": b"invalid parquet",
                })
                api.expect(upload_run).to_be_enabled()
                upload_run.click()
                api.expect(page.get_by_text("Could not import uploaded case:", exact=False)).to_be_visible()
                api.expect(page.get_by_text(f"Verified run: {uploaded_run}", exact=True)).to_be_visible()
                page.get_by_role("button", name="Use configured dataset", exact=True).click()
                api.expect(page.get_by_text(f"Verified run: {new_run}", exact=True)).to_be_visible()
                api.expect(page.get_by_test_id("stException")).to_have_count(0)
                assert not external, f"Core viewer requested external assets: {external}"
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=15)
