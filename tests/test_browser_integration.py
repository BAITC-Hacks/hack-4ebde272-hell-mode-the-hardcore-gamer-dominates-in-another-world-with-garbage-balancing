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
                context = browser.new_context(viewport={"width": 1440, "height": 1000})
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
                    api.expect(page.get_by_role("heading", name=f"gid {int(gid):,}", exact=False)).to_be_visible()
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
                assert not external, f"Core viewer requested external assets: {external}"
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=15)
