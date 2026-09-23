"""Readable, safely serialized node tooltips with an offline browser check."""

from __future__ import annotations

import json
import re
import shutil

import pandas as pd
import pytest

from app import pyvis_html


GID = 100000006866783100
HOSTILE_ROLE = '</script><script>window.tooltip_injected=true</script><a href="x">review</a>'


def render_graph(role: str = "distributor") -> str:
    nodes = pd.DataFrame({
        "gid": [GID], "role": [role], "cluster_id": [9],
        "priority_score": [0.922], "is_seed": [True],
    })
    edges = pd.DataFrame(columns=["src", "dst", "sum_kzt", "n_tx"])
    rendered = pyvis_html(edges, nodes, focus_gid=GID)
    assert rendered is not None
    return rendered


def node_records(rendered: str) -> list[dict]:
    match = re.search(r"nodes = new vis.DataSet\((\[.*?\])\);", rendered, re.S)
    assert match is not None
    return json.loads(match.group(1))


def test_node_tooltip_is_readable_plain_text_with_exact_identifier():
    rendered = render_graph()
    node = node_records(rendered)[0]
    assert node["id"] == str(GID)
    assert node["label"] == f"{GID} ★"
    assert node["title"] == (
        f"gid {GID}\nRole: distributor\nCluster: 9\nPriority: 0.922\nSeed: Yes"
    )
    assert "<b>" not in node["title"] and "<br>" not in node["title"]
    assert 'white-space: pre-line' in rendered
    assert 'max-width: min(360px, calc(100vw - 32px))' in rendered
    assert 'overflow-wrap: anywhere' in rendered


@pytest.mark.parametrize("role", [HOSTILE_ROLE, 'operator & <review>', 'plain href label'])
def test_imported_role_cannot_enable_html_popup_or_escape_script(role):
    rendered = render_graph(role)
    assert f"Role: {role.lower()}\n" in node_records(rendered)[0]["title"]
    assert "popup.innerHTML = nodeData[0].title" not in rendered
    assert "function showPopup(nodeId)" not in rendered
    assert "<script>window.tooltip_injected=true</script>" not in rendered
    assert '<a href="x">review</a>' not in rendered


def test_edges_keep_direction_and_plain_text_amount_tooltips():
    target = GID + 1
    nodes = pd.DataFrame({"gid": [GID, target], "role": ["transit", "distributor"]})
    edges = pd.DataFrame({"src": [GID], "dst": [target], "sum_kzt": [15000], "n_tx": [3]})
    rendered = pyvis_html(edges, nodes)
    match = re.search(r"edges = new vis.DataSet\((\[.*?\])\);", rendered, re.S)
    edge = json.loads(match.group(1))[0]
    assert edge["from"] == str(GID) and edge["to"] == str(target)
    assert edge["arrows"] == "to"
    assert edge["title"] == "15,000 KZT · 3 transaction(s)"


@pytest.mark.parametrize("role,width", [
    ("distributor", 900),
    (HOSTILE_ROLE, 320),
    ("review_" + "longlabel" * 35, 320),
])
def test_real_hover_renders_lines_and_wraps_without_executing_html(role, width):
    api = pytest.importorskip("playwright.sync_api")
    chromium = shutil.which("chromium")
    if not chromium:
        pytest.skip("Run Docker test target for the Chromium tooltip regression")
    rendered = render_graph(role)
    with api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chromium, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": width, "height": 800})
            requested_urls = []

            def block_request(route):
                requested_urls.append(route.request.url)
                route.abort()

            page.route("**/*", block_request)
            page.set_content(rendered, wait_until="load")
            position = page.evaluate("""gid => {
                network.setOptions({physics: {enabled: false}});
                network.moveNode(gid, 0, 0);
                network.moveTo({position: {x: 0, y: 0}, scale: 1});
                network.redraw();
                const point = network.canvasToDOM(network.getPosition(gid));
                const box = document.querySelector('#mynetwork').getBoundingClientRect();
                return {x: point.x + box.x, y: point.y + box.y};
            }""", str(GID))
            page.mouse.move(position["x"], position["y"])
            tooltip = page.locator(".vis-tooltip")
            api.expect(tooltip).to_be_visible(timeout=5000)
            assert tooltip.inner_text() == node_records(rendered)[0]["title"]
            assert tooltip.locator("script, a, img").count() == 0
            assert page.evaluate("window.tooltip_injected === undefined")
            assert tooltip.evaluate("el => getComputedStyle(el).whiteSpace") == "pre-line"
            assert tooltip.evaluate("el => el.scrollWidth <= el.clientWidth")
            bounds = tooltip.bounding_box()
            assert bounds is not None
            assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= width
            assert bounds["height"] >= 5 * 13
            assert not requested_urls
        finally:
            browser.close()
