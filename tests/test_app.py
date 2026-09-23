"""Render every workspace and exercise navigation against real pipeline exports."""
from pathlib import Path
import json
import re

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import load_data, merged_nodes, pyvis_html
from pipeline import run

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Overview", "Investigation queue", "Node card", "Network explorer", "Cluster review", "Resilience", "AI analyst"]


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    destination = tmp_path_factory.mktemp("integration_exports")
    run(ROOT / "data", destination)
    return destination


@pytest.fixture
def ui(exports, monkeypatch):
    monkeypatch.setenv("MONEY_GRAPH_OUTPUT_DIR", str(exports))
    monkeypatch.setenv("MONEY_GRAPH_DATA_DIR", str(ROOT / "data"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()


def button(ui, label):
    return next(item for item in ui.button if item.label == label)


def page(ui, name):
    ui.sidebar.radio[0].set_value(name).run()
    assert not ui.exception, [error.message for error in ui.exception]
    return ui


@pytest.mark.parametrize("name", PAGES)
def test_all_sections_render(ui, name):
    page(ui, name)
    assert not ui.error
    if name == "Overview":
        assert len(ui.get("vega_lite_chart")) == 3
    if name == "Resilience":
        assert len(ui.get("vega_lite_chart")) == 3


def test_queue_covers_every_node_and_filters(ui, exports):
    page(ui, "Investigation queue")
    expected = pd.read_csv(exports / "nodes_roles.csv")
    assert len(ui.dataframe[0].value) == 2248
    assert all(isinstance(gid, str) for gid in ui.dataframe[0].value.gid)
    ui.multiselect[0].set_value(["peripheral"]).run()
    assert len(ui.dataframe[0].value) == expected.role.eq("peripheral").sum()
    ui.multiselect[0].set_value([]).run()
    ui.multiselect[2].set_value([4]).run()
    assert len(ui.dataframe[0].value) == expected.depth.eq(4).sum()
    ui.multiselect[2].set_value([]).run()
    next(x for x in ui.selectbox if x.label == "Seed status").set_value("Seed").run()
    assert len(ui.dataframe[0].value) == 81
    ui.number_input[0].set_value(1.0).run()
    assert ui.dataframe[0].value.empty
    assert not ui.exception


def test_queue_navigation_updates_card_and_selected_id(ui):
    page(ui, "Investigation queue")
    expected = ui.dataframe[0].value.iloc[0].gid
    button(ui, "Open node card").click().run()
    assert ui.sidebar.radio[0].value == "Node card"
    assert ui.text_input(key="node_gid_input").value == expected
    page(ui, "Investigation queue")
    select = next(x for x in ui.selectbox if x.label == "Open selected queue row")
    select.set_value(select.options[1]).run()
    expected = ui.dataframe[0].value.iloc[1].gid
    button(ui, "Open node card").click().run()
    assert ui.text_input(key="node_gid_input").value == expected
    assert not ui.exception
    metrics = {item.label: item.value for item in ui.metric}
    assert metrics["Seed reach"] != "Not exported"
    assert metrics["Anomaly score"] != "Not exported"
    assert any("Score contribution" in df.value.columns for df in ui.dataframe)


def test_node_search_handles_bad_unknown_and_boundary_ids(ui, exports):
    page(ui, "Node card")
    ui.text_input(key="node_gid_input").set_value("nonsense").run()
    assert ui.info and not ui.exception
    ui.text_input(key="node_gid_input").set_value("-99").run()
    assert ui.warning and not ui.exception
    nodes = pd.read_csv(exports / "nodes_roles.csv")
    gid = str(nodes.loc[nodes.depth.eq(4), "gid"].iloc[0])
    ui.text_input(key="node_gid_input").set_value(gid).run()
    assert any("beyond hop 4" in warning.value for warning in ui.warning)
    assert not ui.exception


def test_cluster_navigation_selects_requested_network(ui):
    page(ui, "Cluster review")
    ui.selectbox[0].set_value(ui.selectbox[0].options[1]).run()
    chosen = ui.selectbox[0].value
    button(ui, "Open this cluster in Network explorer").click().run()
    assert ui.sidebar.radio[0].value == "Network explorer"
    assert ui.radio(key="network_view").value == "Selected cluster"
    assert ui.selectbox(key="network_cluster_selection").value == chosen
    assert not ui.exception


def test_network_modes_and_unknown_gid(ui):
    page(ui, "Network explorer")
    ui.radio(key="network_view").set_value("Selected gid · 2 hops").run()
    assert not ui.exception
    ui.text_input(key="network_gid_input").set_value("-99").run()
    assert ui.info and not ui.exception
    ui.radio(key="network_view").set_value("Full network (optional)").run()
    ui.checkbox[0].check().run()
    assert not ui.exception


def test_network_keeps_large_ids_exact_and_embeds_vis_assets(exports):
    data = load_data(str(exports), str(ROOT / "data"))
    html = pyvis_html(data.edges.head(10), merged_nodes(data))
    nodes = json.loads(re.search(r"nodes = new vis.DataSet\((\[.*?\])\);", html, re.S).group(1))
    edges = json.loads(re.search(r"edges = new vis.DataSet\((\[.*?\])\);", html, re.S).group(1))
    expected = set(data.edges.head(10).src.astype(str)) | set(data.edges.head(10).dst.astype(str))
    assert {node["id"] for node in nodes} == expected
    assert all(edge["from"] in expected and edge["to"] in expected for edge in edges)
    assert 'src="lib/' not in html
    assert 'src="https://cdnjs.cloudflare.com/ajax/libs/vis-network' not in html


@pytest.mark.parametrize("name", PAGES)
def test_missing_files_render_useful_empty_states(tmp_path, monkeypatch, name):
    monkeypatch.setenv("MONEY_GRAPH_OUTPUT_DIR", str(tmp_path / "absent"))
    monkeypatch.setenv("MONEY_GRAPH_DATA_DIR", str(tmp_path / "absent"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ui = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    page(ui, name)
    assert not ui.exception


def test_exported_features_and_resilience_are_connected(exports):
    frame = pd.read_csv(exports / "nodes_roles.csv")
    required = {"pagerank", "betweenness", "seed_reach_count", "relay_2d_ratio", "peer_anomaly_score", "boundary_censored", "scc_size"}
    assert required.issubset(frame.columns)
    assert frame.loc[frame.is_seed | frame.boundary_censored, "pass_through"].isna().all()
    assert not frame.loc[frame.boundary_censored, "role"].eq("terminal").any()
    contributions = frame.filter(regex=r"^priority_.*_contribution$").sum(axis=1)
    assert (contributions - frame.priority_score).abs().max() < 1e-12
    resilience = pd.read_csv(exports / "resilience.csv")
    assert resilience.iloc[0].scenario == "baseline"
    assert resilience.iloc[0].fraction_baseline_largest == 1.0
