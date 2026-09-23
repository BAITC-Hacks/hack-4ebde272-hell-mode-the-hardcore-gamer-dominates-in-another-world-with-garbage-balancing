"""Render every workspace and exercise navigation against real pipeline exports."""
from pathlib import Path
import hashlib
import json
import os
import re
from html.parser import HTMLParser

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import EXPORT_FILES, NODE_COLUMNS, SOURCE_FILES, exact_ids, load_data, merged_nodes, pyvis_html, selected_neighborhood
from pipeline import run

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Overview", "Investigation queue", "Node card", "Network explorer", "Cluster review", "Resilience", "AI analyst"]


def test_run_provenance_is_visible(ui, exports):
    metadata = json.loads((exports / "run_metadata.json").read_text())
    captions = "\n".join(caption.value for caption in ui.sidebar.caption)
    assert metadata["run_id"] in captions
    assert metadata["completed_at"] in captions
    assert str(metadata["runtime_seconds"]) in captions


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
    assert len(ui.dataframe[0].value) == pd.read_parquet(ROOT / "data" / "nodes.parquet").depth.eq(4).sum()
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
    nodes = pd.read_parquet(ROOT / "data" / "nodes.parquet")
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
    assert any("Unknown gid" in warning.value for warning in ui.warning) and not ui.exception
    ui.radio(key="network_view").set_value("Full network (optional)").run()
    ui.checkbox[0].check().run()
    assert not ui.exception
    assert any("2,248 nodes" in item.value for item in ui.caption)


def test_network_keeps_large_ids_exact_and_embeds_vis_assets(exports):
    data = load_data(str(exports), str(ROOT / "data"))
    expected = set(data.edges.head(10).src.astype(str)) | set(data.edges.head(10).dst.astype(str))
    selected = merged_nodes(data)
    html = pyvis_html(data.edges.head(10), selected[selected.gid.astype(str).isin(expected)])
    nodes = json.loads(re.search(r"nodes = new vis.DataSet\((\[.*?\])\);", html, re.S).group(1))
    edges = json.loads(re.search(r"edges = new vis.DataSet\((\[.*?\])\);", html, re.S).group(1))
    assert {node["id"] for node in nodes} == expected
    assert all(edge["from"] in expected and edge["to"] in expected for edge in edges)
    assert 'src="lib/' not in html
    assert 'src="https://cdnjs.cloudflare.com/ajax/libs/vis-network' not in html
    parser = AssetParser()
    parser.feed(html)
    assert not parser.remote_assets
    assert not parser.script_sources  # vis and supporting code are entirely inline


@pytest.mark.parametrize("name", PAGES)
def test_missing_files_render_useful_empty_states(tmp_path, monkeypatch, name):
    monkeypatch.setenv("MONEY_GRAPH_OUTPUT_DIR", str(tmp_path / "absent"))
    monkeypatch.setenv("MONEY_GRAPH_DATA_DIR", str(tmp_path / "absent"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ui = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    page(ui, name)
    assert not ui.exception


def test_exported_features_and_resilience_are_connected(exports):
    roles = pd.read_csv(exports / "nodes_roles.csv")
    assert list(roles) == NODE_COLUMNS
    frame = merged_nodes(load_data(str(exports), str(ROOT / "data")))
    required = {"pagerank", "betweenness", "seed_reach_count", "relay_2d_ratio", "peer_anomaly_score", "boundary_censored", "scc_size"}
    assert required.issubset(frame.columns)
    assert frame.loc[frame.is_seed | frame.boundary_censored, "pass_through"].isna().all()
    assert not frame.loc[frame.boundary_censored, "role"].eq("terminal").any()
    contributions = frame.filter(regex=r"^priority_.*_contribution$").sum(axis=1)
    assert (contributions - frame.priority_score).abs().max() < 1e-12
    resilience = pd.read_csv(exports / "resilience.csv")
    assert resilience.iloc[0].scenario == "baseline"
    assert resilience.iloc[0].fraction_baseline_largest == 1.0


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.remote_assets = []
        self.script_sources = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.script_sources.append(attributes["src"])
        for name in ("src", "href"):
            target = attributes.get(name, "")
            if tag in {"script", "link", "img", "iframe"} and target.startswith(("http:", "https:", "//")):
                self.remote_assets.append(target)


def graph_records(content):
    return json.loads(re.search(r"nodes = new vis.DataSet\((\[.*?\])\);", content, re.S).group(1))


def refresh_manifest(out_dir, source_dir):
    manifest = {"schema_version": 1, "status": "complete", "run_id": "test-run-1"}
    for section, directory, names in (("inputs", source_dir, SOURCE_FILES), ("outputs", out_dir, EXPORT_FILES)):
        manifest[section] = {name: {"sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()} for name in names}
    (out_dir / "run_metadata.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def small_export(tmp_path):
    """An independent exact-ID contract, with one connected seed and one isolate."""
    source, out = tmp_path / "data", tmp_path / "out"
    source.mkdir()
    out.mkdir()
    gids = [2**63 - 1, 2**53 + 1, 7]
    nodes = pd.DataFrame({"gid": gids, "depth": [0, 1, 4], "is_seed": [True, False, False]})
    edges = pd.DataFrame({"src": [gids[0]], "dst": [gids[1]], "sum_kzt": [10000.0], "n_tx": [1]})
    tx = edges.drop(columns="n_tx").assign(date=pd.Timestamp("2026-07-15"))
    for name, frame in (("nodes", nodes), ("edges", edges), ("transactions", tx)):
        frame.to_parquet(source / f"{name}.parquet", index=False)
    roles = pd.DataFrame({"gid": gids, "role": ["distributor", "terminal", "peripheral"], "role_score": [.8, .7, .1],
                          "cluster_id": [0, 0, 1], "priority_score": [.9, .6, .1], "evidence": ["Observed outgoing flow", "Observed incoming flow", "No sampled activity"]})
    roles.to_csv(out / "nodes_roles.csv", index=False)
    features = roles.merge(nodes, on="gid", validate="one_to_one").assign(seed_reach_count=[1, 1, 0],
        peer_anomaly_score=[.1, .2, .0], pagerank=[.2, .7, .1], betweenness=[.0, .0, .0],
        priority_test_contribution=[.9, .6, .1])
    features.to_parquet(out / "node_features.parquet", index=False)
    pd.DataFrame({"cluster_id": [0, 1], "n_nodes": [2, 1], "n_seed": [1, 0], "sum_kzt_internal": [10000, 0],
                  "top_gids": [f"{gids[0]};{gids[1]}", "7"], "hypothesis": ["Observed flow", "Isolated"]}).to_csv(out / "clusters.csv", index=False)
    roles[["gid", "role", "priority_score"]].assign(why=["Ranking one", "Ranking two", "Ranking three"]).assign(rank=[1, 2, 3])[
        ["rank", "gid", "role", "priority_score", "why"]].to_csv(out / "top_nodes.csv", index=False)
    pd.DataFrame({"scenario": ["baseline"], "largest_component_size": [2], "n_components": [2], "fraction_baseline_largest": [1.0]}).to_csv(out / "resilience.csv", index=False)
    refresh_manifest(out, source)
    return out, source, gids


def test_strict_join_preserves_exact_int64_and_authoritative_fields(small_export):
    out, source, gids = small_export
    data = load_data(str(out), str(source))
    merged = merged_nodes(data)
    assert str(merged.gid.dtype) == "int64"
    assert merged.gid.tolist() == gids
    assert merged.seed_reach_count.tolist() == [1, 1, 0]
    assert merged.role.tolist() == data.roles.role.tolist()
    assert merged.priority_score.tolist() == data.roles.priority_score.tolist()
    assert merged.set_index("gid").loc[gids[0], "out_kzt"] == 10000
    assert merged.set_index("gid").loc[7, "in_deg"] == 0


@pytest.mark.parametrize("invalid", [[1.0], [True], ["9007199254740993.0"], [str(2**63)], [None], ["bad"]])
def test_ids_reject_lossy_or_invalid_inputs(invalid):
    with pytest.raises(ValueError, match="int64|integer|identifier"):
        exact_ids(pd.Series(invalid), "test.gid")


def test_feature_join_rejects_duplicate_and_mismatched_ids(small_export):
    out, source, gids = small_export
    path = out / "node_features.parquet"
    features = pd.read_parquet(path)
    features.loc[1, "gid"] = gids[0]
    features.to_parquet(path, index=False)
    refresh_manifest(out, source)
    with pytest.raises(ValueError, match="unique"):
        load_data(str(out), str(source))
    features.loc[1, "gid"] = 12
    features.to_parquet(path, index=False)
    refresh_manifest(out, source)
    with pytest.raises(ValueError, match="match one-to-one"):
        load_data(str(out), str(source))


def test_features_cannot_override_authoritative_submission(small_export):
    out, source, _ = small_export
    path = out / "node_features.parquet"
    features = pd.read_parquet(path)
    features.loc[0, "role"] = "coordinator"
    features.to_parquet(path, index=False)
    refresh_manifest(out, source)
    with pytest.raises(ValueError, match="authoritative submission field role"):
        merged_nodes(load_data(str(out), str(source)))


@pytest.mark.parametrize("section,name", [("out", name) for name in EXPORT_FILES] + [("source", name) for name in SOURCE_FILES])
def test_changed_artifact_invalidates_cache_and_rejects_stale_provenance(small_export, section, name):
    out, source, _ = small_export
    load_data(str(out), str(source))
    path = (out if section == "out" else source) / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_data(str(out), str(source))


def test_same_size_same_timestamp_metadata_change_is_not_cached(small_export):
    out, source, _ = small_export
    before = load_data(str(out), str(source))
    path = out / "run_metadata.json"
    stamp = path.stat()
    path.write_text(path.read_text(encoding="utf-8").replace("test-run-1", "test-run-2"), encoding="utf-8")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    after = load_data(str(out), str(source))
    assert before.metadata["run_id"] == "test-run-1"
    assert after.metadata["run_id"] == "test-run-2"


def test_regenerated_features_refresh_card_without_manual_reload(small_export, monkeypatch):
    out, source, gids = small_export
    monkeypatch.setenv("MONEY_GRAPH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("MONEY_GRAPH_DATA_DIR", str(source))
    ui = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    page(ui, "Node card")
    ui.text_input(key="node_gid_input").set_value(str(gids[0])).run()
    assert {item.label: item.value for item in ui.metric}["Seed reach"] == "1"
    path = out / "node_features.parquet"
    features = pd.read_parquet(path)
    features.loc[features.gid.eq(gids[0]), "seed_reach_count"] = 2
    features.to_parquet(path, index=False)
    refresh_manifest(out, source)
    ui.run()
    assert {item.label: item.value for item in ui.metric}["Seed reach"] == "2"
    assert not ui.error and not ui.exception


def test_missing_and_incomplete_metadata_are_rejected(small_export):
    out, source, _ = small_export
    path = out / "run_metadata.json"
    path.unlink()
    with pytest.raises(ValueError, match="no run_metadata"):
        load_data(str(out), str(source))
    refresh_manifest(out, source)
    path.write_text(path.read_text(encoding="utf-8").replace('"complete"', '"running"'), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported or incomplete"):
        load_data(str(out), str(source))


def test_full_graph_and_isolated_cluster_keep_all_nodes(exports):
    data = load_data(str(exports), str(ROOT / "data"))
    nodes = merged_nodes(data)
    rendered = graph_records(pyvis_html(data.edges, nodes))
    assert len(rendered) == 2248
    assert {item["id"] for item in rendered} == set(nodes.gid.astype(str))
    isolated = nodes.loc[nodes.in_deg.eq(0) & nodes.out_deg.eq(0)]
    assert len(isolated) == 19
    focus = int(isolated.gid.iloc[0])
    edges, selected, truncated = selected_neighborhood(data.edges, nodes, focus, 2)
    assert edges.empty and not truncated
    assert [item["id"] for item in graph_records(pyvis_html(edges, selected, focus))] == [str(focus)]
    members = nodes[nodes.cluster_id.eq(isolated.cluster_id.iloc[0])]
    assert len(members) == 1
    assert len(graph_records(pyvis_html(edges, members))) == 1


def test_large_neighborhood_keeps_focus_after_500_edge_limit():
    nodes = pd.DataFrame({"gid": range(503), "role": "peripheral"})
    edges = pd.DataFrame({"src": [0] + [1] * 501, "dst": [1] + list(range(2, 503)), "sum_kzt": [1] + [10000] * 501})
    visible, selected, truncated = selected_neighborhood(edges, nodes, 0, 2)
    assert truncated and len(visible) == 500
    assert 0 not in set(visible.src) | set(visible.dst)
    assert "0" in {item["id"] for item in graph_records(pyvis_html(visible, selected, 0))}


def test_cluster_coloring_and_highlighting(small_export):
    out, source, gids = small_export
    data = load_data(str(out), str(source))
    nodes = merged_nodes(data)
    rendered = graph_records(pyvis_html(data.edges, nodes, color_by="Cluster"))
    by_id = {item["id"]: item for item in rendered}
    assert by_id[str(gids[0])]["color"]["background"] == by_id[str(gids[1])]["color"]["background"]
    assert by_id[str(gids[0])]["color"]["background"] != by_id["7"]["color"]["background"]
    assert by_id[str(gids[0])]["borderWidth"] == 4
    highlighted = graph_records(pyvis_html(data.edges, nodes, highlight_cluster="1"))
    assert next(item for item in highlighted if item["id"] == str(gids[0]))["color"]["background"] == "#e2e8f0"
    assert next(item for item in highlighted if item["id"] == "7")["color"]["border"] == "#f59e0b"


def test_ui_large_id_boundary_isolate_and_unknown(small_export, monkeypatch):
    out, source, gids = small_export
    monkeypatch.setenv("MONEY_GRAPH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("MONEY_GRAPH_DATA_DIR", str(source))
    ui = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    page(ui, "Network explorer")
    ui.text_input(key="network_gid_input").set_value(str(gids[0])).run()
    assert not ui.error and not ui.exception
    assert str(gids[0]) == ui.text_input(key="network_gid_input").value
    ui.text_input(key="network_gid_input").set_value("7").run()
    assert any("Known isolated gid 7" in item.value for item in ui.info)
    assert any("1 nodes and 0" in item.value for item in ui.caption)
    ui.text_input(key="network_gid_input").set_value("8").run()
    assert any("Unknown gid" in item.value for item in ui.warning)
    page(ui, "Node card")
    ui.text_input(key="node_gid_input").set_value("7").run()
    assert any("beyond hop 4" in item.value for item in ui.warning)
    assert not ui.exception


def test_failed_ai_followup_does_not_display_previous_answer(small_export, monkeypatch):
    from src.ai_assistant import GroundedAnswer
    out, source, gids = small_export
    monkeypatch.setenv("MONEY_GRAPH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("MONEY_GRAPH_DATA_DIR", str(source))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-mocked")

    def answer(question, tools):
        if question == "Second question":
            raise RuntimeError("Mocked unavailable model")
        return GroundedAnswer("Unique first answer", (str(gids[0]),))

    monkeypatch.setattr("src.ai_assistant.answer_question_result", answer)
    ui = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    page(ui, "AI analyst")
    ui.text_area[0].set_value("First question").run()
    button(ui, "Ask grounded assistant").click().run()
    assert any(item.value == "Unique first answer" for item in ui.markdown)
    assert any("Answer to: First question" in item.value for item in ui.caption)
    ui.text_area[0].set_value("Second question").run()
    button(ui, "Ask grounded assistant").click().run()
    assert ui.error and not ui.exception
    assert not any(item.value == "Unique first answer" for item in ui.markdown)
    assert not any(item.label.startswith("Open gid ") for item in ui.button)


def test_ai_navigation_and_regeneration_invalidate_saved_answer(small_export, monkeypatch):
    from src.ai_assistant import GroundedAnswer
    out, source, gids = small_export
    monkeypatch.setenv("MONEY_GRAPH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("MONEY_GRAPH_DATA_DIR", str(source))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-mocked")
    monkeypatch.setattr("src.ai_assistant.answer_question_result", lambda *args: GroundedAnswer("Grounded original run", (str(gids[0]), "123")))
    ui = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    page(ui, "AI analyst")
    ui.text_area[0].set_value("Explain").run()
    button(ui, "Ask grounded assistant").click().run()
    assert not any(item.label == "Open gid 123" for item in ui.button)
    button(ui, f"Open gid {gids[0]}").click().run()
    assert ui.sidebar.radio[0].value == "Node card"
    assert ui.text_input(key="node_gid_input").value == str(gids[0])
    page(ui, "AI analyst")
    assert any(item.value == "Grounded original run" for item in ui.markdown)
    path = out / "node_features.parquet"
    features = pd.read_parquet(path)
    features.loc[0, "seed_reach_count"] = 2
    features.to_parquet(path, index=False)
    refresh_manifest(out, source)  # Deliberately preserves the same run_id.
    ui.run()
    assert not any(item.value == "Grounded original run" for item in ui.markdown)
    assert not ui.error and not ui.exception
