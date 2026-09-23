"""Render every workspace and exercise navigation against real pipeline exports."""
from pathlib import Path
import hashlib
import json
import os
import re
from io import BytesIO
from types import SimpleNamespace
from html.parser import HTMLParser

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import EXPORT_FILES, NODE_COLUMNS, SOURCE_FILES, activate_uploaded_case, daily_activity, evidence_records, exact_ids, load_data, merged_nodes, observed_ratio, percentile, pyvis_html, queue_view, selected_neighborhood
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
    selected_cluster = str(expected.cluster_id.iloc[0])
    ui.multiselect[1].set_value([selected_cluster]).run()
    assert len(ui.dataframe[0].value) == expected.cluster_id.astype(str).eq(selected_cluster).sum()
    ui.multiselect[1].set_value([]).run()
    ui.multiselect[2].set_value([4]).run()
    assert len(ui.dataframe[0].value) == pd.read_parquet(ROOT / "data" / "nodes.parquet").depth.eq(4).sum()
    ui.multiselect[2].set_value([]).run()
    next(x for x in ui.selectbox if x.label == "Seed status").set_value("Seed").run()
    assert len(ui.dataframe[0].value) == 81
    next(x for x in ui.selectbox if x.label == "Seed status").set_value("Non-seed").run()
    assert len(ui.dataframe[0].value) == 2248 - 81
    ui.number_input[0].set_value(0.5).run()
    source_nodes = pd.read_parquet(ROOT / "data" / "nodes.parquet")
    seed_ids = source_nodes.loc[source_nodes.is_seed, "gid"]
    assert len(ui.dataframe[0].value) == (expected.priority_score.ge(0.5) & ~expected.gid.isin(seed_ids)).sum()
    ui.number_input[0].set_value(1.0).run()
    assert ui.dataframe[0].value.empty
    assert not any(item.label == "Open node card" for item in ui.button)
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


def test_regenerated_features_reset_selection_and_refresh_card_without_manual_reload(small_export, monkeypatch):
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
    # A different analytical snapshot must not retain the former case selection.
    assert ui.text_input(key="node_gid_input").value == str(min(gids))
    ui.text_input(key="node_gid_input").set_value(str(gids[0])).run()
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


def test_card_shows_exported_winning_rule_and_priority_explanation(ui, exports):
    frame = pd.read_parquet(exports / "node_features.parquet")
    candidate = frame.sort_values("priority_score", ascending=False).iloc[0]
    page(ui, "Node card")
    ui.text_input(key="node_gid_input").set_value(str(candidate.gid)).run()
    shown = [item.value for item in ui.markdown]
    assert candidate.role_rule in shown
    assert candidate.role_rule_details in shown
    assert candidate.priority_explanation in shown
    assert any("Score contribution" in item.value for item in ui.dataframe)
    candidates = next(item.value for item in ui.dataframe if "Candidate" in item.value)
    assert set(candidates.Candidate) == {"coordinator", "consolidator", "distributor", "transit", "terminal"}
    assert "Gate comparison" in candidates and "Available weight" in candidates
    assert not ui.error and not ui.exception


def test_ratio_availability_distinguishes_censored_from_observed_zero():
    unavailable = pd.Series({"relay_2d_ratio": 0.0, "relay_2d_valid": False, "relay_2d_invalid_reason": "no_complete_followup_window"})
    assert "no incoming date has a complete" in observed_ratio(unavailable, "relay_2d_ratio", "relay_2d_valid", "relay_2d_invalid_reason")
    available = unavailable.copy()
    available["relay_2d_valid"] = True
    assert observed_ratio(available, "relay_2d_ratio", "relay_2d_valid", "relay_2d_invalid_reason") == "0.000"


def test_optional_evidence_serializes_exact_nested_gids():
    large = 2**63 - 1
    source = json.dumps([{"src": large, "via": str(large - 1), "dst": "7", "recipient_gids": [large, "7"],
                          "support_days": 2, "date_pairs": [["2026-07-01", "2026-07-03"]], "in_kzt_on_support_dates": 12345.67}])
    parsed = evidence_records(source)
    assert parsed[0]["src"] == str(large)
    assert parsed[0]["via"] == str(large - 1)
    assert parsed[0]["recipient_gids"] == [str(large), "7"]
    assert parsed[0]["in_kzt_on_support_dates"] == 12345.67
    assert parsed[0]["date_pairs"] == [["2026-07-01", "2026-07-03"]]
    with pytest.raises(ValueError, match="floating-point"):
        evidence_records('[{"src": 9007199254740994.0}]')


def test_card_displays_route_burst_amount_and_peer_evidence(ui, exports):
    frame = pd.read_parquet(exports / "node_features.parquet")
    candidate = frame.loc[frame.repeated_route_count.gt(0)].iloc[0]
    page(ui, "Node card")
    ui.text_input(key="node_gid_input").set_value(str(candidate.gid)).run()
    markdown = "\n".join(item.value for item in ui.markdown)
    assert "role strength" in markdown.lower() or candidate.role_rule in markdown
    assert "eligible incoming dates" in markdown
    assert "partial-follow-up dates excluded" in markdown
    assert str(candidate.max_in_senders_date.date()) in markdown
    route = json.loads(candidate.repeated_route_evidence)[0]
    records = [json.loads(item.value) for item in ui.json]
    assert any(isinstance(record, list) and route in record for record in records)
    assert isinstance(route["src"], str) and isinstance(route["via"], str) and isinstance(route["dst"], str)
    anomaly = next(item.value for item in ui.dataframe if "Deviation component" in item.value)
    assert len(anomaly) == 6
    assert not ui.error and not ui.exception


def test_censored_card_and_truncated_search_explain_limits(ui, exports):
    frame = pd.read_parquet(exports / "node_features.parquet")
    candidate = frame.loc[frame.repeated_route_truncated].iloc[0]
    page(ui, "Node card")
    ui.text_input(key="node_gid_input").set_value(str(candidate.gid)).run()
    assert any("lower bounds" in item.value for item in ui.warning)
    boundary = frame.loc[frame.depth.eq(4)].iloc[0]
    ui.text_input(key="node_gid_input").set_value(str(boundary.gid)).run()
    shown = {item.label: item.value for item in ui.metric}
    assert "Unavailable" in shown["Temporal relay"]
    assert "beyond hop 4" in shown["Temporal relay"]
    assert boundary.role_rule in [item.value for item in ui.markdown]
    assert not ui.error and not ui.exception


def test_daily_activity_normalizes_mixed_dates_and_offsets_to_utc_midnight():
    gid = 2**63 - 1
    tx = pd.DataFrame({"src": [gid, gid, 3, 4], "dst": [2, 2, gid, gid],
        "date": ["2026-07-01", "2026-07-01T23:30:00-02:00", "2026-07-02T01:00:00+03:00", pd.Timestamp("2026-07-02 08:30")],
        "sum_kzt": [5000., 7000., 11000., 13000.]})
    original = tx.copy(deep=True)
    daily = daily_activity(tx, gid)
    assert daily.index.tolist() == [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-02")]
    assert daily["Incoming KZT"].tolist() == [11000., 13000.]
    assert daily["Outgoing KZT"].tolist() == [5000., 7000.]
    assert daily.index.tz is None
    pd.testing.assert_frame_equal(tx, original)


def test_card_renders_mixed_date_format_input(small_export, monkeypatch):
    out, source, gids = small_export
    tx = pd.read_parquet(source / "transactions.parquet")
    tx = pd.concat([tx] * 3, ignore_index=True)
    tx["date"] = ["2026-07-01", "2026-07-01T23:30:00-02:00", "2026-07-02T01:00:00+03:00"]
    tx.to_parquet(source / "transactions.parquet", index=False)
    edges = pd.read_parquet(source / "edges.parquet")
    edges["sum_kzt"], edges["n_tx"] = 30000., 3
    edges.to_parquet(source / "edges.parquet", index=False)
    refresh_manifest(out, source)
    monkeypatch.setenv("MONEY_GRAPH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("MONEY_GRAPH_DATA_DIR", str(source))
    ui = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    page(ui, "Node card")
    ui.text_input(key="node_gid_input").set_value(str(gids[0])).run()
    assert not ui.error and not ui.exception
    assert len(ui.get("vega_lite_chart")) == 1


@pytest.mark.parametrize("values,gid,expected", [
    ([4.0], 0, "100.0th percentile"),
    ([5.0, 5.0, 5.0], 1, "66.7th percentile"),
    ([float("inf"), None, 3.0], 2, "100.0th percentile"),
    ([float("inf"), None, 3.0], 0, "Not exported"),
    ([3.0], 8, "Not exported"),
])
def test_ui_percentile_uses_shared_tie_missing_and_singleton_policy(values, gid, expected):
    frame = pd.DataFrame({"gid": range(len(values)), "metric": values})
    assert percentile(frame, "metric", gid) == expected


def test_queue_turnover_uses_total_activity_when_available():
    frame = pd.DataFrame({"gid": [2**63 - 1], "priority_score": [.5], "in_kzt": [10.0], "total_kzt": [30.0]})
    assert queue_view(frame).iloc[0].turnover_kzt == 30.0


def test_case_controls_and_submission_download_are_available(ui):
    assert [item.proto.label for item in ui.get("file_uploader")] == [
        "Nodes Parquet", "Edges Parquet", "Transactions Parquet",
    ]
    assert button(ui, "Validate and run uploaded case").disabled
    downloads = {item.proto.label: item.proto for item in ui.get("download_button")}
    assert "Download submission bundle" in downloads
    assert not downloads["Download submission bundle"].disabled


def test_submission_bundle_cannot_use_a_new_run_under_an_older_displayed_snapshot(small_export):
    import zipfile
    import app

    out, source, _ = small_export
    displayed = app.load_data(str(out), str(source))
    newer = {**displayed.metadata, "run_id": "published-after-page-load"}
    (out / "run_metadata.json").write_text(json.dumps(newer))
    app.submission_bundle.clear()
    with pytest.raises(ValueError, match="changed since the displayed case"):
        app.submission_bundle(str(out), str(source), displayed.metadata)
    payload = app.submission_bundle(str(out), str(source), newer)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        assert json.loads(archive.read("release_metadata.json"))["run_id"] == newer["run_id"]


def test_cached_submission_bundle_remains_bound_to_its_original_displayed_snapshot(small_export):
    import zipfile
    import app

    out, source, _ = small_export
    displayed = app.load_data(str(out), str(source))
    original = app.submission_bundle(str(out), str(source), displayed.metadata)
    newer = {**displayed.metadata, "run_id": "published-after-download-cache"}
    (out / "run_metadata.json").write_text(json.dumps(newer))
    assert app.submission_bundle(str(out), str(source), displayed.metadata) == original
    for expected in (displayed.metadata, newer):
        payload = app.submission_bundle(str(out), str(source), expected)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            assert json.loads(archive.read("release_metadata.json"))["run_id"] == expected["run_id"]


def test_review_shortlist_keeps_exact_ids_across_pages_and_filters(small_export, monkeypatch):
    from src.workspace import build_review_csv
    out, source, gids = small_export
    monkeypatch.setenv("MONEY_GRAPH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("MONEY_GRAPH_DATA_DIR", str(source))
    payloads = []

    def capture(nodes, selected, metadata):
        result = build_review_csv(nodes, selected, metadata)
        payloads.append(result)
        return result

    monkeypatch.setattr("src.workspace.build_review_csv", capture)
    ui = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    page(ui, "Investigation queue")
    selected = [str(gids[0]), str(gids[2])]
    ui.multiselect(key="review_shortlist_selection").set_value(selected).run()
    assert ui.session_state["review_shortlist"] == selected
    parsed = pd.read_csv(BytesIO(payloads[-1]), dtype={"gid": "string"})
    assert parsed.gid.tolist() == selected
    assert parsed.run_id.eq("test-run-1").all()
    assert {"why", "limitations", "next_request"}.issubset(parsed.columns)
    downloads = {item.proto.label: item.proto for item in ui.get("download_button")}
    assert not downloads["Download review shortlist"].disabled
    ui.multiselect(key="queue_roles").set_value(["terminal"]).run()
    assert ui.multiselect(key="review_shortlist_selection").value == selected
    assert len(ui.dataframe[0].value) == 1
    page(ui, "Overview")
    page(ui, "Investigation queue")
    assert ui.multiselect(key="review_shortlist_selection").value == selected
    assert not ui.exception and not ui.error


def test_review_shortlist_and_context_reset_on_regeneration(small_export, monkeypatch):
    out, source, gids = small_export
    monkeypatch.setenv("MONEY_GRAPH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("MONEY_GRAPH_DATA_DIR", str(source))
    ui = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    page(ui, "Investigation queue")
    ui.multiselect(key="review_shortlist_selection").set_value([str(gids[0])]).run()
    ui.multiselect(key="queue_roles").set_value(["terminal"]).run()
    context = ui.session_state["active_run_context"]
    features = pd.read_parquet(out / "node_features.parquet")
    features.loc[0, "seed_reach_count"] = 3
    features.to_parquet(out / "node_features.parquet", index=False)
    refresh_manifest(out, source)  # Byte fingerprints change even with the same run id.
    ui.run()
    assert ui.session_state["active_run_context"] != context
    assert ui.session_state["review_shortlist"] == []
    assert ui.multiselect(key="review_shortlist_selection").value == []
    assert ui.multiselect(key="queue_roles").value == []
    downloads = {item.proto.label: item.proto for item in ui.get("download_button")}
    assert downloads["Download review shortlist"].disabled
    assert not ui.exception and not ui.error


def test_uploaded_case_switch_failure_and_reset_preserve_configured_files(small_export, monkeypatch, tmp_path):
    import shutil
    out, source, gids = small_export
    monkeypatch.setenv("MONEY_GRAPH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("MONEY_GRAPH_DATA_DIR", str(source))
    before = {str(path): path.read_bytes() for parent in (out, source) for path in parent.iterdir()}
    case_root = tmp_path / "uploaded-case"
    case_source, case_out = case_root / "data", case_root / "out"
    shutil.copytree(source, case_source)
    shutil.copytree(out, case_out)
    metadata = json.loads((case_out / "run_metadata.json").read_text())
    metadata["run_id"] = "uploaded-case-run"
    (case_out / "run_metadata.json").write_text(json.dumps(metadata))
    cleaned = []

    def cleanup():
        cleaned.append(True)
        shutil.rmtree(case_root)

    candidate = SimpleNamespace(data_dir=case_source, out_dir=case_out, metadata=metadata, cleanup=cleanup)
    uploaded_files = {}
    monkeypatch.setattr("streamlit.file_uploader", lambda label, **kwargs: uploaded_files.get(label))
    calls = []

    def run_uploaded(files):
        calls.append(files)
        return candidate

    monkeypatch.setattr("src.workspace.run_uploaded_case", run_uploaded)
    ui = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    page(ui, "Investigation queue")
    ui.multiselect(key="review_shortlist_selection").set_value([str(gids[0])]).run()
    for label, filename in (("Nodes Parquet", "nodes.parquet"), ("Edges Parquet", "edges.parquet"), ("Transactions Parquet", "transactions.parquet")):
        uploaded_files[label] = BytesIO((source / filename).read_bytes())
    ui.run()
    assert not button(ui, "Validate and run uploaded case").disabled
    button(ui, "Validate and run uploaded case").click().run()
    assert len(calls) == 1 and set(calls[0]) == set(SOURCE_FILES)
    assert ui.session_state["uploaded_case"].out_dir == case_out
    assert ui.session_state["review_shortlist"] == []
    assert any("uploaded-case-run" in item.value for item in ui.sidebar.caption)
    assert not ui.error and not ui.exception
    assert ui.sidebar.text_input[0].value == str(out)
    assert ui.sidebar.text_input[1].value == str(source)
    context = ui.session_state["active_run_context"]

    def invalid_upload(files):
        raise ValueError("Deliberately invalid upload")

    monkeypatch.setattr("src.workspace.run_uploaded_case", invalid_upload)
    button(ui, "Validate and run uploaded case").click().run()
    assert any("Deliberately invalid upload" in item.value for item in ui.error)
    assert ui.session_state["uploaded_case"].out_dir == case_out
    assert ui.session_state["active_run_context"] == context
    assert not cleaned and not ui.exception
    button(ui, "Use configured dataset").click().run()
    assert "uploaded_case" not in ui.session_state
    assert cleaned == [True] and not case_root.exists()
    assert any("test-run-1" in item.value for item in ui.sidebar.caption)
    assert all(Path(path).read_bytes() == content for path, content in before.items())
    assert not ui.error and not ui.exception


def test_candidate_display_failure_cleans_only_failed_case(monkeypatch):
    import app
    calls = []
    previous = SimpleNamespace(cleanup=lambda: calls.append("previous"))
    candidate = SimpleNamespace(out_dir=Path("new-out"), data_dir=Path("new-data"), cleanup=lambda: calls.append("candidate"))
    state = {"uploaded_case": previous}
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr("src.workspace.run_uploaded_case", lambda files: candidate)

    def cannot_load(*args):
        raise ValueError("Candidate data cannot be displayed")

    monkeypatch.setattr(app, "load_data", cannot_load)
    with pytest.raises(ValueError, match="cannot be displayed"):
        activate_uploaded_case({})
    assert state["uploaded_case"] is previous
    assert calls == ["candidate"]


def test_successful_upload_replacement_cleans_previous_case_after_switch(monkeypatch):
    import app
    state, calls = {}, []
    candidate = SimpleNamespace(out_dir=Path("new-out"), data_dir=Path("new-data"), cleanup=lambda: calls.append("candidate"))

    def cleanup_previous():
        assert state["uploaded_case"] is candidate
        calls.append("previous")

    state["uploaded_case"] = SimpleNamespace(cleanup=cleanup_previous)
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr("src.workspace.run_uploaded_case", lambda files: candidate)
    monkeypatch.setattr(app, "load_data", lambda *args: object())
    monkeypatch.setattr(app, "merged_nodes", lambda data: pd.DataFrame())
    activate_uploaded_case({})
    assert state["uploaded_case"] is candidate
    assert calls == ["previous"]
