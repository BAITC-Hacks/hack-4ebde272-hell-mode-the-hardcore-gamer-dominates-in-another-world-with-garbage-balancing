"""Decision evidence, reproducible graph results and strict export regressions."""
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.clustering import cluster_graph
from src.explanations import add_evidence, evidence_for
from src.exports import CLUSTER_COLUMNS, NODE_COLUMNS, TOP_COLUMNS, cluster_summaries, write_exports
from src.priority import WEIGHTS
from src.resilience import resilience_analysis
from validate_submission import SubmissionValidationError, validate_submission


def test_supplied_dataset_clusters_repeat_after_equivalent_row_permutations():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    edges = pd.read_parquet(data_dir / "edges.parquet")
    baseline = None
    for permutation in (None, 7, 42):
        current_nodes = nodes if permutation is None else nodes.sample(frac=1, random_state=permutation)
        current_edges = edges if permutation is None else edges.sample(frac=1, random_state=permutation)
        graph = nx.from_pandas_edgelist(current_edges, "src", "dst", edge_attr="sum_kzt", create_using=nx.DiGraph)
        graph.add_nodes_from(current_nodes.gid)
        assignment, _ = cluster_graph(graph, current_nodes)
        assignment = assignment.sort_values("gid").reset_index(drop=True)
        if baseline is None:
            baseline = assignment
        else:
            pd.testing.assert_frame_equal(baseline, assignment)


def export_fixture(tmp_path, count=60):
    # Above 2**53: a float conversion would silently corrupt adjacent gids.
    gids = np.arange(2**53 + 1, 2**53 + count + 1, dtype=np.int64)
    frame = pd.DataFrame({"gid": gids, "role": "peripheral", "role_score": 0.4,
                          "cluster_id": np.arange(count) // (count // 2),
                          "priority_score": 0.1 + np.arange(count, 0, -1) / count * 0.2,
                          "evidence": "0 incoming and 0 outgoing counterparties observed",
                          "depth": 1, "is_seed": False, "out_deg": 0,
                          "in_deg": 0, "internal_out_kzt": 0.0,
                          "relay_2d_ratio": np.nan, "boundary_censored": False})
    frame.loc[0, ["is_seed", "depth"]] = [True, 0]
    frame["priority_explanation"] = frame.priority_score.map(lambda value: f"Review priority {value:.3f}: rule strength adds 0.100; seed reach contributes the remainder.")
    for name in WEIGHTS:
        frame[f"priority_{name}_contribution"] = 0.0
    frame["priority_role_strength_contribution"] = 0.1
    frame["priority_seed_reach_count_contribution"] = frame.priority_score - 0.1
    graph = nx.DiGraph()
    graph.add_nodes_from(gids.tolist())
    graph.add_edge(int(gids[0]), int(gids[1]), sum_kzt=25.5)
    frame.loc[0, "internal_out_kzt"] = 25.5
    frame.loc[0, "out_deg"] = 1
    frame.loc[1, "in_deg"] = 1
    sources = frame[["gid", "is_seed", "depth"]].copy()
    clusters = cluster_summaries(frame, graph, sources)
    write_exports(frame, clusters, tmp_path)
    return frame, clusters, sources, graph


def rewrite_csv(tmp_path, name, change):
    frame = pd.read_csv(tmp_path / name, dtype=str, keep_default_na=False)
    changed = change(frame)
    (frame if changed is None else changed).to_csv(tmp_path / name, index=False)


def test_export_schema_and_submission_validation(tmp_path):
    frame, clusters, sources, graph = export_fixture(tmp_path, 2248)
    for name, columns in (("nodes_roles.csv", NODE_COLUMNS), ("clusters.csv", CLUSTER_COLUMNS),
                          ("top_nodes.csv", TOP_COLUMNS)):
        assert pd.read_csv(tmp_path / name).columns.tolist() == columns
    auxiliary = pd.read_parquet(tmp_path / "node_features.parquet")
    assert auxiliary.gid.dtype == np.dtype("int64")
    pd.testing.assert_frame_equal(auxiliary, frame)
    top = pd.read_csv(tmp_path / "top_nodes.csv")
    assert len(top) == 50
    assert not top.why.eq(frame.evidence.head(50)).any()
    assert validate_submission(tmp_path, source_nodes=sources, features=frame, graph=graph)


@pytest.mark.parametrize("filename", ["nodes_roles.csv", "clusters.csv", "top_nodes.csv"])
@pytest.mark.parametrize("mutation", ["extra", "missing", "reordered"])
def test_exact_schema_rejects_changes(tmp_path, filename, mutation):
    export_fixture(tmp_path)
    def change(frame):
        if mutation == "extra":
            frame["unexpected"] = "1"
            return frame
        if mutation == "missing":
            return frame.iloc[:, :-1]
        return frame[frame.columns[::-1]]
    rewrite_csv(tmp_path, filename, change)
    with pytest.raises(SubmissionValidationError, match="columns must be exactly"):
        validate_submission(tmp_path, expected_nodes=60)


@pytest.mark.parametrize(("column", "value", "message"), [
    ("gid", "1.5", "int64"), ("gid", "True", "int64"), ("gid", str(2**63), "int64"),
    ("role", "criminal", "invalid role"), ("role_score", "NaN", "finite"),
    ("priority_score", "inf", "finite"), ("role_score", "-0.1", "nonnegative"),
    ("priority_score", "1.01", "\\[0,1\\]"), ("cluster_id", "0.0", "int64"),
    ("evidence", " ", "nonempty"), ("evidence", "No numeric evidence", "numeric"),
    ("evidence", "1" * 201, "200"),
])
def test_malformed_node_fields_fail(tmp_path, column, value, message):
    export_fixture(tmp_path)
    rewrite_csv(tmp_path, "nodes_roles.csv", lambda frame: frame.__setitem__(column, frame[column].mask(frame.index == 0, value)))
    with pytest.raises(SubmissionValidationError, match=message):
        validate_submission(tmp_path, expected_nodes=60)


@pytest.mark.parametrize(("column", "value", "message"), [
    ("cluster_id", "99", "cluster references"), ("n_nodes", "29", "counts must sum"),
    ("n_seed", "0", "n_seed"), ("sum_kzt_internal", "0", "turnover"),
    ("top_gids", "42", "top_gids"), ("n_nodes", "30.0", "int64"),
    ("hypothesis", " ", "nonempty"), ("sum_kzt_internal", "inf", "finite"),
])
def test_cluster_reconciliation_rejects_malformed_rows(tmp_path, column, value, message):
    export_fixture(tmp_path)
    rewrite_csv(tmp_path, "clusters.csv", lambda frame: frame.__setitem__(column, frame[column].mask(frame.index == 0, value)))
    with pytest.raises(SubmissionValidationError, match=message):
        validate_submission(tmp_path, expected_nodes=60)


def test_population_distribution_checked_even_when_total_matches(tmp_path):
    export_fixture(tmp_path)
    rewrite_csv(tmp_path, "clusters.csv", lambda frame: frame.assign(n_nodes=[29, 31]))
    with pytest.raises(SubmissionValidationError, match="does not match members"):
        validate_submission(tmp_path, expected_nodes=60)


@pytest.mark.parametrize(("column", "value", "message"), [
    ("rank", "2", "contiguous"), ("gid", "42", "unknown gids"),
    ("role", "consolidator", "top.role"), ("priority_score", "0.99", "top.priority_score"),
    ("why", " ", "nonempty"), ("why", "Review 1: fabricated reason", "independent priority"),
])
def test_top_reconciliation_rejects_malformed_rows(tmp_path, column, value, message):
    export_fixture(tmp_path)
    rewrite_csv(tmp_path, "top_nodes.csv", lambda frame: frame.__setitem__(column, frame[column].mask(frame.index == 0, value)))
    with pytest.raises(SubmissionValidationError, match=message):
        validate_submission(tmp_path, expected_nodes=60)


def test_top_ties_use_numeric_gid_order_and_no_fifty_maximum(tmp_path):
    frame, clusters, sources, graph = export_fixture(tmp_path)
    frame["priority_score"] = 0.2
    frame["priority_seed_reach_count_contribution"] = 0.1
    clusters = cluster_summaries(frame, graph, sources)
    write_exports(frame.sample(frac=1, random_state=7), clusters, tmp_path)
    top = frame.sort_values("gid").rename(columns={"priority_explanation": "why"})
    top["rank"] = range(1, len(top) + 1)
    top[TOP_COLUMNS].to_csv(tmp_path / "top_nodes.csv", index=False)
    assert validate_submission(tmp_path, expected_nodes=60)
    rewrite_csv(tmp_path, "top_nodes.csv", lambda rows: rows.iloc[[1, 0, *range(2, len(rows))]].assign(rank=range(1, len(rows) + 1)))
    with pytest.raises(SubmissionValidationError, match="descending priority then ascending gid"):
        validate_submission(tmp_path, expected_nodes=60)


@pytest.mark.parametrize("filename", ["nodes_roles.csv", "clusters.csv", "top_nodes.csv"])
def test_duplicate_identifiers_fail(tmp_path, filename):
    export_fixture(tmp_path)
    column = "cluster_id" if filename == "clusters.csv" else "gid"
    def change(frame):
        frame.loc[1, column] = frame.loc[0, column]
    rewrite_csv(tmp_path, filename, change)
    with pytest.raises(SubmissionValidationError, match="unique|duplicate"):
        validate_submission(tmp_path, expected_nodes=60)


def test_top_requires_twenty_rows(tmp_path):
    export_fixture(tmp_path)
    rewrite_csv(tmp_path, "top_nodes.csv", lambda frame: frame.head(19))
    with pytest.raises(SubmissionValidationError, match="at least 20"):
        validate_submission(tmp_path, expected_nodes=60)


@pytest.mark.parametrize("excluded", ["seed", "boundary"])
def test_terminal_exclusions_checked_from_auxiliary(tmp_path, excluded):
    frame, clusters, sources, graph = export_fixture(tmp_path)
    frame.loc[59, "role"] = "terminal"
    frame.loc[59, "is_seed"] = excluded == "seed"
    frame.loc[59, "depth"] = 4 if excluded == "boundary" else 0
    write_exports(frame, clusters, tmp_path)
    with pytest.raises(SubmissionValidationError, match="classified terminal"):
        validate_submission(tmp_path, expected_nodes=60)


def test_source_node_equality_and_turnover_reconciliation(tmp_path):
    frame, clusters, sources, graph = export_fixture(tmp_path)
    wrong = sources.copy()
    wrong.loc[0, "gid"] = 123
    with pytest.raises(SubmissionValidationError, match="gids do not match"):
        validate_submission(tmp_path, expected_nodes=60, source_nodes=wrong)
    wrong = sources.copy()
    wrong.loc[0, "is_seed"] = False
    with pytest.raises(SubmissionValidationError, match="is_seed disagrees"):
        validate_submission(tmp_path, expected_nodes=60, source_nodes=wrong)
    frame.loc[0, "internal_out_kzt"] = 999
    clusters.loc[0, "sum_kzt_internal"] = 999
    write_exports(frame, clusters, tmp_path)
    assert validate_submission(tmp_path, expected_nodes=60)
    with pytest.raises(SubmissionValidationError, match="disagrees with source edges"):
        validate_submission(tmp_path, expected_nodes=60, source_nodes=sources, graph=graph)


@pytest.mark.parametrize("mutation", ["float_gid", "duplicate_gid", "wrong_evidence", "contribution_sum"])
def test_auxiliary_integrity(tmp_path, mutation):
    frame, _, _, _ = export_fixture(tmp_path)
    if mutation == "float_gid":
        frame["gid"] = frame.gid.astype(float)
    elif mutation == "duplicate_gid":
        frame.loc[1, "gid"] = frame.loc[0, "gid"]
    elif mutation == "wrong_evidence":
        frame.loc[0, "evidence"] = "1 different explanation"
    else:
        frame.loc[0, "priority_role_strength_contribution"] += 0.01
    frame.to_parquet(tmp_path / "node_features.parquet", index=False)
    with pytest.raises(SubmissionValidationError, match="int64|unique|disagrees|do not sum"):
        validate_submission(tmp_path, expected_nodes=60)


def test_validation_is_active_under_python_optimization(tmp_path):
    export_fixture(tmp_path)
    rewrite_csv(tmp_path, "top_nodes.csv", lambda frame: frame.head(19))
    result = subprocess.run([sys.executable, "-O", "-c",
                             "from validate_submission import validate_submission; "
                             f"validate_submission({str(tmp_path)!r}, expected_nodes=60)"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "SubmissionValidationError" in result.stderr
    assert "at least 20" in result.stderr


@pytest.mark.parametrize(("role", "reason"), [
    ("consolidator", "Collection"), ("distributor", "Distribution"), ("transit", "Relay"),
    ("coordinator", "Coordination"), ("terminal", "Observed endpoint"), ("peripheral", "Peripheral"),
])
def test_role_evidence_readable_and_numeric(role, reason):
    row = pd.Series({"role": role, "depth": 2, "is_seed": False, "in_deg": 11, "out_deg": 4,
                     "in_kzt": 8.2e6, "out_kzt": 7.1e6, "out_tx": 12, "seed_reach_count": 14,
                     "cross_cluster_out_deg": 2, "cross_cluster_degree": 5,
                     "pass_through": 0.86, "relay_2d_ratio": 0.75,
                     "decision_betweenness_percentile": .99, "decision_pagerank_percentile": .98})
    text = evidence_for(row)
    assert reason in text
    assert 0 < len(text) <= 200
    assert any(character.isdigit() for character in text)
    assert "in_deg" not in text and "seed_reach" not in text


def test_boundary_consolidator_keeps_role_reason():
    text = evidence_for(pd.Series({"role": "consolidator", "depth": 4, "in_deg": 11,
                                   "out_deg": 0, "in_kzt": 8.2e6, "seed_reach_count": 14}))
    assert "Collection pattern" in text and "11" in text and "14" in text
    assert "Beyond hop 4: activity unobserved" in text
    assert len(text) <= 200


def test_empty_isolated_and_missing_peripheral_evidence():
    empty = add_evidence(pd.DataFrame(columns=["gid", "role"]))
    assert empty.empty and "evidence" in empty
    isolated = evidence_for(pd.Series({"role": "peripheral", "is_seed": True, "in_deg": 0, "out_deg": 0}))
    assert "0 incoming and 0 outgoing" in isolated and "Seed inflows incomplete" in isolated
    unavailable = evidence_for(pd.Series({"role": "peripheral"}))
    assert "unavailable" in unavailable and "0.55" in unavailable


@pytest.mark.parametrize("validity_flag", ["relay_2d_valid", "relay_2d_ratio_valid", "relay_2d_ratio_available", "decision_relay_2d_ratio_valid"])
@pytest.mark.parametrize("invalid", [False, np.nan, pd.NA])
def test_transit_explanation_ignores_invalid_raw_temporal_ratio(validity_flag, invalid):
    row = pd.Series({"role": "transit", "depth": 2, "in_deg": 3, "out_deg": 4,
                     "pass_through": .8, "relay_2d_ratio": .95, validity_flag: invalid})
    text = evidence_for(row)
    assert "2-day timing unavailable" in text
    assert "95%" not in text
    assert "0.80" in text


def test_transit_explanation_uses_sanitized_decision_ratios():
    row = pd.Series({"role": "transit", "depth": 2, "in_deg": 3, "out_deg": 4,
                     "pass_through": .8, "relay_2d_ratio": .95,
                     "decision_pass_through": .7, "decision_relay_2d_ratio": .6})
    text = evidence_for(row)
    assert "0.70" in text and "60%" in text
    assert "0.80" not in text and "95%" not in text


def test_clustering_repeated_runs_and_equivalent_row_permutations():
    graph = nx.gnp_random_graph(30, .2, seed=7, directed=True)
    for index, (source, target) in enumerate(sorted(graph.edges)):
        graph[source][target]["sum_kzt"] = (index % 17 + 1) * 123.25
    isolated_gid = 2**53 + 1
    nodes = pd.DataFrame({"gid": [*range(30), isolated_gid],
                          "is_seed": [gid in {0, 8} for gid in range(30)] + [True]})
    shuffled = nx.DiGraph()
    shuffled.add_nodes_from(reversed(list(graph.nodes)))
    edges = list(graph.edges(data=True))
    for index in np.random.default_rng(7).permutation(len(edges)):
        source, target, data = edges[index]
        shuffled.add_edge(source, target, **data)
    baseline, baseline_metadata = cluster_graph(graph, nodes)
    original_edges = list(graph.edges(data=True))
    for candidate_graph, candidate_nodes in ((graph, nodes),
                                            (shuffled, nodes.sample(frac=1, random_state=7)),
                                            (shuffled, nodes.sample(frac=1, random_state=9))):
        actual, metadata = cluster_graph(candidate_graph, candidate_nodes)
        pd.testing.assert_frame_equal(baseline.sort_values("gid").reset_index(drop=True),
                                      actual.sort_values("gid").reset_index(drop=True))
        pd.testing.assert_frame_equal(baseline_metadata, metadata)
    assert graph.is_directed() and list(graph.edges(data=True)) == original_edges
    assert isolated_gid not in graph and isolated_gid in baseline.gid.values
    assert baseline.gid.dtype == np.dtype("int64")
    assert len(baseline) == len(nodes)


def test_louvain_projection_aggregates_reciprocals_and_uses_fixed_settings(monkeypatch):
    graph = nx.DiGraph()
    graph.add_edge(2, 10, sum_kzt=1.25)
    graph.add_edge(10, 2, sum_kzt=2.75)
    graph.add_edge(2, 2, sum_kzt=.5)
    nodes = pd.DataFrame({"gid": [10, 2, 99], "is_seed": [False, False, True]})
    real_louvain = nx.community.louvain_communities
    captured = {}
    def capture(projection, **kwargs):
        captured["projection"] = projection.copy()
        captured["settings"] = kwargs
        return real_louvain(projection, **kwargs)
    monkeypatch.setattr(nx.community, "louvain_communities", capture)
    features, metadata = cluster_graph(graph, nodes)
    projection = captured["projection"]
    assert not projection.is_directed()
    assert list(projection) == [2, 10, 99]
    assert projection[2][10]["sum_kzt"] == 4
    assert projection[2][2]["sum_kzt"] == .5
    assert captured["settings"] == {"weight": "sum_kzt", "resolution": 1.0, "seed": 42}
    assert features.internal_out_kzt.sum() == 4.5
    assert metadata.iloc[0].gids == [99]
    assert graph[2][10]["sum_kzt"] == 1.25


def test_clustering_empty_graph_retains_isolates_and_empty_types():
    gids = [10, 2, 2**53 + 1]
    nodes = pd.DataFrame({"gid": gids, "is_seed": False})
    actual, metadata = cluster_graph(nx.DiGraph(), nodes)
    assert actual.gid.tolist() == gids
    assert metadata.gids.tolist() == [[2], [10], [2**53 + 1]]
    assert actual.drop(columns=["gid", "cluster_id"]).eq(0).all().all()
    empty_nodes = pd.DataFrame({"gid": pd.Series(dtype="int64"), "is_seed": pd.Series(dtype=bool)})
    empty, empty_metadata = cluster_graph(nx.DiGraph(), empty_nodes)
    assert empty.empty and empty_metadata.empty
    assert empty.gid.dtype == np.dtype("int64")
    assert empty.cluster_id.dtype == np.dtype("int64")


def test_resilience_preserves_baseline_isolates_and_resolves_component_ties():
    graph = nx.DiGraph([(20, 21), (2, 3)])
    features = pd.DataFrame({"gid": [20, 21, 2, 3, 99], "is_seed": [True, False, False, False, True],
                             "priority_score": [.5, .5, 1, .5, .5]})
    actual = resilience_analysis(graph, features)
    baseline, top_one, top_five = actual.iloc[0], actual.iloc[1], actual.iloc[2]
    assert baseline.n_weak_components == 3
    assert baseline.largest_weak_component_size == 2
    assert baseline.fraction_baseline_largest == 1
    assert baseline.seeds_in_largest_component == 0
    assert top_one.seeds_in_largest_component == 1
    assert top_five.n_weak_components == 0
    assert top_five.largest_weak_component_size == 0
    reordered = nx.DiGraph([(2, 3), (20, 21)])
    pd.testing.assert_frame_equal(actual, resilience_analysis(reordered, features.sample(frac=1, random_state=7)))
    assert 99 not in graph and graph.number_of_nodes() == 4
    with pytest.raises(ValueError, match="nonnegative integers"):
        resilience_analysis(graph, features, removal_counts=(-1,))


def test_top_monotonicity_checked_even_for_tiny_difference(tmp_path):
    frame, _, sources, graph = export_fixture(tmp_path)
    frame["priority_score"] = .2
    frame["priority_seed_reach_count_contribution"] = .1
    write_exports(frame, cluster_summaries(frame, graph, sources), tmp_path)
    def change(top):
        top.loc[0, "priority_score"] = str(.2 - 1e-13)
    rewrite_csv(tmp_path, "top_nodes.csv", change)
    with pytest.raises(SubmissionValidationError, match="monotonically descending"):
        validate_submission(tmp_path, expected_nodes=60)
