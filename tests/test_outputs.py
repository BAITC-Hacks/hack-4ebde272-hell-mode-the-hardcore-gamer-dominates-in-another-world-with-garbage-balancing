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
    ("consolidator", "collection point"), ("distributor", "distribution role"), ("transit", "incoming amount"),
    ("coordinator", "coordinating account"), ("terminal", "suggesting an endpoint"), ("peripheral", "no specific role"),
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
    assert "counterparties" not in text and "percentile" not in text
    assert text.endswith(".")


def test_boundary_consolidator_keeps_role_reason():
    text = evidence_for(pd.Series({"role": "consolidator", "depth": 4, "in_deg": 11,
                                   "out_deg": 0, "in_kzt": 8.2e6, "seed_reach_count": 14}))
    assert "collection point" in text and "11" in text and "14" in text
    assert "Outgoing transfers beyond hop 4 are unobserved." in text
    assert len(text) <= 200


def test_empty_isolated_and_missing_peripheral_evidence():
    empty = add_evidence(pd.DataFrame(columns=["gid", "role"]))
    assert empty.empty and "evidence" in empty
    isolated = evidence_for(pd.Series({"role": "peripheral", "is_seed": True, "in_deg": 0, "out_deg": 0}))
    assert "0 incoming, 0 outgoing" in isolated and "Initial case account; incoming transfers are incomplete." in isolated
    unavailable = evidence_for(pd.Series({"role": "peripheral"}))
    assert "unavailable" in unavailable and "0.55" in unavailable


@pytest.mark.parametrize("validity_flag", ["relay_2d_valid", "relay_2d_ratio_valid", "relay_2d_ratio_available", "decision_relay_2d_ratio_valid"])
@pytest.mark.parametrize("invalid", [False, np.nan, pd.NA])
def test_transit_explanation_ignores_invalid_raw_temporal_ratio(validity_flag, invalid):
    row = pd.Series({"role": "transit", "depth": 2, "in_deg": 3, "out_deg": 4,
                     "pass_through": .8, "relay_2d_ratio": .95, validity_flag: invalid})
    text = evidence_for(row)
    assert "Two-day timing is unavailable." in text
    assert "95%" not in text
    assert "80%" in text


def test_transit_explanation_uses_sanitized_decision_ratios():
    row = pd.Series({"role": "transit", "depth": 2, "in_deg": 3, "out_deg": 4,
                     "pass_through": .8, "relay_2d_ratio": .95,
                     "decision_pass_through": .7, "decision_relay_2d_ratio": .6})
    text = evidence_for(row)
    assert "70%" in text and "60%" in text
    assert "80%" not in text and "95%" not in text


def test_collection_and_distribution_explain_observed_numbers_in_plain_language():
    collection = evidence_for(pd.Series({"role": "consolidator", "depth": 2,
        "in_deg": 11, "out_deg": 2, "in_kzt": 12500.50, "seed_reach_count": 3}))
    assert collection == ("Received 12,500.5 KZT from 11 senders; a possible collection point. "
                          "Reachable from 3 initial case accounts.")
    distribution = evidence_for(pd.Series({"role": "distributor", "depth": 0,
        "is_seed": True, "in_deg": 0, "out_deg": 8, "out_tx": 12, "out_kzt": 125000}))
    assert distribution == ("Sent 125,000 KZT to 8 recipients in 12 transfers, suggesting a distribution role. "
                            "Initial case account; incoming transfers are incomplete.")


def test_relay_explanation_names_eligible_dates_and_does_not_trace_funds():
    row = pd.Series({"role": "transit", "depth": 2, "in_deg": 2, "out_deg": 3,
                     "pass_through": .8, "relay_2d_ratio": .75})
    text = evidence_for(row)
    assert "Outgoing amount is 80% of incoming amount" in text
    assert "on or up to 2 days after 75% of eligible incoming dates" in text
    assert "Dates cannot prove order or trace funds." in text
    row["relay_2d_ratio"] = np.nan
    text = evidence_for(row)
    assert "80%" in text and "Two-day timing is unavailable." in text
    assert "0%" not in text.replace("80%", "")


def test_one_sender_and_one_recipient_are_described_in_the_singular():
    text = evidence_for(pd.Series({"role": "peripheral", "in_deg": 1, "out_deg": 1}))
    assert text == "Observed 1 sender and 1 recipient; no specific role has enough support."


def test_initial_case_account_reach_explains_its_own_zero_hop_membership():
    text = evidence_for(pd.Series({"role": "consolidator", "is_seed": True, "depth": 0,
        "in_deg": 2, "out_deg": 1, "in_kzt": 1000, "seed_reach_count": 1}))
    assert "Reachable from 1 initial case account (including itself)." in text
    assert "incoming transfers are incomplete" in text
    assert len(text) <= 200


def test_coordinator_relative_ranks_remain_numeric_and_clearly_hypothetical():
    text = evidence_for(pd.Series({"role": "coordinator", "depth": 1,
        "in_deg": 2, "out_deg": 3, "seed_reach_count": 1, "cross_cluster_degree": 0,
        "decision_betweenness_percentile": 1.0, "decision_pagerank_percentile": 1.0}))
    assert text == ("Possible coordinating account: rank 1 for linking payment paths; "
                    "rank 1 for network importance.")
    assert "percentile" not in text and "organizer" not in text


@pytest.mark.parametrize("role", ["consolidator", "distributor", "coordinator", "transit", "terminal", "peripheral"])
@pytest.mark.parametrize("seed,boundary", [(False, False), (True, False), (False, True), (True, True)])
def test_evidence_budget_preserves_whole_sentences_and_complete_observation_warnings(role, seed, boundary):
    row = pd.Series({"role": role, "depth": 4 if boundary else 1, "is_seed": seed,
        "in_deg": 2**63 - 1, "out_deg": 2**63 - 1, "in_kzt": 1e300,
        "out_kzt": 1e300, "out_tx": 2**63 - 1, "seed_reach_count": 2**63 - 1,
        "cross_cluster_degree": 2**63 - 1, "pass_through": 1.0, "relay_2d_ratio": .75,
        "decision_betweenness_percentile": .99, "decision_pagerank_percentile": .98})
    text = evidence_for(row)
    assert 0 < len(text) <= 200 and text.endswith(".")
    assert any(char.isdigit() for char in text)
    assert "..." not in text and "unavailabl." not in text
    if seed:
        assert "Initial case account" in text and "incomplete" in text
    if boundary:
        assert "outgoing transfers beyond hop 4 are unobserved." in text.lower()
    # Bounded alternatives end on authored complete phrases, not chopped words.
    endings = ("incomplete.", "unobserved.", "collection point.", "initial case accounts.",
               "distribution role.", "distributor.", "network importance.", "coordinating role.",
               "trace funds.", "timing is unavailable.", "account balance.", "balance is unknown.",
               "enough support.", "cutoff is 0.55.")
    assert text.endswith(endings)


def cluster_narrative(gids, transfers, *, seeds=(), roles=None):
    graph = nx.DiGraph()
    graph.add_nodes_from(gids)
    for source, destination, amount in transfers:
        graph.add_edge(source, destination, sum_kzt=amount)
    members = pd.DataFrame({"gid": gids, "cluster_id": 0, "priority_score": 0.5,
        "role": roles or ["peripheral"] * len(gids),
        "in_deg": [graph.in_degree(gid) for gid in gids],
        "out_deg": [graph.out_degree(gid) for gid in gids]})
    nodes = pd.DataFrame({"gid": sorted(graph), "is_seed": [gid in seeds for gid in sorted(graph)]})
    result = cluster_summaries(members, graph, nodes)
    repeated = cluster_summaries(members.iloc[::-1], graph, nodes.iloc[::-1])
    pd.testing.assert_frame_equal(result, repeated)
    return result.iloc[0]


def test_cluster_hypothesis_distinguishes_an_isolated_initial_case_account():
    cluster = cluster_narrative([1], [], seeds=[1])
    assert cluster.n_nodes == 1 and cluster.n_seed == 1 and cluster.sum_kzt_internal == 0
    assert cluster.hypothesis.startswith("This group contains 1 initial case account with no observed transfers.")
    assert "active outside the observed period or bank" in cluster.hypothesis
    assert "convergence" not in cluster.hypothesis


def test_multiple_case_accounts_in_a_cluster_do_not_claim_directed_convergence():
    # Seed 1 reaches 3, while seed 2 has no outgoing path; shared membership
    # cannot establish a common downstream collector or common controller.
    cluster = cluster_narrative([1, 2, 3, 4], [(1, 3, 100), (4, 2, 200)], seeds=[1, 2])
    assert "4 accounts, including 2 initial case accounts" in cluster.hypothesis
    assert "300 KZT transferred within the group" in cluster.hypothesis
    assert "does not prove that funds converge or that one person controls the accounts" in cluster.hypothesis


@pytest.mark.parametrize("reverse,label", [(False, "collection"), (True, "distribution")])
def test_group_flow_hypothesis_uses_actual_link_counts_and_internal_amount(reverse, label):
    transfers = [(1, 3, 10), (2, 3, 20), (5, 4, 30), (3, 4, 40)]
    if reverse:
        transfers = [(dst, src, amount) for src, dst, amount in transfers]
    cluster = cluster_narrative([3, 4], transfers)
    assert f"Possible {label} group" in cluster.hypothesis
    assert "4 outgoing and 1 incoming" in cluster.hypothesis if reverse else "4 incoming and 1 outgoing" in cluster.hypothesis
    assert cluster.sum_kzt_internal == 40 and "Internal transfers total 40 KZT" in cluster.hypothesis


@pytest.mark.parametrize("roles,expected", [
    (["transit", "transit", "peripheral", "terminal"], "2 of 4 accounts have a relay role"),
    (["peripheral"] * 4, "Most accounts (4 of 4) lack enough evidence"),
    (["consolidator", "distributor", "peripheral", "terminal"], "1 possible collector and 1 possible distributor"),
])
def test_group_hypothesis_explains_role_composition_without_asserting_purpose(roles, expected):
    cluster = cluster_narrative([1, 2, 3, 4], [(1, 2, 100), (2, 3, 100), (3, 4, 100), (4, 1, 100)], roles=roles)
    assert expected in cluster.hypothesis
    assert "400 KZT" in cluster.hypothesis and cluster.hypothesis.endswith(".")
    assert "criminal" not in cluster.hypothesis and "organizer" not in cluster.hypothesis


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
