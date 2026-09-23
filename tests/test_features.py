"""Graph and temporal feature behavior tests."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.graph_features import (
    _peer_anomaly_components,
    build_features,
    build_graph,
    percentile_rank,
    weighted_pagerank,
)
from src.loader import load_inputs
from src.schema import validate_inputs
from src.temporal import temporal_features
import src.temporal as temporal


def _case():
    nodes = pd.DataFrame({
        "gid": [10, 20, 30, 40],
        "depth": [0, 1, 2, 4],
        "is_seed": [True, False, False, False],
    })
    transactions = pd.DataFrame({
        "src": [10, 10, 20, 30, 30],
        "dst": [20, 30, 30, 10, 40],
        "date": ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03", "2026-01-04"],
        "sum_kzt": [10.0, 15.0, 4.0, 3.0, 7.0],
    })
    edges = transactions.groupby(["src", "dst"], as_index=False).agg(
        sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size")
    )
    edges["depth"] = 1
    return nodes, edges, transactions


def test_seed_ratio_metrics_are_invalid_for_seed_nodes():
    features = build_features(*_case())

    assert features.loc[10, "in_kzt"] > 0
    assert features.loc[10, "out_kzt"] > 0
    assert np.isnan(features.loc[10, "pass_through"])
    assert np.isnan(features.loc[10, "fanin_share"])
    assert np.isnan(features.loc[10, "fanout_share"])
    assert np.isnan(features.loc[10, "same_day_flow_ratio"])
    assert np.isnan(features.loc[10, "relay_2d_ratio"])
    assert np.isnan(features.loc[10, "peak_day_share"])
    assert np.isfinite(features.loc[20, "pass_through"])


def test_hop_four_boundary_is_marked_without_inventing_terminal_label():
    features = build_features(*_case())

    assert features.loc[40, "boundary_censored"]
    assert features.loc[40, "out_deg"] == 0
    assert features.loc[40, "in_kzt"] > 0
    assert np.isnan(features.loc[40, "pass_through"])
    assert np.isnan(features.loc[40, "fanout_share"])
    assert "terminal" not in features.columns
    assert "truncated_by_depth" not in features.columns


def test_graph_is_directed_and_keeps_aggregated_attributes():
    nodes, edges, _ = _case()
    nodes, edges, _ = validate_inputs(*_case())
    graph = build_graph(nodes, edges)

    assert graph.is_directed()
    assert graph[10][20]["sum_kzt"] == 10
    assert graph[10][20]["n_tx"] == 1
    assert "distance" in graph[10][20]


def test_percentile_ranks_are_bounded_and_tie_deterministic():
    ranks = percentile_rank(pd.Series([4.0, 1.0, 1.0, np.nan]))

    assert ranks.iloc[0] == 1.0
    assert ranks.iloc[1] == ranks.iloc[2] == 0.5
    assert np.isnan(ranks.iloc[3])


def test_feature_output_is_gid_indexed_and_includes_requested_signals():
    features = build_features(*_case())

    assert features.index.name == "gid"
    assert features.index.tolist() == [10, 20, 30, 40]
    for column in (
        "in_deg", "out_deg", "in_kzt", "out_kzt", "in_tx", "out_tx", "total_kzt", "total_tx",
        "pagerank", "betweenness", "seed_reach_count", "seed_reach_fraction", "min_seed_distance",
        "scc_id", "scc_size", "in_cycle", "reciprocal_relationship_count", "active_days",
        "same_day_flow_ratio", "relay_2d_ratio", "max_in_senders_day", "peak_day_share", "peer_anomaly_score",
        "pagerank_pct", "betweenness_pct",
    ):
        assert column in features
    assert features["peer_anomaly_score"].between(0, 1).all()
    assert features.loc[10, "seed_reach_count"] == 1
    assert features.loc[30, "seed_reach_count"] == 1


def _tables(rows, *, gids=None, seeds=(), depths=None):
    """Create consistent small fixtures without imposing the supplied profile."""
    tx = pd.DataFrame(rows, columns=["src", "dst", "date", "sum_kzt"])
    if gids is None:
        gids = sorted(set(tx.src) | set(tx.dst))
    depths = depths or {}
    nodes = pd.DataFrame({"gid": gids, "depth": [depths.get(g, 0 if g in seeds else 1) for g in gids],
                          "is_seed": [g in seeds for g in gids]})
    edges = tx.groupby(["src", "dst"], as_index=False).agg(
        sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
    edges["depth"] = 1
    return validate_inputs(nodes, edges, tx)


def test_base_features_and_directed_amount_distance_betweenness_exact():
    nodes, edges, tx = _tables([(1, 2, "2026-07-01", 10), (2, 3, "2026-07-03", 4)])
    result = build_features(nodes, edges, tx)
    middle = result.loc[2]
    assert middle[["in_deg", "out_deg", "in_tx", "out_tx", "total_tx"]].tolist() == [1, 1, 1, 1, 2]
    assert middle[["in_kzt", "out_kzt", "total_kzt"]].tolist() == [10, 4, 14]
    assert middle.pass_through == 0.4
    assert middle.fanin_share == middle.fanout_share == 0.5
    assert middle.pass_through_valid and middle.flow_share_valid
    assert middle.pass_through_invalid_reason == ""
    assert result.betweenness.tolist() == [0.0, 0.5, 0.0]
    graph = build_graph(nodes, edges)
    assert graph[1][2]["distance"] == pytest.approx(1 / np.log1p(10))
    assert not graph.has_edge(2, 1)
    assert result.pagerank.sum() == pytest.approx(1)


def test_weighted_pagerank_matches_two_node_closed_form():
    nodes, edges, _ = _tables([(1, 2, "2026-07-01", 100)])
    ranks = weighted_pagerank(build_graph(nodes, edges), tol=1e-12)
    assert ranks[1] == pytest.approx(20 / 57, abs=1e-10)
    assert ranks[2] == pytest.approx(37 / 57, abs=1e-10)


def test_amount_distance_can_prefer_two_high_amount_edges_over_direct_edge():
    tables = _tables([(1, 4, "2026-07-01", 1), (1, 2, "2026-07-01", 1000),
                      (2, 4, "2026-07-02", 1000)], gids=[1, 2, 3, 4])
    result = build_features(*tables)
    assert result.loc[2, "betweenness"] == pytest.approx(1 / 6)
    assert result.loc[[1, 3, 4], "betweenness"].eq(0).all()


def test_seed_convergence_cutoff_self_distance_and_isolates_exact():
    pairs = [(1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    tables = _tables([(a, b, "2026-07-01", 10) for a, b in pairs],
                     gids=list(range(1, 9)), seeds=(1, 2, 8),
                     depths={3: 1, 4: 2, 5: 3, 6: 4, 7: 4})
    result = build_features(*tables)
    assert result.loc[3, "seed_reach_count"] == 2
    assert result.loc[3, "seed_reach_fraction"] == pytest.approx(2 / 3)
    assert result.loc[3, "min_seed_distance"] == 1
    assert result.loc[6, "seed_reach_count"] == 2
    assert result.loc[6, "min_seed_distance"] == 4
    assert result.loc[7, "seed_reach_count"] == 0
    assert np.isnan(result.loc[7, "min_seed_distance"])
    assert result.loc[8, "seed_reach_count"] == 1  # each seed reaches itself at D=0
    assert result.loc[8, "min_seed_distance"] == 0
    assert result.loc[8, ["in_deg", "out_deg", "active_days", "total_kzt"]].eq(0).all()
    assert result.loc[8, "pass_through_invalid_reason"] == "seed_inbound_incomplete"
    assert len(build_graph(tables[0], tables[1])) == 8


def test_scc_reciprocity_and_singleton_self_loop_semantics():
    pairs = [(1, 2), (2, 3), (3, 1), (2, 1), (4, 4)]
    tables = _tables([(a, b, "2026-07-01", 10) for a, b in pairs], gids=[1, 2, 3, 4, 5])
    result = build_features(*tables)
    assert result.loc[[1, 2, 3], "scc_id"].nunique() == 1
    assert result.scc_size.tolist() == [3, 3, 3, 1, 1]
    assert result.in_cycle.tolist() == [True, True, True, False, False]
    assert result.reciprocal_relationship_count.tolist() == [1, 1, 0, 0, 0]


@pytest.mark.parametrize("values,expected", [
    ([np.nan, np.inf, -np.inf], [np.nan, np.nan, np.nan]),
    ([0.25, np.nan], [1.0, np.nan]),
    ([0, 0, 0], [2 / 3, 2 / 3, 2 / 3]),
    ([4, 1, 1, np.nan], [1, .5, .5, np.nan]),
    ([], []),
])
def test_percentile_policy(values, expected):
    actual = percentile_rank(pd.Series(values, dtype=float))
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_peer_components_zero_mad_singleton_and_formula():
    frame = pd.DataFrame({
        "depth": [1, 1, 1, 2, 3, 3, 3], "in_deg": [0, 0, 3, 90, 1, 2, 4],
        "out_deg": [0] * 7, "in_kzt": [0] * 7, "out_kzt": [0] * 7,
        "in_tx": [0] * 7, "out_tx": [0] * 7,
    })
    signals = _peer_anomaly_components(frame)
    assert signals.loc[:3, "peer_anomaly_in_deg"].tolist() == [0, 0, 1, 0]
    z = 0.67448975 * 2  # final depth-3 node: median=2, MAD=1, deviation=2
    assert signals.loc[6, "peer_anomaly_in_deg"] == pytest.approx(z / (1 + z))
    assert signals.drop(columns="peer_anomaly_in_deg").eq(0).all().all()
    result = build_features(*_case())
    component_columns = list(signals.columns)
    np.testing.assert_allclose(result.peer_anomaly_score, result[component_columns].mean(axis=1))
    assert result.peer_group_size.tolist() == [1, 1, 1, 1]
    assert result.peer_anomaly_score.eq(0).all()


@pytest.mark.parametrize("offset,expected", [(0, 1), (1, 1), (2, 1), (3, 0)])
def test_relay_offsets_exact(offset, expected):
    day = pd.Timestamp("2026-07-10") + pd.Timedelta(days=offset)
    tables = _tables([(1, 2, "2026-07-10", 10), (2, 3, day, 5),
                      (8, 9, "2026-07-31", 1)])
    row = temporal_features(tables[0], tables[2]).loc[2]
    assert row.relay_2d_ratio == expected
    assert row.relay_2d_eligible_days == 1
    assert row.relay_2d_matched_days == expected
    assert row.relay_2d_censored_days == 0
    assert row.relay_2d_valid
    assert row.same_day_flow_ratio == (1 if offset == 0 else 0)


def test_relay_month_end_excludes_even_positive_partial_windows():
    tables = _tables([(1, 2, "2026-07-28", 10), (2, 3, "2026-07-29", 5),
                      (1, 2, "2026-07-29", 10), (1, 2, "2026-07-30", 10),
                      (1, 2, "2026-07-31", 10), (2, 3, "2026-07-31", 5),
                      (1, 4, "2026-07-31", 10)])
    result = build_features(*tables)
    assert result.loc[2, "relay_2d_ratio"] == 1
    assert result.loc[2, "inbound_active_days"] == 4
    assert result.loc[2, "relay_2d_eligible_days"] == 2
    assert result.loc[2, "relay_2d_matched_days"] == 2
    assert result.loc[2, "relay_2d_censored_days"] == 2
    assert np.isnan(result.loc[4, "relay_2d_ratio"])
    assert not result.loc[4, "relay_2d_valid"]
    assert result.loc[4, "relay_2d_invalid_reason"] == "no_complete_followup_window"
    assert result.loc[2, "temporal_observation_end"] == pd.Timestamp("2026-07-31")


def test_boundary_and_seed_ratios_invalid_even_with_observed_outgoing():
    tables = _tables([(1, 2, "2026-07-01", 10), (2, 1, "2026-07-02", 2),
                      (2, 3, "2026-07-03", 5), (8, 9, "2026-07-05", 1)],
                     seeds=(1,), depths={2: 4})
    result = build_features(*tables)
    for gid, reason in [(1, "seed_inbound_incomplete"), (2, "boundary_outbound_incomplete")]:
        assert result.loc[gid, ["pass_through", "same_day_flow_ratio", "relay_2d_ratio"]].isna().all()
        assert not result.loc[gid, "pass_through_valid"]
        assert not result.loc[gid, "relay_2d_valid"]
        assert result.loc[gid, "relay_2d_invalid_reason"] == reason
    assert np.isnan(result.loc[1, "peak_day_share"])
    assert result.loc[2, "peak_day_share_valid"]  # describes observed activity only


def test_temporal_daily_amounts_sender_burst_and_earliest_ties():
    tables = _tables([(1, 2, "2026-07-01", 10), (3, 2, "2026-07-01", 20),
                      (2, 4, "2026-07-01", 5), (2, 4, "2026-07-02", 35),
                      (8, 9, "2026-07-05", 1)])
    row = build_features(*tables).loc[2]
    assert row.active_days == 2
    assert row.inbound_active_days == 1
    assert row.outbound_active_days == 2
    assert row.same_day_flow_ratio == 5 / 40
    assert row.peak_day_share == .5
    assert row.peak_activity_kzt == 35
    assert row.peak_activity_date == pd.Timestamp("2026-07-01")
    assert row.max_in_senders_day == 2
    assert row.max_in_senders_date == pd.Timestamp("2026-07-01")


def test_no_seed_or_transaction_nodes_and_empty_input():
    tables = _tables([], gids=[9, 2])
    result = build_features(*tables)
    assert result.index.tolist() == [9, 2]
    assert result.seed_reach_count.eq(0).all()
    assert result.seed_reach_fraction.eq(0).all()
    assert result.min_seed_distance.isna().all()
    assert result.pagerank.eq(.5).all()
    assert result.active_days.eq(0).all()
    assert result.relay_2d_ratio.isna().all()
    assert result.temporal_observation_end.isna().all()
    assert result.repeated_route_evidence.eq("[]").all()
    assert result.outgoing_repeated_amount_tx_share.isna().all()
    empty = build_features(*_tables([], gids=[]))
    assert empty.empty and empty.index.name == "gid"
    assert empty.columns.tolist() == result.columns.tolist()
    for column in ("active_days", "repeated_route_count", "in_deg"):
        assert str(empty[column].dtype) == "int64"
    for column in ("in_kzt", "out_kzt", "total_kzt"):
        assert str(empty[column].dtype) == str(result[column].dtype) == "float64"
    pd.testing.assert_series_equal(empty.dtypes, build_features(*_case()).dtypes)


def test_node_amount_overflow_is_reported_before_graph_algorithms():
    tables = _tables([(1, 2, "2026-07-01", 1e308), (1, 3, "2026-07-01", 1e308)])
    with pytest.raises(ValueError, match="node amount totals"):
        build_features(*tables)


def test_exact_large_ids_and_input_permutations_preserve_all_features():
    ids = [100000000011452101, 100000000011452102, 100000000011452103, 100000000011452104]
    tables = _tables([(ids[0], ids[1], "2026-07-01", 10),
                      (ids[0], ids[1], "2026-07-03", 10),
                      (ids[1], ids[2], "2026-07-02", 10),
                      (ids[1], ids[2], "2026-07-04", 10)], gids=ids, seeds=(ids[0],))
    first = build_features(*tables)
    shuffled = [table.sample(frac=1, random_state=7) for table in tables]
    second = build_features(*shuffled)
    assert second.index.tolist() == shuffled[0].gid.tolist()
    assert_frame_equal(first.sort_index(), second.sort_index(), check_exact=True)
    a, b = build_graph(*tables[:2]), build_graph(*shuffled[:2])
    assert list(a.nodes) == list(b.nodes) == ids
    assert list(a.edges(data=True)) == list(b.edges(data=True))
    records = json.loads(first.loc[ids[1], "repeated_route_evidence"])
    assert records[0]["src"] == str(ids[0])
    assert records[0]["via"] == str(ids[1])
    assert records[0]["dst"] == str(ids[2])
    buffer = io.BytesIO()
    first.to_parquet(buffer)
    buffer.seek(0)
    assert_frame_equal(first, pd.read_parquet(buffer))


def test_feature_construction_does_not_mutate_caller_tables():
    tables = _case()
    original = [table.copy(deep=True) for table in tables]
    build_features(*tables)
    for before, after in zip(original, tables):
        assert_frame_equal(before, after)


def test_repeated_routes_return_dates_and_evidence_do_not_double_count_out_days():
    tables = _tables([(1, 2, "2026-07-01", 10), (1, 2, "2026-07-02", 20),
                      (2, 3, "2026-07-03", 8), (2, 1, "2026-07-03", 4),
                      (8, 9, "2026-07-06", 1)])
    row = build_features(*tables).loc[2]
    assert row.repeated_route_count == 1
    assert row.repeated_route_max_support_days == 2
    assert row.temporal_return_count == 1
    assert row.temporal_return_max_support_days == 2
    evidence = json.loads(row.repeated_route_evidence)[0]
    assert evidence["date_pairs"] == [["2026-07-01", "2026-07-03"], ["2026-07-02", "2026-07-03"]]
    assert evidence["in_kzt_on_support_dates"] == 30
    assert evidence["out_kzt_on_matched_dates"] == 8  # same outgoing date counted once
    assert not row.repeated_route_truncated


def test_positive_patterns_include_observed_end_dates_and_reject_d_plus_three():
    tables = _tables([(1, 2, "2026-07-27", 10), (1, 2, "2026-07-30", 10),
                      (2, 3, "2026-07-30", 8), (2, 3, "2026-07-31", 8),
                      (2, 1, "2026-07-31", 5)])
    row = build_features(*tables).loc[2]
    assert row.repeated_route_count == 0  # Jul27->30 is D+3; only Jul30 matches
    assert row.temporal_return_count == 1
    assert row.relay_2d_ratio == 0  # Jul30 excluded from the complete-window denominator
    assert row.relay_2d_censored_days == 1


def test_pattern_pair_and_day_budgets_report_lower_bounds(monkeypatch):
    rows = [(source, 2, date, 10) for source in [1, 3] for date in ["2026-07-01", "2026-07-02"]]
    rows += [(2, target, date, 5) for target in [1, 3, 4, 5] for date in ["2026-07-02", "2026-07-03"]]
    tables = _tables(rows)
    monkeypatch.setattr(temporal, "PATTERN_PAIR_LIMIT", 1)
    row = build_features(*tables).loc[2]
    assert row.repeated_route_truncated and row.temporal_return_truncated
    assert row.repeated_route_count <= 1 and row.temporal_return_count <= 1
    monkeypatch.setattr(temporal, "PATTERN_PAIR_LIMIT", 512)
    monkeypatch.setattr(temporal, "PATTERN_DAY_CHECK_LIMIT", 1)
    row = build_features(*tables).loc[2]
    assert row.repeated_route_truncated and row.temporal_return_truncated
    assert row.repeated_route_count == 0  # one probe cannot establish two support dates


def test_observed_amount_patterns_exact_counts_denominators_and_tolerance():
    tables = _tables([(2, 3, "2026-07-01", 100), (2, 4, "2026-07-01", 100),
                      (2, 5, "2026-07-01", 105), (2, 6, "2026-07-01", 105.01),
                      (2, 3, "2026-07-02", 300)])
    row = build_features(*tables).loc[2]
    assert row.outgoing_repeated_amount_tx_count == 2
    assert row.outgoing_repeated_amount_tx_share == 2 / 5
    assert row.outgoing_similar_amount_tx_count == 3
    assert row.outgoing_similar_amount_tx_share == 3 / 5
    assert row.similar_amount_group_count == 1
    evidence = json.loads(row.amount_pattern_evidence)
    similar = next(record for record in evidence if record["kind"] == "same_day_similar_amounts")
    assert similar["n_tx"] == 3 and similar["n_recipients"] == 3
    assert similar["min_kzt"] == 100 and similar["max_kzt"] == 105


def test_similar_amounts_require_multiple_destinations_and_same_date():
    tables = _tables([(2, 3, "2026-07-01", 100), (2, 3, "2026-07-01", 101),
                      (2, 3, "2026-07-01", 102), (2, 4, "2026-07-02", 100)])
    row = build_features(*tables).loc[2]
    assert row.similar_amount_group_count == 0
    assert row.outgoing_similar_amount_tx_share == 0


def test_supplied_dataset_features_and_permutation_contract():
    root = Path(__file__).resolve().parents[1]
    tables = load_inputs(root / "data")
    result = build_features(*tables)
    assert len(result) == 2248 and result.index.is_unique
    assert str(result.index.dtype) == "int64"
    assert result.is_seed.sum() == 81
    assert result.boundary_censored.sum() == 444
    unavailable = result.is_seed | result.boundary_censored
    assert result.loc[unavailable, ["pass_through", "relay_2d_ratio", "same_day_flow_ratio"]].isna().all().all()
    assert result.loc[unavailable, "relay_2d_valid"].eq(False).all()
    assert ((result.in_deg == 0) & (result.out_deg == 0)).sum() == 19
    assert result.loc[result.relay_2d_valid, "relay_2d_ratio"].between(0, 1).all()
    assert result.temporal_observation_end.eq(pd.Timestamp("2026-07-31")).all()
    shuffled = [table.sample(frac=1, random_state=7) for table in tables]
    assert_frame_equal(result.sort_index(), build_features(*shuffled).sort_index(), check_exact=True)
    buffer = io.BytesIO()
    result.to_parquet(buffer)
    buffer.seek(0)
    assert_frame_equal(result, pd.read_parquet(buffer))
