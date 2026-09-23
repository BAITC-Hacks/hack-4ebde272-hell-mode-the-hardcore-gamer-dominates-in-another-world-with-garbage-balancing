"""Graph and temporal feature behavior tests."""

import numpy as np
import pandas as pd

from src.graph_features import build_features, build_graph, percentile_rank
from src.schema import validate_inputs


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
