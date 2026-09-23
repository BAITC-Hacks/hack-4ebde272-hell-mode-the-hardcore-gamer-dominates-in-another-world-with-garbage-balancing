import numpy as np
import pandas as pd
import pytest

from src.graph_features import percentile_rank
from src.priority import WEIGHTS, score_priority
from src.roles import assign_roles, observed_ratio, percentile


def test_role_assignment_is_valid_and_cutoff_is_not_terminal():
    features = pd.DataFrame([
        {"gid": "seed", "is_seed": True, "depth": 0, "in_deg": 0, "out_deg": 2,
         "in_tx": 0, "out_tx": 5, "in_kzt": 0, "out_kzt": 900, "seed_reach_count": 1,
         "pass_through": float("nan"), "relay_2d_ratio": float("nan"),
         "same_day_flow_ratio": float("nan"), "balance_score": float("nan"),
         "fanout_share": 1.0, "betweenness": 0.5, "pagerank": 0.5,
         "cross_cluster_out_deg": 1, "cross_cluster_degree": 1, "total_kzt": 900},
        {"gid": "cutoff", "is_seed": False, "depth": 4, "in_deg": 1, "out_deg": 0,
         "in_tx": 2, "out_tx": 0, "in_kzt": 100, "out_kzt": 0, "seed_reach_count": 1,
         "pass_through": 0.0, "relay_2d_ratio": 0.0, "same_day_flow_ratio": 0.0,
         "balance_score": 0.0, "fanout_share": 0.0, "betweenness": 0.1,
         "pagerank": 0.1, "cross_cluster_out_deg": 0, "cross_cluster_degree": 0,
         "total_kzt": 100},
        {"gid": "sink", "is_seed": False, "depth": 2, "in_deg": 3, "out_deg": 0,
         "in_tx": 5, "out_tx": 0, "in_kzt": 500, "out_kzt": 0, "seed_reach_count": 3,
         "pass_through": 0.0, "relay_2d_ratio": 0.0, "same_day_flow_ratio": 0.0,
         "balance_score": 0.0, "fanout_share": 0.0, "betweenness": 0.4,
         "pagerank": 0.4, "cross_cluster_out_deg": 0, "cross_cluster_degree": 1,
         "total_kzt": 500},
    ])
    result = assign_roles(features).set_index("gid")
    assert set(result.role).issubset({"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"})
    assert result.loc["cutoff", "role"] != "terminal"
    assert 0 <= result.role_score.min() <= result.role_score.max() <= 1


def test_invalid_seed_ratio_does_not_poison_role_score():
    frame = pd.DataFrame([{"gid": 1, "is_seed": True, "depth": 0, "in_deg": 0,
                           "out_deg": 2, "out_tx": 4, "out_kzt": 1000,
                           "in_tx": 0, "in_kzt": 0, "seed_reach_count": 1,
                           "pass_through": float("nan"), "relay_2d_ratio": float("nan"),
                           "same_day_flow_ratio": float("nan"), "balance_score": float("nan"),
                           "fanout_share": 1, "betweenness": 0.5, "pagerank": 0.5,
                           "cross_cluster_out_deg": 1, "cross_cluster_degree": 1, "total_kzt": 1000}])
    result = assign_roles(frame).iloc[0]
    assert result.role in {"distributor", "coordinator", "peripheral"}
    assert 0 <= result.role_score <= 1


def feature_row(**overrides):
    row = {"gid": 1, "is_seed": False, "depth": 2, "in_deg": 1, "out_deg": 1,
           "in_tx": 1, "out_tx": 1, "in_kzt": 100.0, "out_kzt": 100.0,
           "seed_reach_count": np.nan, "betweenness": np.nan, "pagerank": np.nan,
           "cross_cluster_degree": np.nan, "cross_cluster_out_deg": 0,
           "total_kzt": 200.0, "pass_through": 1.0, "relay_2d_ratio": 1.0,
           "same_day_flow_ratio": 1.0, "fanout_share": 1.0,
           "peer_anomaly_score": 0.0}
    row.update(overrides)
    return row


@pytest.mark.parametrize("values", [[0.2], [5.0, np.nan], [2.0, 2.0, 2.0],
                                    [1.0, 2.0, 2.0, 4.0, np.nan], [np.nan]])
def test_decision_percentiles_use_shared_policy(values):
    series = pd.Series(values)
    pd.testing.assert_series_equal(percentile(series), percentile_rank(series))


def test_nonfinite_percentiles_stay_unavailable():
    result = percentile(pd.Series([np.inf, -np.inf, np.nan, 0.2]))
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == 1.0


@pytest.mark.parametrize("ratio,relay,expected", [
    (0.5, 0.0, True), (1.5, 0.0, True), (0.499999, 0.0, False),
    (1.500001, 0.0, False), (0.1, 0.5, True), (0.1, 0.499999, False),
])
def test_transit_structural_thresholds_are_inclusive(ratio, relay, expected):
    row = feature_row(pass_through=ratio, out_kzt=100 * ratio, relay_2d_ratio=relay)
    result = assign_roles(pd.DataFrame([row])).iloc[0]
    assert bool(result.role_transit_gate) == expected


@pytest.mark.parametrize("ratio,expected", [(0.0, True), (0.1, True), (0.100001, False)])
def test_terminal_observed_amount_threshold(ratio, expected):
    row = feature_row(pass_through=ratio, out_kzt=100 * ratio, relay_2d_ratio=0)
    result = assign_roles(pd.DataFrame([row])).iloc[0]
    assert bool(result.role_terminal_gate) == expected


@pytest.mark.parametrize("changes", [{"is_seed": True}, {"depth": 4},
                                    {"in_kzt": 0}, {"depth": np.nan}])
def test_terminal_requires_observed_nonseed_interior_inflow(changes):
    row = feature_row(out_deg=0, out_kzt=0, pass_through=0, **changes)
    result = assign_roles(pd.DataFrame([row])).iloc[0]
    assert not result.role_terminal_gate
    assert result.role != "terminal"


@pytest.mark.parametrize("fanout,eligible", [(1.0, True), (0.999999, False)])
def test_exact_role_evidence_threshold_and_peripheral_fallback(fanout, eligible):
    rows = [feature_row(gid=i, in_deg=0, in_tx=0, in_kzt=0, out_deg=i+2,
                        out_tx=i+2, out_kzt=i+2, cross_cluster_out_deg=i+2,
                        seed_reach_count=1, betweenness=0, pagerank=1,
                        cross_cluster_degree=0, fanout_share=fanout) for i in range(20)]
    result = assign_roles(pd.DataFrame(rows)).set_index("gid").loc[9]
    assert bool(result.role_distributor_eligible) == eligible
    assert result.role == ("distributor" if eligible else "peripheral")
    assert result.role_score == pytest.approx(0.55 if eligible else 0.5499999)
    assert "0.55" in result.role_rule


@pytest.mark.parametrize("rank,expected", [(9, True), (8, False)])
def test_coordinator_requires_two_signals_at_ninetieth_percentile(rank, expected):
    rows = [feature_row(gid=i, seed_reach_count=i+1, betweenness=i+1,
                        pagerank=1, cross_cluster_degree=0) for i in range(10)]
    result = assign_roles(pd.DataFrame(rows)).set_index("gid").loc[rank-1]
    assert bool(result.role_coordinator_gate) == expected


def test_coordinator_wins_exact_tie_and_isolated_singleton_is_peripheral():
    row = feature_row(in_deg=2, out_deg=2, seed_reach_count=2,
                      betweenness=1, pagerank=1, cross_cluster_degree=1, retention=1,
                      balance_score=1)
    result = assign_roles(pd.DataFrame([row])).iloc[0]
    assert result.role_coordinator_score == result.role_consolidator_score == result.role_distributor_score == 1
    assert result.role == "coordinator"
    isolated = feature_row(in_deg=0, out_deg=0, in_tx=0, out_tx=0, in_kzt=0, out_kzt=0,
                          is_seed=True, depth=0, seed_reach_count=1, pagerank=1,
                          betweenness=0, cross_cluster_degree=0)
    result = assign_roles(pd.DataFrame([isolated])).iloc[0]
    assert result.role == "peripheral"
    assert result.role_score == 0
    assert not result.role_coordinator_gate
    assert "No structural gate passed" in result.role_rule_details


def test_consolidator_then_distributor_win_remaining_exact_ties():
    row = feature_row(in_deg=2, out_deg=2, retention=1, balance_score=1)
    result = assign_roles(pd.DataFrame([row])).iloc[0]
    assert result.role_consolidator_score == result.role_distributor_score == result.role_transit_score == 1
    assert result.role == "consolidator"
    row["in_deg"] = 1
    result = assign_roles(pd.DataFrame([row])).iloc[0]
    assert result.role_distributor_score == result.role_transit_score == 1
    assert result.role == "distributor"


@pytest.mark.parametrize("changes", [{"is_seed": True}, {"depth": 4},
                                    {"boundary_censored": True}, {"truncated_by_depth": True}])
def test_unavailable_observation_ratios_are_ignored_and_weights_renormalized(changes):
    row = feature_row(in_deg=2, out_deg=2, seed_reach_count=2, retention=0.3,
                      balance_score=0.3, **changes)
    result = assign_roles(pd.DataFrame([row])).iloc[0]
    assert result.role_consolidator_available_weight == pytest.approx(0.9)
    assert result.role_distributor_available_weight == pytest.approx(0.9)
    assert result.role_consolidator_score == pytest.approx(1)
    assert result.role_distributor_score == pytest.approx(1)
    assert not result.decision_pass_through_valid
    assert not result.decision_relay_2d_ratio_valid
    assert not result.role_transit_gate


def test_explicit_invalid_flags_are_not_overridden_by_ratio_fallback():
    row = feature_row(retention=0.3, retention_valid=False, balance_score_available=False,
                      relay_2d_ratio_valid=False, same_day_flow_ratio_available=False)
    result = assign_roles(pd.DataFrame([row])).iloc[0]
    for name in ("retention", "balance_score", "relay_2d_ratio", "same_day_flow_ratio"):
        assert not result[f"decision_{name}_valid"]
        assert pd.isna(result[f"decision_{name}"])


@pytest.mark.parametrize("source_flag", ["pass_through_valid", "pass_through_available"])
@pytest.mark.parametrize("invalid", [False, np.nan, pd.NA])
def test_invalid_pass_through_also_blocks_precomputed_retention_and_balance(source_flag, invalid):
    row = feature_row(**{
        source_flag: invalid, "retention": 1.0, "balance_score": 1.0,
        "retention_valid": True, "balance_score_available": True,
    })
    result = assign_roles(pd.DataFrame([row])).iloc[0]
    for metric in ("pass_through", "retention", "balance_score"):
        assert not result[f"decision_{metric}_valid"]
        assert pd.isna(result[f"decision_{metric}"])
    assert result.role_consolidator_available_weight == pytest.approx(0.7)
    assert result.role_transit_available_weight == pytest.approx(0.55)


def test_valid_pass_through_preserves_precomputed_or_fallback_derivatives():
    rows = [feature_row(gid=1, pass_through_valid=True, pass_through=0.8),
            feature_row(gid=2, pass_through_valid=True, pass_through=0.8,
                        retention=0.2, balance_score=0.8)]
    result = assign_roles(pd.DataFrame(rows))
    assert result.decision_retention_valid.all()
    assert result.decision_balance_score_valid.all()
    np.testing.assert_allclose(result.decision_retention, [0.2, 0.2])
    np.testing.assert_allclose(result.decision_balance_score, [0.8, 0.8])


@pytest.mark.parametrize("metric,producer_flag", [
    ("relay_2d_ratio", "relay_2d_valid"),
    ("same_day_flow_ratio", "same_day_flow_valid"),
    ("fanin_share", "flow_share_valid"),
    ("fanout_share", "flow_share_valid"),
])
@pytest.mark.parametrize("invalid", [False, np.nan, pd.NA])
def test_producer_validity_flags_override_numeric_ratios_and_true_aliases(metric, producer_flag, invalid):
    frame = pd.DataFrame([feature_row(**{
        metric: 1.0, producer_flag: invalid, f"{metric}_valid": True,
        f"{metric}_available": True,
    })])
    assert observed_ratio(frame, metric).isna().all()
    if metric != "fanin_share":
        result = assign_roles(frame).iloc[0]
        assert not result[f"decision_{metric}_valid"]
        assert pd.isna(result[f"decision_{metric}"])


@pytest.mark.parametrize("metric,producer_flag", [
    ("relay_2d_ratio", "relay_2d_valid"),
    ("same_day_flow_ratio", "same_day_flow_valid"),
    ("fanout_share", "flow_share_valid"),
])
def test_true_producer_flag_preserves_values_but_cannot_override_false_legacy_flag(metric, producer_flag):
    frame = pd.DataFrame([feature_row(**{metric: 0.75, producer_flag: True})])
    assert observed_ratio(frame, metric).iloc[0] == 0.75
    frame[f"{metric}_available"] = False
    assert observed_ratio(frame, metric).isna().all()


@pytest.mark.parametrize("invalid", [False, np.nan, pd.NA])
def test_priority_honors_feature_contract_temporal_flags(invalid):
    features = pd.DataFrame([feature_row(
        relay_2d_ratio=1.0, relay_2d_valid=invalid,
        same_day_flow_ratio=1.0, same_day_flow_valid=invalid,
    )])
    roles = pd.DataFrame({"gid": [1], "role": ["peripheral"], "role_score": [0.0]})
    result = score_priority(features, roles).iloc[0]
    assert not result.priority_temporal_signal_available
    assert pd.isna(result.priority_temporal_signal_value)
    assert result.priority_temporal_signal_contribution == 0.0


def test_missing_balance_is_derived_from_observed_pass_through():
    result = assign_roles(pd.DataFrame([feature_row(pass_through=1, balance_score=np.nan)])).iloc[0]
    assert result.decision_balance_score == 1
    assert result.decision_retention == 0


def test_empty_and_gid_indexed_frames_keep_api_contract():
    empty = pd.DataFrame({"gid": pd.Series(dtype="int64")})
    result = assign_roles(empty)
    assert result.empty
    assert {"gid", "role", "role_score", "role_rule", "role_rule_details"} <= set(result.columns)
    assert score_priority(empty, result).empty
    indexed = pd.DataFrame([feature_row()]).set_index("gid")
    assert assign_roles(indexed).gid.tolist() == [1]


def test_priority_contributions_sum_and_explain_largest_ranking_terms():
    features = pd.DataFrame([feature_row(gid=1, seed_reach_count=5, betweenness=0.7,
                                        pagerank=0.2, cross_cluster_degree=3,
                                        peer_anomaly_score=0.4),
                             feature_row(gid=2, seed_reach_count=1, betweenness=0.1,
                                         pagerank=0.1, cross_cluster_degree=0)])
    roles = pd.DataFrame({"gid": [1, 2], "role": ["transit", "peripheral"], "role_score": [0.8, 0.0]})
    result = score_priority(features, roles)
    contributions = result[[f"priority_{name}_contribution" for name in WEIGHTS]]
    np.testing.assert_allclose(contributions.sum(axis=1), result.priority_score, rtol=0, atol=1e-14)
    reason = result.loc[0, "priority_explanation"]
    assert "transit rule strength 0.80 (+0.200)" in reason
    assert "reachable from 5 seeds (+0.200)" in reason
    assert "path centrality at percentile 100.0 (+0.150)" in reason
    assert "in_deg" not in reason and "betweenness" not in reason
    repeated = score_priority(features, roles)
    pd.testing.assert_frame_equal(result, repeated)


def test_priority_unavailable_temporal_inputs_have_zero_contribution_and_no_renormalization():
    features = pd.DataFrame([feature_row(is_seed=True, relay_2d_ratio=1.0,
                                        same_day_flow_ratio=1.0, peer_anomaly_score=np.inf)])
    roles = pd.DataFrame({"gid": [1], "role": ["peripheral"], "role_score": [0.0]})
    result = score_priority(features, roles).iloc[0]
    assert not result.priority_temporal_signal_available
    assert result.priority_temporal_signal_contribution == 0
    assert result.priority_peer_anomaly_score_contribution == 0
    assert result.priority_total_kzt_contribution == 0.10


def test_priority_without_observations_explains_zero_honestly():
    features = pd.DataFrame({"gid": [1]})
    roles = pd.DataFrame({"gid": [1], "role": ["peripheral"], "role_score": [0.0]})
    result = score_priority(features, roles).iloc[0]
    assert result.priority_score == 0
    assert "no positive contribution" in result.priority_explanation
