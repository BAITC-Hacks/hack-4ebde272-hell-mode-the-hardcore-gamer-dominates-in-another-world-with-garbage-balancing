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
    # Independent expectations for the original fixed weights: presentation
    # changes must not change priorities or the role supplied by the engine.
    np.testing.assert_allclose(result.priority_score, [0.895, 0.4], rtol=0, atol=1e-14)
    assert result.role.tolist() == ["transit", "peripheral"]
    assert result.role_score.tolist() == [0.8, 0.0]
    reason = result.loc[0, "priority_explanation"]
    assert reason == (
        "It matches the transit review rule, with observed payments from 1 account and to 1 account. "
        "It is reachable from 5 starting case accounts within four directed hops and "
        "lies on shortest directed routes between other accounts in the observed graph."
    )
    assert all(term not in reason for term in ("in_deg", "betweenness", "percentile", "(+", "rule strength"))
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
    assert result.priority_explanation == "No observed signal raises this account's review priority above 0."


def explain_single_priority_signal(**observations):
    """Score one feature record without silently adding other ranking signals."""
    frame = pd.DataFrame([{"gid": 1, "depth": 2, **observations}])
    roles = pd.DataFrame({"gid": [1], "role": ["peripheral"], "role_score": [0.0]})
    result = score_priority(frame, roles).iloc[0]
    assert any(character.isdigit() for character in result.priority_explanation)
    return result


@pytest.mark.parametrize("observations,expected,score", [
    ({"seed_reach_count": 9},
     "It is reachable from 9 starting case accounts within four directed hops.", 0.20),
    ({"betweenness": 0.2},
     "It lies on shortest directed routes between other accounts in the observed graph. Its review priority is 0.150.", 0.15),
    ({"pagerank": 0.2, "in_deg": 3},
     "It has incoming links from 3 accounts that contribute to its network ranking.", 0.10),
    ({"total_kzt": 123456.78},
     "It has 123,456.78 KZT in combined observed incoming and outgoing transfers.", 0.10),
    ({"cross_cluster_degree": 3},
     "It has 3 directed relationships crossing community boundaries.", 0.10),
    ({"peer_anomaly_score": 0.6},
     "It differs in observed relationships, transfer counts or amounts from accounts at sampling depth 2.", 0.03),
    ({"peer_anomaly_score": 0.6, "peer_anomaly_in_deg": 0.9,
      "peer_anomaly_out_deg": 0.1, "in_deg": 12, "out_deg": 2},
     "It has 12 incoming relationships, differing from accounts at sampling depth 2.", 0.03),
])
def test_priority_text_uses_actual_observations_without_arithmetic(observations, expected, score):
    result = explain_single_priority_signal(**observations)
    assert result.priority_explanation == expected
    assert result.priority_score == pytest.approx(score)


@pytest.mark.parametrize("observations,expected", [
    ({"seed_reach_count": 1, "is_seed": True},
     "It is a starting case account with 0 other starting accounts observed upstream within four hops."),
    ({"seed_reach_count": 3, "is_seed": True},
     "It is reachable from 2 accounts that started the case within four directed hops, in addition to reaching itself."),
    ({"seed_reach_count": 0},
     "It is reachable from 0 starting case accounts in the observed graph, but tied zero values still affect its rank."),
    ({"betweenness": 0},
     "It lies on 0 shortest routes between other accounts, but tied zero values still affect its rank."),
    ({"pagerank": 0.2, "in_deg": 0},
     "It has 0 observed incoming relationships, so its network ranking comes from the baseline."),
    ({"cross_cluster_degree": 0},
     "It has 0 relationships crossing community boundaries, but tied zero values still affect its rank."),
])
def test_relative_scores_do_not_invent_activity_or_other_seed_connections(observations, expected):
    result = explain_single_priority_signal(**observations)
    assert result.priority_score > 0
    assert result.priority_explanation == expected


@pytest.mark.parametrize("relay,same,flags,expected,score", [
    (0.6, 0.2, {},
     "It has outgoing activity on or up to two days after 60% of eligible incoming dates.", 0.03),
    (0.2, 0.6, {},
     "It sends 60% of its observed outgoing amount on dates with incoming activity.", 0.03),
    (0.6, 0.6, {},
     "It has outgoing activity on or up to two days after 60% of eligible incoming dates.", 0.03),
    (1.0, 0.4, {"relay_2d_valid": False},
     "It sends 40% of its observed outgoing amount on dates with incoming activity.", 0.02),
    (0.4, 1.0, {"same_day_flow_valid": False},
     "It has outgoing activity on or up to two days after 40% of eligible incoming dates.", 0.02),
    (1.0, 1.0, {"depth": 4},
     "No observed signal raises this account's review priority above 0.", 0.0),
    (1.0, 1.0, {"is_seed": True},
     "No observed signal raises this account's review priority above 0.", 0.0),
])
def test_temporal_priority_explanation_uses_the_selected_valid_denominator(relay, same, flags, expected, score):
    result = explain_single_priority_signal(
        relay_2d_ratio=relay, same_day_flow_ratio=same, **flags)
    assert result.priority_explanation == expected
    assert result.priority_score == pytest.approx(score)
    assert all(term not in result.priority_explanation for term in ("same money", "passed on", "within hours"))


@pytest.mark.parametrize("role,observations,expected", [
    ("consolidator", {"in_deg": 8},
     "It matches the collection review rule, with observed payments from 8 accounts."),
    ("distributor", {"out_deg": 99},
     "It matches the distribution review rule, with observed payments to 99 accounts."),
    ("terminal", {"in_kzt": 10000, "out_kzt": 500.25},
     "It matches the low-outgoing-flow review rule, with 500.25 KZT sent against 10,000 KZT received in the sample."),
    ("coordinator", {"in_deg": 3, "out_deg": 4},
     "It matches the coordination review rule across 7 observed directed relationships."),
    ("peripheral", {},
     "It partly matches a review rule, below the threshold for a stronger role. Its review priority is 0.100."),
])
def test_priority_role_support_is_readable_and_does_not_change_the_given_role(role, observations, expected):
    frame = pd.DataFrame([{"gid": 1, **observations}])
    strength = 0.4 if role == "peripheral" else 0.8
    roles = pd.DataFrame({"gid": [1], "role": [role], "role_score": [strength]})
    result = score_priority(frame, roles).iloc[0]
    assert result.priority_explanation == expected
    assert result.priority_score == pytest.approx(0.25 * strength)
    assert result.role == role and result.role_score == strength


def test_numeric_support_prefers_actual_relationships_to_score_only_fallback():
    result = explain_single_priority_signal(betweenness=0.2, in_deg=4, out_deg=0)
    assert result.priority_explanation == (
        "It lies on shortest directed routes between other accounts in the observed graph. "
        "It has 4 observed incoming and 0 outgoing relationships."
    )
    assert "0.150" not in result.priority_explanation


def test_supplied_pipeline_validates_numeric_priority_reasons_for_every_node(tmp_path):
    """Catch all-node explanation failures before the viewer's shared setup runs."""
    from pathlib import Path

    import pipeline

    data = Path(__file__).resolve().parents[1] / "data"
    output = tmp_path / "out"
    counts = pipeline.run(data, output)
    assert counts["nodes_roles"] == counts["node_features"] == 2248
    rich = pd.read_parquet(output / "node_features.parquet")
    assert rich.priority_explanation.str.contains(r"\d").all()
    isolates = rich.loc[rich.in_deg.eq(0) & rich.out_deg.eq(0)]
    assert len(isolates) == 19
    assert isolates.priority_explanation.str.contains("0 other starting accounts", regex=False).all()
    assert isolates.priority_explanation.str.contains("0 relationships crossing", regex=False).all()
    top = pd.read_csv(output / "top_nodes.csv", dtype={"gid": "int64"})
    assert top.why.str.contains(r"\d").all()
    reasons = rich.set_index("gid").priority_explanation
    assert top.why.tolist() == reasons.loc[top.gid].tolist()
    assert pipeline.validate_artifacts(
        output, pd.read_parquet(data / "nodes.parquet"), pd.read_parquet(data / "edges.parquet")
    ) == counts
