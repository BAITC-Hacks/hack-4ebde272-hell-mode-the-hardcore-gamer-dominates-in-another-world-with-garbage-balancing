import pandas as pd

from src.roles import assign_roles


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
    assert result.role in {"distributor", "peripheral"}
    assert 0 <= result.role_score <= 1
