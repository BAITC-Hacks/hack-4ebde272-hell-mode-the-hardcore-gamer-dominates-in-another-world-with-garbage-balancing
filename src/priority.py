"""Decomposed investigation priority, distinct from a role hypothesis."""
from __future__ import annotations

import pandas as pd

from .roles import numeric_column, observed_ratio, percentile

WEIGHTS = {
    "role_strength": 0.25, "seed_reach_count": 0.20, "betweenness": 0.15,
    "pagerank": 0.10, "total_kzt": 0.10, "cross_cluster_degree": 0.10,
    "temporal_signal": 0.05, "peer_anomaly_score": 0.05,
}


def _ranking_explanation(row: pd.Series) -> str:
    """Explain the three largest additive contributions; ties use weight order."""
    labels = {
        "role_strength": lambda: f"{row['role']} rule strength {row['priority_role_strength_value']:.2f}",
        "seed_reach_count": lambda: f"reachable from {row['seed_reach_count']:.0f} seeds",
        "betweenness": lambda: f"path centrality at percentile {100 * row['priority_betweenness_value']:.1f}",
        "pagerank": lambda: f"network importance at percentile {100 * row['priority_pagerank_value']:.1f}",
        "total_kzt": lambda: f"observed incoming plus outgoing amount {row['total_kzt']:,.0f} KZT",
        "cross_cluster_degree": lambda: f"{row['cross_cluster_degree']:.0f} cross-community relationships",
        "temporal_signal": lambda: f"date overlap {100 * row['priority_temporal_signal_value']:.1f}%",
        "peer_anomaly_score": lambda: f"depth-peer anomaly index {row['priority_peer_anomaly_score_value']:.2f}",
    }
    ordered = sorted(WEIGHTS, key=lambda name: -row[f"priority_{name}_contribution"])
    positive = [name for name in ordered if row[f"priority_{name}_contribution"] > 0][:3]
    if not positive:
        return "Review priority 0.000: no positive contribution from available observations."
    terms = [f"{labels[name]()} (+{row[f'priority_{name}_contribution']:.3f})" for name in positive]
    return f"Review priority {row['priority_score']:.3f}: " + "; ".join(terms) + "."


def score_priority(features: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Add fixed-weight contributions and independent readable ranking reasons.

    Missing terms contribute zero, with explicit availability flags. The fixed
    priority weights are not renormalized: lack of observation adds no priority.
    This differs from within-role evidence weight renormalization.
    """
    features = features.reset_index() if "gid" not in features and features.index.name == "gid" else features
    if {"role", "role_score"}.issubset(features.columns):
        df = features.copy()
    else:
        df = features.merge(roles, on="gid", how="left", validate="one_to_one")
    terms = {"role_strength": numeric_column(df, "role_score").clip(0, 1)}
    for name in ("seed_reach_count", "betweenness", "pagerank", "total_kzt", "cross_cluster_degree"):
        terms[name] = percentile(numeric_column(df, name))
    relay = observed_ratio(df, "relay_2d_ratio")
    same = observed_ratio(df, "same_day_flow_ratio")
    terms["temporal_signal"] = pd.concat([relay, same], axis=1).max(axis=1, skipna=True).clip(0, 1)
    terms["peer_anomaly_score"] = numeric_column(df, "peer_anomaly_score").clip(0, 1)
    for name, value in terms.items():
        df[f"priority_{name}_available"] = value.notna()
        df[f"priority_{name}_value"] = value
        df[f"priority_{name}_contribution"] = value.fillna(0) * WEIGHTS[name]
    df["priority_score"] = sum(df[f"priority_{name}_contribution"] for name in WEIGHTS).clip(0, 1)
    df["priority_explanation"] = pd.Series(
        [_ranking_explanation(row) for _, row in df.iterrows()], index=df.index, dtype="object")
    return df
