"""Decomposed investigation prioritization score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .roles import percentile

WEIGHTS = {
    "role_strength": 0.25, "seed_reach_count": 0.20, "betweenness": 0.15,
    "pagerank": 0.10, "total_kzt": 0.10, "cross_cluster_degree": 0.10,
    "temporal_signal": 0.05, "peer_anomaly_score": 0.05,
}


def score_priority(features: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    if {"role", "role_score"}.issubset(features.columns):
        df = features.copy()
    else:
        df = features.merge(roles[["gid", "role", "role_score"]], on="gid", how="left", validate="one_to_one")
    terms = {"role_strength": pd.to_numeric(df.role_score, errors="coerce").fillna(0).clip(0, 1)}
    for name in ("seed_reach_count", "betweenness", "pagerank", "total_kzt", "cross_cluster_degree"):
        terms[name] = percentile(df[name]) if name in df else pd.Series(0.0, index=df.index)
    relay = pd.to_numeric(df.get("relay_2d_ratio", pd.Series(np.nan, index=df.index)), errors="coerce")
    same = pd.to_numeric(df.get("same_day_flow_ratio", pd.Series(np.nan, index=df.index)), errors="coerce")
    terms["temporal_signal"] = pd.concat([relay, same], axis=1).max(axis=1, skipna=True).clip(0, 1).fillna(0)
    terms["peer_anomaly_score"] = pd.to_numeric(df.get("peer_anomaly_score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0).clip(0, 1)
    for name, value in terms.items():
        df[f"priority_{name}_contribution"] = value * WEIGHTS[name]
    df["priority_score"] = sum(df[f"priority_{name}_contribution"] for name in WEIGHTS).clip(0, 1)
    return df
