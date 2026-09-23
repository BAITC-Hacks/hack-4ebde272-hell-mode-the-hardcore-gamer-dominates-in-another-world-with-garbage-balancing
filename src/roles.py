"""Interpretable, percentile-based investigation role hypotheses."""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLES = ("consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral")
TIE_ORDER = {r: i for i, r in enumerate(("coordinator", "consolidator", "distributor",
                                         "transit", "terminal", "peripheral"))}


def percentile(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() <= 1:
        return x.fillna(0.0).clip(0, 1)
    return x.rank(method="average", pct=True).fillna(0.0).clip(0, 1)


def assign_roles(features: pd.DataFrame) -> pd.DataFrame:
    """Return gid, role and evidence-strength scores; these are not probabilities."""
    df = features.copy().reset_index(drop=True)
    n = len(df)
    def col(name, default=0.0):
        if name in df:
            return pd.to_numeric(df[name], errors="coerce")
        ratio_features = {"pass_through", "retention", "relay_2d_ratio", "same_day_flow_ratio", "balance_score"}
        fallback = np.nan if name in ratio_features else default
        return pd.Series(fallback, index=df.index, dtype=float)
    p = {name: percentile(col(name)) for name in (
        "in_deg", "in_tx", "in_kzt", "out_deg", "out_tx", "out_kzt",
        "seed_reach_count", "betweenness", "pagerank", "cross_cluster_out_deg",
        "cross_cluster_degree", "total_kzt")}
    is_seed = df.get("is_seed", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    depth = col("depth").fillna(4)
    in_deg, out_deg = col("in_deg").fillna(0), col("out_deg").fillna(0)
    in_kzt, out_kzt = col("in_kzt").fillna(0), col("out_kzt").fillna(0)
    in_tx = col("in_tx").fillna(0)
    reach = col("seed_reach_count").fillna(0)
    pass_ratio = col("pass_through").where(~is_seed)
    relay = col("relay_2d_ratio").where(~is_seed)
    same_day = col("same_day_flow_ratio").where(~is_seed)
    retention = col("retention").combine_first(1.0 - pass_ratio)
    # Seed-client inflows are censored in the supplied graph, so their ratios are invalid.
    retention = retention.where(~is_seed & (in_kzt > 0) & retention.notna()).clip(0, 1)
    balance = col("balance_score").where(col("balance_score").notna(), retention).where(~is_seed)
    fanout = col("fanout_share")
    no_out = pd.Series(np.where(out_deg.eq(0), 1.0,
                                np.where(pass_ratio.notna(), (1 - pass_ratio / 0.1).clip(0, 1), 0.0)), index=df.index)

    def weighted(components):
        numerator = pd.Series(0.0, index=df.index)
        denominator = pd.Series(0.0, index=df.index)
        for weight, value in components:
            vals = pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0, 1)
            numerator += vals.fillna(0) * weight
            denominator += vals.notna().astype(float) * weight
        return (numerator / denominator.replace(0, np.nan)).fillna(0).clip(0, 1)

    scores = {
        "consolidator": weighted([(0.30,p["in_deg"]),(0.20,p["in_tx"]),(0.20,p["in_kzt"]),
                                   (0.20,p["seed_reach_count"]),(0.10,retention)]),
        "distributor": weighted([(0.40,p["out_deg"]),(0.20,p["out_tx"]),(0.20,p["out_kzt"]),
                                  (0.10,fanout),(0.10,p["cross_cluster_out_deg"])]),
        "transit": weighted([(0.25,balance),(0.20,relay),(0.20,p["betweenness"]),
                              (0.15,p["in_deg"]),(0.15,p["out_deg"]),(0.05,same_day)]),
        "coordinator": weighted([(0.25,p["seed_reach_count"]),(0.25,p["betweenness"]),
                                  (0.15,p["pagerank"]),(0.15,p["cross_cluster_degree"]),
                                  (0.10,p["in_deg"]),(0.10,p["out_deg"])]),
        "terminal": weighted([(0.50,no_out),(0.20,retention),(0.15,p["in_kzt"]),
                              (0.10,p["in_tx"]),(0.05,p["in_deg"])]),
    }
    # Coordinator requires at least two independent strong structural signals.
    coordinator_signals = sum((p[k] >= 0.90).astype(int) for k in
                              ("seed_reach_count", "betweenness", "pagerank", "cross_cluster_degree"))
    transit_eligible = (in_deg > 0) & (out_deg > 0) & (
        (pass_ratio.between(0.5, 1.5, inclusive="both")) | (relay >= 0.5))
    terminal_eligible = (~is_seed) & (depth < 4) & (in_deg > 0) & (
        out_deg.eq(0) | (pass_ratio.notna() & (pass_ratio <= 0.10)))
    eligible = {
        "consolidator": (in_deg >= 2) & ((reach >= 2) | (p["in_deg"] >= 0.80)) & (scores["consolidator"] >= 0.55),
        "distributor": (out_deg >= 2) & (scores["distributor"] >= 0.55),
        "transit": transit_eligible & (scores["transit"] >= 0.55),
        "coordinator": (coordinator_signals >= 2) & (scores["coordinator"] >= 0.55),
        "terminal": terminal_eligible & (scores["terminal"] >= 0.55),
    }
    roles, strengths = [], []
    for i in df.index:
        options = [(float(scores[r].iloc[i]), r) for r in scores if bool(eligible[r].iloc[i])]
        if options:
            score, role = sorted(options, key=lambda x: (-x[0], TIE_ORDER[x[1]]))[0]
        else:
            structural = max((float(v.iloc[i]) for v in scores.values()), default=0.0)
            role, score = "peripheral", min(structural, 0.549999)
        roles.append(role)
        strengths.append(float(np.clip(score, 0, 1)))
    result = pd.DataFrame({"gid": df["gid"], "role": roles, "role_score": strengths})
    result["role_score"] = result.role_score.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
    return result
