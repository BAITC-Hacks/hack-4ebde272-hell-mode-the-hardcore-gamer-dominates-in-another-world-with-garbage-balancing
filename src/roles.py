"""Interpretable investigation hypotheses and inspectable decision rules.

Scores measure heuristic evidence strength, never calibrated probabilities.
The formulas and observation policy are in documentation/decision-rules.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .graph_features import percentile_rank

ROLES = ("consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral")
TIE_ORDER = {role: index for index, role in enumerate(
    ("coordinator", "consolidator", "distributor", "transit", "terminal", "peripheral"))}
EVIDENCE_THRESHOLD = 0.55
PERCENTILE_FEATURES = (
    "in_deg", "in_tx", "in_kzt", "out_deg", "out_tx", "out_kzt", "seed_reach_count",
    "betweenness", "pagerank", "cross_cluster_out_deg", "cross_cluster_degree", "total_kzt",
)


def numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Keep unavailable observations missing; do not turn them into measured zero."""
    values = frame[name] if name in frame else pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(values, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def percentile(series: pd.Series) -> pd.Series:
    """Use Member 1's average-tie ranks, singleton=1 and missing=missing.

    Nonfinite values are unavailable in decision scoring and are removed before
    calling the shared helper. Callers explicitly choose their missing policy.
    """
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return percentile_rank(clean)


def _flag(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[name].fillna(default).astype(bool)


def ratio_validity_flags(name: str) -> tuple[str, ...]:
    """List producer and legacy validity flags that must all allow a ratio.

    The feature contract uses shared/short names for temporal and flow-share
    flags. Retain generic aliases for existing callers; an explicit false or
    missing value in any present flag takes precedence over a true alias.
    """
    producer_flags = {
        "relay_2d_ratio": ("relay_2d_valid",),
        "same_day_flow_ratio": ("same_day_flow_valid",),
        "fanin_share": ("flow_share_valid",),
        "fanout_share": ("flow_share_valid",),
        # These are transformations of pass-through, not independent measured
        # quantities. Precomputed values cannot outlive their source validity.
        "retention": ("pass_through_valid", "pass_through_available"),
        "balance_score": ("pass_through_valid", "pass_through_available"),
    }
    return (*producer_flags.get(name, ()), f"{name}_valid", f"{name}_available")


def observed_ratio(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return ratios only where seed/cutoff and explicit flags allow them.

    Honor the feature contract's ``relay_2d_valid``, ``same_day_flow_valid``
    and ``flow_share_valid``, plus generic legacy validity/availability aliases.
    Missing explicit flags are unavailable. Depth 4 and seed ratios are invalid
    even for older feature data containing artificial zeroes instead of missing.
    """
    valid = ~_flag(frame, "is_seed")
    valid &= ~numeric_column(frame, "depth").ge(4)
    valid &= ~_flag(frame, "boundary_censored") & ~_flag(frame, "truncated_by_depth")
    for flag in ratio_validity_flags(name):
        if flag in frame:
            valid &= _flag(frame, flag)
    values = numeric_column(frame, name).where(valid)
    return values.where(values.ge(0))


def assign_roles(features: pd.DataFrame) -> pd.DataFrame:
    """Return one row per gid with role strength and every evaluated rule.

    Existing callers may select gid/role/role_score. Extra columns support the
    auxiliary node_features.parquet artifact and analyst explanation of rules.
    """
    df = features.copy()
    if "gid" not in df and df.index.name == "gid":
        df = df.reset_index()
    df = df.reset_index(drop=True)
    if "gid" not in df:
        raise ValueError("Role features must contain a gid column or gid index")
    p = {name: percentile(numeric_column(df, name)) for name in PERCENTILE_FEATURES}
    is_seed = _flag(df, "is_seed")
    depth = numeric_column(df, "depth")
    complete_observation = (~is_seed) & depth.lt(4) & ~_flag(df, "boundary_censored") & ~_flag(df, "truncated_by_depth")
    in_deg, out_deg = numeric_column(df, "in_deg"), numeric_column(df, "out_deg")
    in_kzt, out_kzt = numeric_column(df, "in_kzt"), numeric_column(df, "out_kzt")
    reach = numeric_column(df, "seed_reach_count")
    pass_ratio = observed_ratio(df, "pass_through").where(in_kzt.gt(0))
    relay = observed_ratio(df, "relay_2d_ratio").clip(0, 1)
    same_day = observed_ratio(df, "same_day_flow_ratio").clip(0, 1)
    retention = numeric_column(df, "retention").combine_first((1.0 - pass_ratio).clip(0, 1))
    retention = observed_ratio(df.assign(retention=retention), "retention")
    retention = retention.where(in_kzt.gt(0)).clip(0, 1)
    balance = numeric_column(df, "balance_score").combine_first((1.0 - (1.0 - pass_ratio).abs()).clip(0, 1))
    balance = observed_ratio(df.assign(balance_score=balance), "balance_score")
    balance = balance.where(in_kzt.gt(0)).clip(0, 1)
    fanout = observed_ratio(df, "fanout_share").clip(0, 1)
    no_out = (1 - pass_ratio / 0.1).clip(0, 1)
    no_out = no_out.mask(out_deg.eq(0) & out_kzt.eq(0) & in_kzt.gt(0), 1.0)
    no_out = no_out.where(complete_observation)

    components = {
        "consolidator": [(0.30, "incoming relationship percentile", p["in_deg"]),
                         (0.20, "incoming transaction percentile", p["in_tx"]),
                         (0.20, "incoming amount percentile", p["in_kzt"]),
                         (0.20, "seed reach percentile", p["seed_reach_count"]),
                         (0.10, "observed retained share", retention)],
        "distributor": [(0.40, "outgoing relationship percentile", p["out_deg"]),
                        (0.20, "outgoing transaction percentile", p["out_tx"]),
                        (0.20, "outgoing amount percentile", p["out_kzt"]),
                        (0.10, "outgoing relationship share", fanout),
                        (0.10, "outgoing cross-community relationship percentile", p["cross_cluster_out_deg"])],
        "transit": [(0.25, "observed flow balance", balance), (0.20, "two-day date overlap", relay),
                    (0.20, "path centrality percentile", p["betweenness"]),
                    (0.15, "incoming relationship percentile", p["in_deg"]),
                    (0.15, "outgoing relationship percentile", p["out_deg"]),
                    (0.05, "same-day outgoing amount share", same_day)],
        "coordinator": [(0.25, "seed reach percentile", p["seed_reach_count"]),
                        (0.25, "path centrality percentile", p["betweenness"]),
                        (0.15, "network importance percentile", p["pagerank"]),
                        (0.15, "cross-community relationship percentile", p["cross_cluster_degree"]),
                        (0.10, "incoming relationship percentile", p["in_deg"]),
                        (0.10, "outgoing relationship percentile", p["out_deg"])],
        "terminal": [(0.50, "little observed outgoing activity", no_out),
                     (0.20, "observed retained share", retention),
                     (0.15, "incoming amount percentile", p["in_kzt"]),
                     (0.10, "incoming transaction percentile", p["in_tx"]),
                     (0.05, "incoming relationship percentile", p["in_deg"])],
    }
    scores, available_weights = {}, {}
    for role, terms in components.items():
        numerator = pd.Series(0.0, index=df.index)
        denominator = pd.Series(0.0, index=df.index)
        for weight, _, values in terms:
            numerator += values.fillna(0) * weight
            denominator += values.notna().astype(float) * weight
        available_weights[role] = denominator
        scores[role] = (numerator / denominator.replace(0, np.nan)).fillna(0).clip(0, 1)

    coordinator_signals = sum((p[name] >= 0.90).astype(int) for name in
                              ("seed_reach_count", "betweenness", "pagerank", "cross_cluster_degree"))
    observed_relationships = in_deg.fillna(0).add(out_deg.fillna(0))
    observed_activity = observed_relationships.gt(0)
    gates = {
        "consolidator": in_deg.ge(2) & (reach.ge(2) | p["in_deg"].ge(0.80)),
        "distributor": out_deg.ge(2),
        "transit": in_deg.gt(0) & out_deg.gt(0) & (pass_ratio.between(0.5, 1.5) | relay.ge(0.5)),
        "coordinator": coordinator_signals.ge(2) & observed_activity,
        "terminal": complete_observation & in_deg.gt(0) & in_kzt.gt(0)
                    & out_kzt.ge(0) & out_kzt.le(0.10 * in_kzt)
                    & (out_deg.eq(0) | pass_ratio.le(0.10)),
    }
    eligible = {role: gate & scores[role].ge(EVIDENCE_THRESHOLD) for role, gate in gates.items()}

    def fmt(value):
        return "unavailable" if pd.isna(value) else f"{float(value):.6g}"

    def gate_reason(role, i):
        if role == "consolidator":
            return (f"incoming relationships {fmt(in_deg[i])} >= 2; seed reach {fmt(reach[i])} >= 2 "
                    f"or incoming relationship percentile {fmt(p['in_deg'][i])} >= 0.80")
        if role == "distributor":
            return f"outgoing relationships {fmt(out_deg[i])} >= 2"
        if role == "transit":
            return (f"incoming/outgoing relationships {fmt(in_deg[i])}/{fmt(out_deg[i])} > 0; "
                    f"observed outgoing/incoming amount {fmt(pass_ratio[i])} in [0.5, 1.5] "
                    f"or two-day date overlap {fmt(relay[i])} >= 0.5")
        if role == "coordinator":
            return (f"{coordinator_signals[i]} of 4 structural percentiles >= 0.90; requires >= 2; "
                    f"observed relationships {fmt(observed_relationships[i])} > 0")
        return (f"seed={int(is_seed[i])} must be 0; depth {fmt(depth[i])} < 4; "
                f"incoming relationships {fmt(in_deg[i])} > 0; observed incoming amount {fmt(in_kzt[i])} > 0; "
                f"observed outgoing amount {fmt(out_kzt[i])} <= 0.10 * incoming; "
                f"outgoing relationships {fmt(out_deg[i])} = 0 or observed outgoing/incoming {fmt(pass_ratio[i])} <= 0.10")

    selected, strengths, best_roles, best_scores, rules, details = [], [], [], [], [], []
    for i in df.index:
        gated = sorted((role for role in gates if bool(gates[role][i])),
                       key=lambda role: (-float(scores[role][i]), TIE_ORDER[role]))
        winners = [role for role in gated if bool(eligible[role][i])]
        best = gated[0] if gated else None
        role = winners[0] if winners else "peripheral"
        # Fallback considers candidates that pass a structural gate. An isolate
        # has no such candidate, and therefore no positive role evidence.
        strength = float(scores[role][i]) if winners else (float(scores[best][i]) if best else 0.0)
        selected.append(role)
        strengths.append(strength)
        best_roles.append(best or "none")
        best_scores.append(float(scores[best][i]) if best else 0.0)
        if role == "peripheral":
            rules.append("Peripheral: no structurally eligible role reaches evidence threshold 0.55.")
            if best:
                details.append(f"Strongest eligible structure: {best}; score {strength:.6f} < 0.55. " + gate_reason(best, i))
            else:
                details.append(f"No structural gate passed; observed incoming/outgoing relationships "
                               f"{fmt(in_deg[i])}/{fmt(out_deg[i])}; role strength 0.")
        else:
            rules.append(f"{role.capitalize()}: structural gate passed and score {strength:.6f} >= 0.55; "
                         "highest eligible score; ties use documented role order.")
            terms = [f"{weight:.2f} * {label}={fmt(values[i])}" for weight, label, values in components[role]]
            details.append(gate_reason(role, i) + "; score = sum(available weighted values) / "
                           f"{available_weights[role][i]:.2f} available weight; " + "; ".join(terms))

    result = pd.DataFrame({"gid": df["gid"], "role": pd.Series(selected, dtype="object"),
                           "role_score": pd.Series(strengths, dtype=float),
                           "role_rule": pd.Series(rules, dtype="object"),
                           "role_rule_details": pd.Series(details, dtype="object"),
                           "role_best_candidate": pd.Series(best_roles, dtype="object"),
                           "role_best_candidate_score": pd.Series(best_scores, dtype=float)})
    for role in components:
        result[f"role_{role}_score"] = scores[role]
        result[f"role_{role}_gate"] = gates[role]
        result[f"role_{role}_eligible"] = eligible[role]
        result[f"role_{role}_available_weight"] = available_weights[role]
        result[f"role_{role}_gate_reason"] = pd.Series([gate_reason(role, i) for i in df.index], dtype="object")
    for name, values in (("pass_through", pass_ratio), ("retention", retention), ("balance_score", balance),
                         ("fanout_share", fanout), ("relay_2d_ratio", relay), ("same_day_flow_ratio", same_day)):
        result[f"decision_{name}_valid"] = values.notna()
        result[f"decision_{name}"] = values
    for name, values in p.items():
        result[f"decision_{name}_percentile"] = values
    return result
