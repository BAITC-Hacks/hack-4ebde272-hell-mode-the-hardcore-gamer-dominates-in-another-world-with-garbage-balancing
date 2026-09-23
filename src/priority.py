"""Decomposed investigation priority, distinct from a role hypothesis."""
from __future__ import annotations

import math

import pandas as pd

from .roles import numeric_column, observed_ratio, percentile

WEIGHTS = {
    "role_strength": 0.25, "seed_reach_count": 0.20, "betweenness": 0.15,
    "pagerank": 0.10, "total_kzt": 0.10, "cross_cluster_degree": 0.10,
    "temporal_signal": 0.05, "peer_anomaly_score": 0.05,
}


def _number(row: pd.Series, name: str) -> float | None:
    """Read optional supporting observations without inventing missing values."""
    try:
        value = float(row.get(name, float("nan")))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _amount(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _accounts(value: float) -> str:
    return f"{value:g} {'account' if value == 1 else 'accounts'}"


def _role_reason(row: pd.Series) -> str:
    """Translate the assigned rule into a short clause with observed support."""
    role = row.get("role")
    incoming, outgoing = _number(row, "in_deg"), _number(row, "out_deg")
    if role == "consolidator":
        support = f", with observed payments from {_accounts(incoming)}" if incoming is not None else ""
        return "matches the collection review rule" + support
    if role == "distributor":
        support = f", with observed payments to {_accounts(outgoing)}" if outgoing is not None else ""
        return "matches the distribution review rule" + support
    if role == "transit":
        support = (f", with observed payments from {_accounts(incoming)} and to {_accounts(outgoing)}"
                   if incoming is not None and outgoing is not None else "")
        return "matches the transit review rule" + support
    if role == "terminal":
        received, sent = _number(row, "in_kzt"), _number(row, "out_kzt")
        support = (f", with {_amount(sent)} KZT sent against {_amount(received)} KZT received in the sample"
                   if received is not None and sent is not None else "")
        return "matches the low-outgoing-flow review rule" + support
    if role == "coordinator":
        support = (f" across {incoming + outgoing:g} observed directed relationships"
                   if incoming is not None and outgoing is not None else "")
        return "matches the coordination review rule" + support
    if role == "peripheral":
        return "partly matches a review rule, below the threshold for a stronger role"
    return "matches its assigned review rule"


def _seed_reason(row: pd.Series) -> str:
    count = _number(row, "seed_reach_count")
    seed_flag = row.get("is_seed", False)
    is_seed = False if pd.isna(seed_flag) else bool(seed_flag)
    if count is None:
        return "has an available starting-account reachability signal"
    if count == 0:
        return "is reachable from 0 starting case accounts in the observed graph, but tied zero values still affect its rank"
    if is_seed and count == 1:
        return "is a starting case account with 0 other starting accounts observed upstream within four hops"
    if is_seed:
        return (f"is reachable from {_accounts(count - 1)} that started the case within four directed hops, "
                "in addition to reaching itself")
    return f"is reachable from {count:g} starting case {'account' if count == 1 else 'accounts'} within four directed hops"


def _pagerank_reason(row: pd.Series) -> str:
    incoming = _number(row, "in_deg")
    if incoming == 0:
        return "has 0 observed incoming relationships, so its network ranking comes from the baseline"
    if incoming is not None:
        return f"has incoming links from {_accounts(incoming)} that contribute to its network ranking"
    return "has an amount-weighted network ranking among the observed accounts"


def _peer_reason(row: pd.Series) -> str:
    depth = _number(row, "depth")
    group = f"accounts at sampling depth {depth:g}" if depth is not None else "accounts at the same sampling depth"
    metrics = {
        "in_deg": ("in_deg", "incoming relationships"),
        "out_deg": ("out_deg", "outgoing relationships"),
        "log1p_in_kzt": ("in_kzt", "KZT received"),
        "log1p_out_kzt": ("out_kzt", "KZT sent"),
        "in_tx": ("in_tx", "incoming transfers"),
        "out_tx": ("out_tx", "outgoing transfers"),
    }
    available = [(name, _number(row, f"peer_anomaly_{name}")) for name in metrics]
    available = [(name, deviation) for name, deviation in available if deviation is not None and deviation > 0]
    if available:
        name, _ = max(available, key=lambda item: item[1])
        field, label = metrics[name]
        value = _number(row, field)
        if value is not None:
            return f"has {_amount(value)} {label}, differing from {group}"
    return f"differs in observed relationships, transfer counts or amounts from {group}"


def _with_numeric_support(explanation: str, row: pd.Series) -> str:
    """Keep the export's numeric-support contract without fabricating counts.

    Ordinary reasons already contain the selected counts or amounts. Degenerate
    ties and incomplete caller-provided feature rows may not: prefer actual
    relationship counts, then an observed amount, and finally the computed score.
    """
    if any(character.isdigit() for character in explanation):
        return explanation
    incoming, outgoing = _number(row, "in_deg"), _number(row, "out_deg")
    if incoming is not None and outgoing is not None:
        return explanation + f" It has {incoming:g} observed incoming and {outgoing:g} outgoing relationships."
    for value, direction in ((incoming, "incoming"), (outgoing, "outgoing")):
        if value is not None:
            return explanation + f" It has {value:g} observed {direction} {'relationship' if value == 1 else 'relationships'}."
    amount = _number(row, "total_kzt")
    if amount is not None:
        return explanation + f" Its combined observed incoming and outgoing transfers total {_amount(amount)} KZT."
    return explanation + f" Its review priority is {row['priority_score']:.3f}."


def _ranking_explanation(row: pd.Series, temporal_source: str = "relay") -> str:
    """Explain the three leading contributions as observed reasons, not arithmetic.

    Contribution order and ties are unchanged. Shortest routes refer to the
    graph's amount-derived distance, not traced funds. Positive percentile scores
    for tied zeros are disclosed rather than described as observed activity.
    """
    def between_reason():
        if _number(row, "betweenness") == 0:
            return "lies on 0 shortest routes between other accounts, but tied zero values still affect its rank"
        return "lies on shortest directed routes between other accounts in the observed graph"

    def cross_reason():
        count = _number(row, "cross_cluster_degree")
        if count == 0:
            return "has 0 relationships crossing community boundaries, but tied zero values still affect its rank"
        return f"has {count:g} directed {'relationship' if count == 1 else 'relationships'} crossing community boundaries"

    def temporal_reason():
        share = f"{100 * row['priority_temporal_signal_value']:.1f}".rstrip("0").rstrip(".")
        if temporal_source == "relay":
            return f"has outgoing activity on or up to two days after {share}% of eligible incoming dates"
        return f"sends {share}% of its observed outgoing amount on dates with incoming activity"

    labels = {
        "role_strength": lambda: _role_reason(row),
        "seed_reach_count": lambda: _seed_reason(row),
        "betweenness": between_reason,
        "pagerank": lambda: _pagerank_reason(row),
        "total_kzt": lambda: f"has {_amount(row['total_kzt'])} KZT in combined observed incoming and outgoing transfers",
        "cross_cluster_degree": cross_reason,
        "temporal_signal": temporal_reason,
        "peer_anomaly_score": lambda: _peer_reason(row),
    }
    ordered = sorted(WEIGHTS, key=lambda name: -row[f"priority_{name}_contribution"])
    positive = [name for name in ordered if row[f"priority_{name}_contribution"] > 0][:3]
    if not positive:
        return "No observed signal raises this account's review priority above 0."
    terms = [labels[name]() for name in positive]
    first = f"It {terms[0]}."
    explanation = first if len(terms) == 1 else first + " It " + " and ".join(terms[1:]) + "."
    return _with_numeric_support(explanation, row)


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
    # Describe the ratio that actually supplied max(relay, same-day). On an
    # exact tie choose relay consistently, without changing the numeric score.
    temporal_sources = ["relay" if pd.notna(r) and (pd.isna(s) or r >= s) else "same_day"
                        for r, s in zip(relay, same)]
    df["priority_explanation"] = pd.Series(
        [_ranking_explanation(row, source)
         for (_, row), source in zip(df.iterrows(), temporal_sources)], index=df.index, dtype="object")
    return df
