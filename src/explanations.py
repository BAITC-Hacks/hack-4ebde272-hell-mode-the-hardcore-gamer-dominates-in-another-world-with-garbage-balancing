"""Readable role hypotheses with numeric evidence and observation limits."""
from __future__ import annotations

import math

import pandas as pd


def _number(value, digits=2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if not math.isfinite(value):
        return "unavailable"
    return f"{value:,.{digits}f}"


def _amount(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if not math.isfinite(value):
        return "unavailable"
    for divisor, suffix in ((1e9, " billion"), (1e6, " million"), (1e3, " thousand")):
        if abs(value) >= divisor:
            return f"{value / divisor:.3g}{suffix} KZT"
    return f"{value:,.2f} KZT"


def _flag(row, name) -> bool:
    value = row.get(name, False)
    return False if pd.isna(value) else bool(value)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _percentile(row, name) -> float:
    for key in (f"decision_{name}_percentile", f"{name}_pct", f"{name}_percentile"):
        value = row.get(key)
        if _finite(value):
            return float(value)
    return float("nan")


def _ratio(row, name) -> float:
    for flag in (f"decision_{name}_valid", f"{name}_valid", f"{name}_available"):
        if flag in row and not _flag(row, flag):
            return float("nan")
    value = row.get(f"decision_{name}", row.get(name))
    return float(value) if _finite(value) and float(value) >= 0 else float("nan")


def _bounded(body: str, warnings: list[str]) -> str:
    suffix = " ".join(warnings)
    budget = 200 - len(suffix) - (1 if suffix else 0)
    if len(body) > budget:
        body = body[:budget - 1].rsplit(" ", 1)[0].rstrip(" ,;.") + "."
    return body + (" " + suffix if suffix else "")


def evidence_for(row: pd.Series) -> str:
    """Explain the assigned role first; append censoring without hiding it."""
    role = str(row.get("role", "peripheral"))
    depth = row.get("depth")
    boundary = (_finite(depth) and float(depth) >= 4) or _flag(row, "boundary_censored") or _flag(row, "truncated_by_depth")
    seed = _flag(row, "is_seed")
    incoming = _number(row.get("in_deg"), 0)
    outgoing = _number(row.get("out_deg"), 0)
    seed_reach = _number(row.get("seed_reach_count"), 0)
    warnings = []
    if boundary:
        warnings.append("Beyond hop 4: activity unobserved.")
    if seed:
        warnings.append("Seed inflows incomplete.")

    if role == "consolidator":
        body = (f"Collection pattern: {incoming} incoming counterparties; "
                f"{_amount(row.get('in_kzt'))} received; reachable from {seed_reach} seeds.")
    elif role == "distributor":
        body = (f"Distribution pattern: {outgoing} outgoing counterparties; "
                f"{_number(row.get('out_tx'), 0)} transfers; {_amount(row.get('out_kzt'))} sent; "
                f"{_number(row.get('cross_cluster_out_deg'), 0)} links cross communities.")
    elif role == "coordinator":
        signals = [("seed reach", "seed_reach_count"), ("path bridging", "betweenness"),
                   ("network influence", "pagerank"), ("community links", "cross_cluster_degree")]
        ranked = sorted(enumerate(signals), key=lambda item: (
            -(_percentile(row, item[1][1]) if _finite(_percentile(row, item[1][1])) else -1), item[0]))
        strongest = [f"{label} at {_number(_percentile(row, name) * 100, 0)}th percentile"
                     for _, (label, name) in ranked[:2] if _finite(_percentile(row, name))]
        body = f"Coordination pattern: reachable from {seed_reach} seeds; " + "; ".join(strongest) + "."
        if not strongest:
            body = (f"Coordination hypothesis: {seed_reach} seed routes, "
                    f"{_number(row.get('cross_cluster_degree'), 0)} cross-community links; percentiles unavailable.")
    elif role == "transit":
        ratio = _ratio(row, "pass_through")
        relay = _ratio(row, "relay_2d_ratio")
        body = f"Relay pattern: {incoming} incoming, {outgoing} outgoing counterparties; "
        if not (seed or boundary) and _finite(ratio):
            body += f"observed outflow/inflow={_number(ratio)}; "
        if not (seed or boundary) and _finite(relay):
            body += f"2-day date overlap={_number(float(relay) * 100, 0)}% (date-only)."
        else:
            body += "2-day timing unavailable."
    elif role == "terminal":
        body = (f"Observed endpoint: {incoming} incoming counterparties; "
                f"{_amount(row.get('in_kzt'))} received, {_amount(row.get('out_kzt'))} sent "
                f"at hop {_number(depth, 0)}. Sample only; not an account balance.")
    else:
        in_value, out_value = row.get("in_deg"), row.get("out_deg")
        if _finite(in_value) and _finite(out_value) and float(in_value) == 0 and float(out_value) == 0:
            body = "Peripheral: 0 incoming and 0 outgoing counterparties observed; no structural role qualifies."
        else:
            body = (f"Peripheral: {incoming} incoming, {outgoing} outgoing counterparties; "
                    "no role meets eligibility and strength >=0.55.")
    if not any(char.isdigit() for char in body):
        body += " Rule threshold=0.55."
    return _bounded(body, warnings)


def add_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["evidence"] = (result.apply(evidence_for, axis=1) if len(result)
                          else pd.Series(index=result.index, dtype=str))
    return result
