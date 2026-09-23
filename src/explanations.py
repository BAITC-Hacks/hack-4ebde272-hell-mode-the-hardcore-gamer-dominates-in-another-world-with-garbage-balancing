"""Readable role hypotheses with numeric evidence and observation limits."""
from __future__ import annotations

import math
from numbers import Integral

import pandas as pd

from .roles import ratio_validity_flags


def _number(value, digits=2) -> str:
    if digits == 0 and isinstance(value, Integral) and not isinstance(value, bool):
        return f"{int(value):,}"
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
    if abs(value) >= 1e15:
        return f"about {value:.3g} KZT"
    return f"{value:,.2f}".rstrip("0").rstrip(".") + " KZT"


def _counted(value, noun: str) -> str:
    """Format a measured count with a grammatical singular/plural label."""
    singular = _finite(value) and float(value) == 1
    return f"{_number(value, 0)} {noun if singular else noun + 's'}"


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
    for flag in (f"decision_{name}_valid", *ratio_validity_flags(name)):
        if flag in row and not _flag(row, flag):
            return float("nan")
    value = row.get(f"decision_{name}", row.get(name))
    return float(value) if _finite(value) and float(value) >= 0 else float("nan")


def _bounded(alternatives: list[str], warning: str) -> str:
    """Choose a complete explanation; never slice a sentence or numeric value."""
    fallback = "Observed metrics are insufficient to explain the role; the support cutoff is 0.55."
    for body in (*alternatives, fallback):
        result = body + (" " + warning if warning else "")
        if len(result) <= 200:
            return result
    raise ValueError("Observation warning exceeds the evidence character budget")


def _top_rank(value: float) -> str:
    """Translate a percentile to a short relative-rank phrase without jargon."""
    if value >= 1:
        return "rank 1"
    # Keep a nonzero top-share label even when a value is very close to 1.
    share = max(0.1, (1.0 - value) * 100)
    return f"top {share:.1f}".rstrip("0").rstrip(".") + "%"


def evidence_for(row: pd.Series) -> str:
    """Explain a role in everyday language, with complete text <=200 characters.

    Counts/amounts describe this sample. Initial case accounts are the supplied
    seeds. Shorter complete alternatives reserve room for mandatory observation
    warnings; detailed gates and components remain in the auxiliary artifact.
    """
    role = str(row.get("role", "peripheral"))
    depth = row.get("depth")
    boundary = (_finite(depth) and float(depth) >= 4) or _flag(row, "boundary_censored") or _flag(row, "truncated_by_depth")
    seed = _flag(row, "is_seed")
    senders = _counted(row.get("in_deg"), "sender")
    recipients = _counted(row.get("out_deg"), "recipient")
    seed_sources = _counted(row.get("seed_reach_count"), "initial case account")
    if seed and _finite(row.get("seed_reach_count")) and float(row["seed_reach_count"]) >= 1:
        seed_sources += " (including itself)"
    if seed and boundary:
        warning = "Initial case account: inflows are incomplete; outgoing transfers beyond hop 4 are unobserved."
    elif seed:
        warning = "Initial case account; incoming transfers are incomplete."
    elif boundary:
        warning = "Outgoing transfers beyond hop 4 are unobserved."
    else:
        warning = ""

    if not any(_finite(row.get(name)) for name in ("in_deg", "out_deg", "in_kzt", "out_kzt")):
        return _bounded(["Activity counts are unavailable; no role can be explained against the 0.55 support cutoff."], warning)

    if role == "consolidator":
        collection = f"Received {_amount(row.get('in_kzt'))} from {senders}; a possible collection point."
        alternatives = []
        if _finite(row.get("in_kzt")):
            if _finite(row.get("seed_reach_count")):
                alternatives.append(collection + f" Reachable from {seed_sources}.")
            alternatives.append(collection)
        alternatives.append(f"Receives from {senders}; a possible collection point.")
    elif role == "distributor":
        transfers = _counted(row.get("out_tx"), "transfer")
        alternatives = [
            f"Sent {_amount(row.get('out_kzt'))} to {recipients} in {transfers}, suggesting a distribution role.",
            f"Sent funds to {recipients} in {transfers}; a possible distributor.",
            f"Sent funds to {recipients}; a possible distributor.",
        ]
    elif role == "coordinator":
        phrases = {
            "seed_reach_count": f"reachable from {seed_sources}",
            "betweenness": f"{_top_rank(_percentile(row, 'betweenness'))} for linking payment paths",
            "pagerank": f"{_top_rank(_percentile(row, 'pagerank'))} for network importance",
            "cross_cluster_degree": f"{_number(row.get('cross_cluster_degree'), 0)} payment links to other groups",
        }
        ranked = sorted(enumerate(phrases), key=lambda item: (
            -(_percentile(row, item[1]) if _finite(_percentile(row, item[1])) else -1), item[0]))
        strongest = [phrases[name] for _, name in ranked if _finite(_percentile(row, name))][:2]
        alternatives = []
        if strongest:
            alternatives.append("Possible coordinating account: " + "; ".join(strongest) + ".")
        alternatives.extend([
            f"Reachable from {seed_sources}, with {_number(row.get('cross_cluster_degree'), 0)} links to other groups; review for a coordinating role.",
            f"Has {senders} and {recipients}; review for a possible coordinating role.",
        ])
    elif role == "transit":
        ratio = _ratio(row, "pass_through") if not (seed or boundary) else float("nan")
        relay = _ratio(row, "relay_2d_ratio") if not (seed or boundary) else float("nan")
        alternatives = []
        if _finite(relay):
            timing = f"outgoing activity occurs on or up to 2 days after {_number(relay * 100, 0)}% of eligible incoming dates"
            if _finite(ratio):
                alternatives.append(f"Outgoing amount is {_number(ratio * 100, 0)}% of incoming amount; {timing}. Dates cannot prove order or trace funds.")
            alternatives.append(timing.capitalize() + ", suggesting a relay role. Dates cannot prove order or trace funds.")
        elif _finite(ratio):
            alternatives.append(f"Outgoing amount is {_number(ratio * 100, 0)}% of incoming amount, consistent with a relay role. Two-day timing is unavailable.")
        alternatives.append(f"Observed {senders} and {recipients}; two-day timing is unavailable.")
    elif role == "terminal":
        alternatives = [
            f"Received {_amount(row.get('in_kzt'))} and sent {_amount(row.get('out_kzt'))} in this sample, suggesting an endpoint. This is not a full account balance.",
            f"Receives from {senders} with little observed onward flow. Full account balance is unknown.",
        ]
    else:
        in_value, out_value = row.get("in_deg"), row.get("out_deg")
        if _finite(in_value) and _finite(out_value) and float(in_value) == 0 and float(out_value) == 0:
            alternatives = ["No transfers observed (0 incoming, 0 outgoing); there is too little evidence for a specific role."]
        else:
            alternatives = [f"Observed {senders} and {recipients}; no specific role has enough support."]
    return _bounded(alternatives, warning)


def add_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["evidence"] = (result.apply(evidence_for, axis=1) if len(result)
                          else pd.Series(index=result.index, dtype=str))
    return result
