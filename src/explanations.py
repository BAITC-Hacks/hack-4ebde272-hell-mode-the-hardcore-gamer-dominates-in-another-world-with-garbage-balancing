"""Deterministic, numeric evidence text for analyst review."""
from __future__ import annotations

import math
import pandas as pd


def _num(value, digits=2):
    try:
        x = float(value)
        return "n/a" if not math.isfinite(x) else f"{x:,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def evidence_for(row: pd.Series) -> str:
    role = str(row.get("role", "peripheral"))
    depth = float(row.get("depth", 0) or 0)
    out_degree = float(row.get("out_deg", 0) or 0)
    vals = {
        "in_deg": _num(row.get("in_deg", 0), 0), "out_deg": _num(row.get("out_deg", 0), 0),
        "in_kzt": _num(row.get("in_kzt", 0), 0), "out_kzt": _num(row.get("out_kzt", 0), 0),
        "seed_reach_count": _num(row.get("seed_reach_count", 0), 0),
        "retention": _num(row.get("retention", float("nan"))),
    }
    if depth >= 4 and out_degree == 0:
        text = f"depth={_num(depth,0)}, in_deg={vals['in_deg']}, out_deg={vals['out_deg']}; boundary-censored beyond hop 4"
    elif role == "consolidator":
        text = f"in_deg={vals['in_deg']}, in={vals['in_kzt']} KZT, seed_reach={vals['seed_reach_count']}, retention={vals['retention']}"
    elif role == "coordinator":
        bridge = _num(row.get("cross_cluster_degree", 0), 0)
        text = f"seed_reach={vals['seed_reach_count']}, betweenness={_num(row.get('betweenness_percentile', 0)*100,0)}p, PageRank={_num(row.get('pagerank_percentile',0)*100,0)}p, bridges={bridge}"
    elif role == "distributor":
        text = f"out_deg={vals['out_deg']}, out={vals['out_kzt']} KZT, out_tx={_num(row.get('out_tx',0),0)}, cross_out={_num(row.get('cross_cluster_out_deg',0),0)}"
    elif role == "transit":
        text = f"in_deg={vals['in_deg']}, out_deg={vals['out_deg']}, pass={_num(row.get('pass_through'))}, relay_2d={_num(row.get('relay_2d_ratio'))}"
    elif role == "terminal":
        text = f"depth={_num(row.get('depth'),0)}, in_deg={vals['in_deg']}, out_deg={vals['out_deg']}, retention={vals['retention']}"
    else:
        text = f"in_deg={vals['in_deg']}, out_deg={vals['out_deg']}, seed_reach={vals['seed_reach_count']}, depth={_num(row.get('depth',0),0)}"
    # Defensive guarantee: evidence is compact, numeric, and never an accusation.
    if not any(c.isdigit() for c in text):
        text = f"role_score={_num(row.get('role_score', 0))}; {text}"
    return text[:200]


def add_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["evidence"] = result.apply(evidence_for, axis=1)
    return result
