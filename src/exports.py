"""Stable CSV output schemas for the HackAlem submission."""
from __future__ import annotations

from pathlib import Path
from math import fsum
import re
import numpy as np
import pandas as pd

from .explanations import _amount

NODE_COLUMNS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
CLUSTER_COLUMNS = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
TOP_COLUMNS = ["rank", "gid", "role", "priority_score", "why"]


def _cluster_hypothesis(group: pd.DataFrame, graph, seeds: int, turnover: float) -> str:
    """Describe observed group structure without treating membership as control.

    Existing degree/role thresholds select a narrative, not an inferred purpose.
    In particular, multiple initial case accounts in the same undirected Louvain
    group do not establish common directed destinations or coordinated intent.
    """
    count = len(group)
    accounts = f"{count} {'account' if count == 1 else 'accounts'}"
    have = "has" if count == 1 else "have"
    members = group.gid.tolist()
    internal_amount = _amount(turnover)
    degrees = [graph.degree(gid) if gid in graph else 0 for gid in members]
    if not any(degrees):
        if count == 1:
            account = "initial case account" if seeds else "account"
            return (f"This group contains 1 {account} with no observed transfers. "
                    "The sample cannot establish whether it was active outside the observed period or bank.")
        return (f"All {count} accounts have no observed transfers. "
                "No shared financial purpose can be inferred from this sample.")
    if seeds >= 2:
        return (f"This group contains {count} accounts, including {seeds} initial case accounts, "
                f"with {internal_amount} transferred within the group. "
                "Review their shared links; group membership alone does not prove that funds converge or that one person controls the accounts.")

    incoming = int(sum(graph.in_degree(gid) if gid in graph else 0 for gid in members))
    outgoing = int(sum(graph.out_degree(gid) if gid in graph else 0 for gid in members))
    in_mean = group.get("in_deg", pd.Series(0, index=group.index)).mean()
    out_mean = group.get("out_deg", pd.Series(0, index=group.index)).mean()
    if in_mean > out_mean * 1.25:
        return (f"Possible collection group: its {accounts} {have} {incoming} incoming and {outgoing} outgoing payment links. "
                f"Internal transfers total {internal_amount}; full account balances remain unknown.")
    if out_mean > in_mean * 1.25:
        return (f"Possible distribution group: its {accounts} {have} {outgoing} outgoing and {incoming} incoming payment links. "
                f"Internal transfers total {internal_amount}; this describes the sample, not a proven purpose.")
    roles = group.get("role", pd.Series("unassigned", index=group.index))
    transit_count = int(roles.eq("transit").sum())
    peripheral_count = int(roles.eq("peripheral").sum())
    if transit_count / count >= 0.35:
        return (f"Possible relay group: {transit_count} of {count} accounts have a relay role, "
                f"and {internal_amount} moved within the group. "
                "Date-only observations cannot establish that the same funds were passed onward.")
    if count <= 3:
        noun = "account" if count == 1 else "accounts"
        return (f"Small group of {count} {noun} with {internal_amount} transferred internally. "
                "Review the observed links; group size alone does not establish its purpose.")
    if peripheral_count / count >= 0.70:
        return (f"Most accounts ({peripheral_count} of {count}) lack enough evidence for a specific role. "
                f"Internal transfers total {internal_amount}; more data is needed to explain the group's purpose.")
    labels = {
        "consolidator": ("possible collector", "possible collectors"),
        "distributor": ("possible distributor", "possible distributors"),
        "transit": ("possible relay", "possible relays"),
        "coordinator": ("possible coordinating account", "possible coordinating accounts"),
        "terminal": ("account with little observed outflow", "accounts with little observed outflow"),
        "peripheral": ("account without a clear role", "accounts without a clear role"),
    }
    composition = sorted(((int(roles.eq(role).sum()), role) for role in labels),
                         key=lambda item: (-item[0], item[1]))
    details = [f"{number} {labels[role][0 if number == 1 else 1]}" for number, role in composition[:2] if number]
    interpretation = ("Observed roles include " + " and ".join(details) + "; no single financial purpose is established."
                      if details else "No single financial purpose is established.")
    return (f"Mixed payment group: {count} accounts transferred {internal_amount} internally. "
            + interpretation)


def exact_int64(values: pd.Series, label: str) -> pd.Series:
    """Parse integral identifiers without ever routing them through a float."""
    parsed = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} must contain exact int64 integers, not booleans")
        if isinstance(value, (int, np.integer)):
            number = int(value)
        elif isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value):
            number = int(value)
        else:
            raise ValueError(f"{label} must contain exact int64 integers")
        if not -(2**63) <= number < 2**63:
            raise ValueError(f"{label} contains an integer outside int64 range")
        parsed.append(number)
    return pd.Series(parsed, index=values.index, name=values.name, dtype="int64")


def cluster_summaries(features: pd.DataFrame, graph, nodes: pd.DataFrame) -> pd.DataFrame:
    seed = nodes.set_index("gid")["is_seed"].fillna(False).astype(bool).to_dict()
    cluster_of = features.set_index("gid")["cluster_id"].to_dict()
    amounts_by_cluster = {int(cid): [] for cid in features.cluster_id.unique()}
    for src, dst, data in graph.edges(data=True):
        cid = cluster_of.get(src)
        if cid is not None and cid == cluster_of.get(dst):
            amounts_by_cluster[int(cid)].append(float(data.get("sum_kzt", 0.0)))
    rows = []
    for cid, group in features.groupby("cluster_id", sort=True):
        members = set(group.gid)
        turnover = fsum(sorted(amounts_by_cluster[int(cid)]))
        ranked = group.sort_values(["priority_score", "gid"], ascending=[False, True], kind="mergesort")
        gids = ranked.gid.head(5).tolist()
        seeds = sum(bool(seed.get(g, False)) for g in members)
        hypothesis = _cluster_hypothesis(group, graph, seeds, turnover)
        rows.append({"cluster_id": int(cid), "n_nodes": len(group), "n_seed": seeds,
                     "sum_kzt_internal": turnover, "top_gids": ",".join(map(str, gids)),
                     "hypothesis": hypothesis})
    return pd.DataFrame(rows, columns=CLUSTER_COLUMNS)


def write_exports(features: pd.DataFrame, clusters: pd.DataFrame, out_dir: str | Path) -> dict[str, int]:
    """Write strict submission CSVs and the lossless, gid-keyed UI artifact.

    The node CSV is authoritative for its six columns. ``node_features.parquet``
    retains those identical fields and every analytical column supplied by the
    caller, including rule diagnostics and priority contributions. DataFrame
    attrs are not an analytical schema and are deliberately not serialized.
    """
    features = features.copy()
    if "gid" not in features.columns and features.index.name == "gid":
        features = features.reset_index()
    features["gid"] = exact_int64(features["gid"], "features.gid")
    if not features.gid.is_unique:
        raise ValueError("features.gid must be unique")
    features["cluster_id"] = exact_int64(features["cluster_id"], "features.cluster_id")
    if "priority_explanation" not in features:
        raise ValueError("features must include priority_explanation from score_priority")
    reasons = features["priority_explanation"]
    if reasons.isna().any() or not reasons.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("priority_explanation must contain nonempty readable text")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes = features[NODE_COLUMNS].copy()
    nodes.to_csv(out / "nodes_roles.csv", index=False)
    clusters[CLUSTER_COLUMNS].to_csv(out / "clusters.csv", index=False)
    top = features.sort_values(["priority_score", "gid"], ascending=[False, True], kind="mergesort").head(min(50, len(features))).copy()
    tops = pd.DataFrame({"rank": range(1, len(top) + 1), "gid": top.gid.values,
                         "role": top.role.values, "priority_score": top.priority_score.values,
                         "why": top.priority_explanation.values}, columns=TOP_COLUMNS)
    tops.to_csv(out / "top_nodes.csv", index=False)
    auxiliary = features.copy()
    auxiliary.attrs = {}
    auxiliary.to_parquet(out / "node_features.parquet", index=False)
    if "resilience_frame" in features.attrs:
        features.attrs["resilience_frame"].to_csv(out / "resilience.csv", index=False)
    return {"nodes_roles": len(nodes), "clusters": len(clusters), "top_nodes": len(tops),
            "node_features": len(auxiliary)}
