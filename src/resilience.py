"""Structural concentration under priority-node removals.

These hypothetical graph cuts measure observed connectivity and concentration.
They are not evidence of criminal-network disruption or real-world causality.
Weak components ignore edge direction for connectivity; they are not Louvain
communities. The original directed graph and transaction observations are kept.
"""
from __future__ import annotations

from numbers import Integral

import networkx as nx
import pandas as pd

from .clustering import _int64_gid


def resilience_analysis(graph: nx.DiGraph, features: pd.DataFrame,
                        removal_counts=(1, 5, 10, 20)) -> pd.DataFrame:
    """Return baseline and top-N removal scenarios with deterministic ties.

    Nodes rank by decreasing review priority, then increasing exact numeric gid.
    Equally large weak components are resolved by their minimum gid, ensuring
    reproducible seed counts. Supplied isolated feature rows remain in baseline.
    """
    if not graph.is_directed():
        raise ValueError("Resilience analysis requires a directed source graph")
    gids = [_int64_gid(gid) for gid in features["gid"]]
    if len(set(gids)) != len(gids):
        raise ValueError("Resilience features must contain unique gids")
    for gid in graph.nodes:
        _int64_gid(gid)
    counts = tuple(removal_counts)
    if any(isinstance(count, bool) or not isinstance(count, Integral) or count < 0 for count in counts):
        raise ValueError("Removal counts must be nonnegative integers")
    analysis_graph = graph.copy()
    analysis_graph.add_nodes_from(gids)
    base_nodes = set(analysis_graph.nodes)
    base_largest = max((len(c) for c in nx.weakly_connected_components(analysis_graph)), default=0)
    seed_flags = features.get("is_seed", pd.Series(False, index=features.index))
    seed_set = set(features.loc[seed_flags.fillna(False).astype(bool), "gid"])
    ordered = features.sort_values(
        ["priority_score", "gid"], ascending=[False, True], kind="mergesort"
    )["gid"].tolist()
    rows = []
    for count in dict.fromkeys((0, *counts)):
        removed = set(ordered[:count])
        sub = analysis_graph.subgraph(base_nodes - removed)
        components = list(nx.weakly_connected_components(sub))
        largest = min(components, key=lambda members: (-len(members), min(members))) if components else set()
        rows.append({
            "scenario": "baseline" if count == 0 else f"remove_top_{count}",
            "removed_top_n": len(removed),
            "n_weak_components": len(components),
            "largest_weak_component_size": len(largest),
            "fraction_baseline_largest": len(largest) / base_largest if base_largest else 0.0,
            "seeds_in_largest_component": len(largest & seed_set),
        })
    return pd.DataFrame(rows)
