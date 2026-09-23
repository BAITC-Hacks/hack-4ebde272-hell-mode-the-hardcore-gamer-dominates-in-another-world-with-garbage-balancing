"""Weak-connectivity resilience scenarios for priority-node removals."""
from __future__ import annotations

import networkx as nx
import pandas as pd


def resilience_analysis(graph: nx.DiGraph, features: pd.DataFrame,
                        removal_counts=(1, 5, 10, 20)) -> pd.DataFrame:
    base_nodes = set(graph.nodes)
    base_largest = max((len(c) for c in nx.weakly_connected_components(graph)), default=0)
    seed_set = set(features.loc[features.get("is_seed", pd.Series(False, index=features.index)).fillna(False).astype(bool), "gid"])
    ordered = features.sort_values(["priority_score", "gid"], ascending=[False, True], kind="mergesort")["gid"].tolist()
    rows = []
    for count in dict.fromkeys((0, *removal_counts)):
        removed = set(ordered[:count])
        remaining = base_nodes - removed
        sub = graph.subgraph(remaining)
        components = list(nx.weakly_connected_components(sub))
        largest = max(components, key=len) if components else set()
        rows.append({"scenario": "baseline" if count == 0 else f"remove_top_{count}", "removed_top_n": len(removed), "n_weak_components": len(components),
                     "largest_weak_component_size": len(largest),
                     "fraction_baseline_largest": len(largest) / base_largest if base_largest else 0.0,
                     "seeds_in_largest_component": len(largest & seed_set)})
    return pd.DataFrame(rows)
