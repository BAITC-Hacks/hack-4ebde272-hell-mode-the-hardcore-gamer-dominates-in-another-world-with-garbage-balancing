"""Deterministic community assignment for the AML investigation graph."""
from __future__ import annotations

import networkx as nx
import pandas as pd


def cluster_graph(graph: nx.DiGraph, nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign stable Louvain clusters and return cross-cluster directed degrees.

    Only this projection is undirected. The input graph remains directed and is
    used for all subsequent measurements.
    """
    ug = nx.Graph()
    ug.add_nodes_from(graph.nodes)
    for src, dst, data in graph.edges(data=True):
        w = float(data.get("sum_kzt", 0.0))
        if ug.has_edge(src, dst):
            ug[src][dst]["sum_kzt"] += w
        else:
            ug.add_edge(src, dst, sum_kzt=w)

    communities = list(nx.community.louvain_communities(
        ug, weight="sum_kzt", resolution=1.0, seed=42
    ))
    seed_set = set(nodes.loc[nodes["is_seed"].fillna(False).astype(bool), "gid"])

    raw_owner = {gid: i for i, community in enumerate(communities) for gid in community}
    turnovers = [0.0] * len(communities)
    for src, dst, data in graph.edges(data=True):
        owner = raw_owner.get(src)
        if owner is not None and owner == raw_owner.get(dst):
            turnovers[owner] += float(data.get("sum_kzt", 0.0))
    communities = sorted(communities, key=lambda c: (-len(c & seed_set),
                            -turnovers[raw_owner[next(iter(c))]],
                            min((str(g) for g in c), default="")))
    cluster_of = {gid: i for i, community in enumerate(communities) for gid in community}
    # Include any supplied node absent from the graph, preserving the one-row-per-node contract.
    for gid in nodes["gid"]:
        if gid not in cluster_of:
            cluster_of[gid] = len(communities)
            communities.append({gid})

    result = nodes[["gid"]].copy()
    result["cluster_id"] = result["gid"].map(cluster_of).astype(int)
    cross_in = {gid: 0 for gid in cluster_of}
    cross_out = {gid: 0 for gid in cluster_of}
    for src, dst in graph.edges:
        if cluster_of.get(src) != cluster_of.get(dst):
            cross_out[src] = cross_out.get(src, 0) + 1
            cross_in[dst] = cross_in.get(dst, 0) + 1
    result["cross_cluster_in_deg"] = result.gid.map(cross_in).fillna(0).astype(int)
    result["cross_cluster_out_deg"] = result.gid.map(cross_out).fillna(0).astype(int)
    result["cross_cluster_degree"] = result["cross_cluster_in_deg"] + result["cross_cluster_out_deg"]
    return result, pd.DataFrame({"cluster_id": range(len(communities)),
                                 "gids": [sorted(c, key=str) for c in communities]})
