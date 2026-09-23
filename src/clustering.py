"""Deterministic Louvain communities and directed cross-community measurements."""
from __future__ import annotations

from collections import defaultdict
import math
from numbers import Integral

import networkx as nx
import pandas as pd


def _int64_gid(value) -> int:
    """Preserve identifiers exactly; a float cannot prove an exact int64 gid."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Every gid must be an integer, never a float or boolean")
    gid = int(value)
    if not -(2**63) <= gid < 2**63:
        raise ValueError("Every gid must fit signed int64")
    return gid


def cluster_graph(graph: nx.DiGraph, nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return gid-keyed cluster features and cluster membership metadata.

    Only the Louvain projection is undirected. Reciprocal amounts are summed
    with ``math.fsum`` after canonical numeric ordering of nodes and edges.
    Missing graph nodes are inserted into the projection as supplied isolates.
    The input graph is never mutated. Cluster ids sort by decreasing seed count,
    decreasing directed internal turnover, then the smallest numeric gid.

    ``internal_out_kzt`` is each node's observed outgoing amount to members of
    its own cluster; summing it reconciles the cluster turnover export.
    """
    if not graph.is_directed():
        raise ValueError("Community analysis requires a directed source graph")
    supplied_gids = [_int64_gid(gid) for gid in nodes["gid"]]
    if len(set(supplied_gids)) != len(supplied_gids):
        raise ValueError("Supplied node gids must be unique")
    known_gids = set(supplied_gids)
    graph_gids = {_int64_gid(gid) for gid in graph.nodes}
    if not graph_gids.issubset(known_gids):
        raise ValueError("The source graph contains gids absent from supplied nodes")

    # Aggregate parallel directed edges as well as reciprocal projection edges
    # in canonical order, so input row permutations cannot alter float sums.
    directed_amounts = defaultdict(list)
    for source, target, data in graph.edges(data=True):
        amount = float(data.get("sum_kzt", 0.0))
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Graph sum_kzt weights must be finite and nonnegative")
        directed_amounts[(_int64_gid(source), _int64_gid(target))].append(amount)
    directed_edges = [
        (source, target, math.fsum(sorted(amounts)))
        for (source, target), amounts in sorted(directed_amounts.items())
    ]
    projection_amounts = defaultdict(list)
    for source, target, amount in directed_edges:
        projection_amounts[(min(source, target), max(source, target))].append(amount)
    ug = nx.Graph()
    ug.add_nodes_from(sorted(known_gids))
    for (source, target), amounts in sorted(projection_amounts.items()):
        ug.add_edge(source, target, sum_kzt=math.fsum(sorted(amounts)))

    # Louvain's modularity denominator is zero for empty/zero-turnover graphs.
    # With no weighted community evidence, retain each supplied node separately.
    if not directed_edges or math.fsum(edge[2] for edge in directed_edges) == 0:
        communities = [{gid} for gid in sorted(known_gids)]
    else:
        communities = list(nx.community.louvain_communities(
            ug, weight="sum_kzt", resolution=1.0, seed=42
        ))
    seed_flags = nodes.get("is_seed", pd.Series(False, index=nodes.index))
    seed_set = set(nodes.loc[seed_flags.fillna(False).astype(bool), "gid"])
    raw_owner = {gid: i for i, community in enumerate(communities) for gid in community}
    internal_amounts = defaultdict(list)
    for source, target, amount in directed_edges:
        if raw_owner[source] == raw_owner[target]:
            internal_amounts[raw_owner[source]].append(amount)
    turnovers = {
        i: math.fsum(internal_amounts[i]) for i in range(len(communities))
    }
    communities.sort(key=lambda members: (
        -len(members & seed_set),
        -turnovers[raw_owner[min(members)]],
        min(members),
    ))
    cluster_of = {gid: i for i, community in enumerate(communities) for gid in community}
    cross_in = dict.fromkeys(known_gids, 0)
    cross_out = dict.fromkeys(known_gids, 0)
    adjacent_clusters = {gid: set() for gid in known_gids}
    internal_out = defaultdict(list)
    for source, target, amount in directed_edges:
        if cluster_of[source] != cluster_of[target]:
            cross_out[source] += 1
            cross_in[target] += 1
            adjacent_clusters[source].add(cluster_of[target])
            adjacent_clusters[target].add(cluster_of[source])
        else:
            internal_out[source].append(amount)

    # Output order follows the supplied node table, independently of the
    # canonical order required inside the randomized clustering algorithm.
    result = pd.DataFrame({"gid": pd.Series(supplied_gids, dtype="int64")})
    result["cluster_id"] = result["gid"].map(cluster_of).astype("int64")
    result["cross_cluster_in_deg"] = result["gid"].map(cross_in).astype("int64")
    result["cross_cluster_out_deg"] = result["gid"].map(cross_out).astype("int64")
    result["cross_cluster_degree"] = result["cross_cluster_in_deg"] + result["cross_cluster_out_deg"]
    result["cross_cluster_count"] = result["gid"].map(
        {gid: len(clusters) for gid, clusters in adjacent_clusters.items()}
    ).astype("int64")
    result["internal_out_kzt"] = result["gid"].map(
        {gid: math.fsum(internal_out[gid]) for gid in known_gids}
    ).astype(float)
    return result, pd.DataFrame({
        "cluster_id": pd.Series(range(len(communities)), dtype="int64"),
        "gids": [sorted(community) for community in communities],
    })
