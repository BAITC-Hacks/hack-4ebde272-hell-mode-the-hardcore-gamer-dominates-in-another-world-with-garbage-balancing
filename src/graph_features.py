"""Deterministic graph and temporal feature construction for the Money Graph."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd

from .schema import validate_inputs
from .temporal import temporal_features


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph retaining all nodes and aggregated edge data."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["gid"].tolist())
    for row in edges.itertuples(index=False):
        amount = float(row.sum_kzt)
        graph.add_edge(
            row.src,
            row.dst,
            sum_kzt=amount,
            n_tx=int(row.n_tx),
            depth=int(row.depth),
            distance=1.0 / math.log1p(amount),
        )
    return graph


def percentile_rank(values: pd.Series) -> pd.Series:
    """Return deterministic average-tie percentile ranks in [0, 1].

    Missing values remain missing. A singleton valid group receives rank 1.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    count = int(numeric.notna().sum())
    if count == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return numeric.rank(method="average", pct=True)


def weighted_pagerank(graph: nx.DiGraph, *, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> dict:
    """Compute weighted PageRank with NumPy-free-of-SciPy graph iteration."""
    nodes = list(graph.nodes)
    count = len(nodes)
    if count == 0:
        return {}
    rank = {node: 1.0 / count for node in nodes}
    out_weight = {
        node: sum(float(data.get("sum_kzt", 1.0)) for _, _, data in graph.out_edges(node, data=True))
        for node in nodes
    }
    for _ in range(max_iter):
        dangling_rank = sum(rank[node] for node in nodes if out_weight[node] == 0.0) / count
        next_rank = {node: (1.0 - alpha) / count + alpha * dangling_rank for node in nodes}
        for source in nodes:
            denominator = out_weight[source]
            if denominator:
                contribution = alpha * rank[source] / denominator
                for _, target, data in graph.out_edges(source, data=True):
                    next_rank[target] += contribution * float(data.get("sum_kzt", 1.0))
        error = sum(abs(next_rank[node] - rank[node]) for node in nodes)
        rank = next_rank
        if error < count * tol:
            return rank
    raise nx.PowerIterationFailedConvergence(max_iter)


def _seed_reach(graph: nx.DiGraph, seeds: list) -> pd.DataFrame:
    """Count seed reachability and shortest seed distance up to four hops."""
    reach_count = {gid: 0 for gid in graph.nodes}
    min_distance = {gid: np.nan for gid in graph.nodes}
    for seed in sorted(seeds, key=str):
        distances = nx.single_source_shortest_path_length(graph, seed, cutoff=4)
        for gid, distance in distances.items():
            reach_count[gid] += 1
            old = min_distance[gid]
            if pd.isna(old) or distance < old:
                min_distance[gid] = distance
    denominator = len(seeds)
    return pd.DataFrame({
        "seed_reach_count": pd.Series(reach_count, dtype="int64"),
        "seed_reach_fraction": pd.Series({gid: count / denominator if denominator else 0.0 for gid, count in reach_count.items()}),
        "min_seed_distance": pd.Series(min_distance, dtype=float),
    })


def _peer_anomaly(frame: pd.DataFrame) -> pd.Series:
    """Combine depth-peer robust deviations into a bounded anomaly score."""
    metrics = {
        "in_deg": frame["in_deg"].astype(float),
        "out_deg": frame["out_deg"].astype(float),
        "log1p_in_kzt": np.log1p(frame["in_kzt"]),
        "log1p_out_kzt": np.log1p(frame["out_kzt"]),
        "in_tx": frame["in_tx"].astype(float),
        "out_tx": frame["out_tx"].astype(float),
    }
    signals = []
    for values in metrics.values():
        signal = pd.Series(0.0, index=frame.index)
        for _, indices in frame.groupby("depth", sort=True).groups.items():
            group = values.loc[indices]
            median = float(group.median())
            mad = float((group - median).abs().median())
            deviation = (group - median).abs()
            if mad > 0:
                robust_z = 0.67448975 * deviation / mad
                signal.loc[indices] = (robust_z / (1.0 + robust_z)).clip(0.0, 1.0)
            else:
                signal.loc[indices] = (deviation > 0).astype(float)
        signals.append(signal)
    return pd.concat(signals, axis=1).mean(axis=1).clip(0.0, 1.0)


def build_features(nodes: pd.DataFrame, edges: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Validate inputs and return one deterministic feature row per node gid."""
    nodes, edges, transactions = validate_inputs(nodes, edges, transactions)
    graph = build_graph(nodes, edges)
    gids = nodes["gid"].tolist()
    features = nodes.set_index("gid")[["depth", "is_seed"]].copy()
    features["is_seed"] = features["is_seed"].astype(bool)
    features["boundary_censored"] = features["depth"].eq(4)

    for name, mapping in (
        ("in_deg", dict(graph.in_degree())),
        ("out_deg", dict(graph.out_degree())),
        ("in_kzt", dict(graph.in_degree(weight="sum_kzt"))),
        ("out_kzt", dict(graph.out_degree(weight="sum_kzt"))),
        ("in_tx", dict(graph.in_degree(weight="n_tx"))),
        ("out_tx", dict(graph.out_degree(weight="n_tx"))),
    ):
        features[name] = pd.Series(mapping).reindex(gids).fillna(0).to_numpy()
    features["in_deg"] = features["in_deg"].astype("int64")
    features["out_deg"] = features["out_deg"].astype("int64")
    features["in_tx"] = features["in_tx"].astype("int64")
    features["out_tx"] = features["out_tx"].astype("int64")
    features["total_kzt"] = features["in_kzt"] + features["out_kzt"]
    features["total_tx"] = features["in_tx"] + features["out_tx"]

    pagerank = weighted_pagerank(graph)
    features["pagerank"] = pd.Series(pagerank).reindex(gids).fillna(0.0)
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight="distance")
    features["betweenness"] = pd.Series(betweenness).reindex(gids).fillna(0.0)

    non_seed = ~features["is_seed"]
    complete_observation = non_seed & ~features["boundary_censored"]
    total_degree = features["in_deg"] + features["out_deg"]
    features["pass_through"] = np.nan
    valid_pass = complete_observation & features["in_kzt"].gt(0)
    features.loc[valid_pass, "pass_through"] = features.loc[valid_pass, "out_kzt"] / features.loc[valid_pass, "in_kzt"]
    features["fanin_share"] = np.nan
    features["fanout_share"] = np.nan
    valid_share = complete_observation & total_degree.gt(0)
    features.loc[valid_share, "fanin_share"] = features.loc[valid_share, "in_deg"] / total_degree[valid_share]
    features.loc[valid_share, "fanout_share"] = features.loc[valid_share, "out_deg"] / total_degree[valid_share]

    seeds = nodes.loc[nodes["is_seed"].astype(bool), "gid"].tolist()
    features = features.join(_seed_reach(graph, seeds).reindex(gids))

    # Stable SCC identifiers are assigned by the smallest string representation
    # of a component member, independent of NetworkX traversal order.
    components = sorted(nx.strongly_connected_components(graph), key=lambda comp: min(map(str, comp)))
    scc_id, scc_size = {}, {}
    for component_id, component in enumerate(components):
        for gid in component:
            scc_id[gid] = component_id
            scc_size[gid] = len(component)
    features["scc_id"] = pd.Series(scc_id).reindex(gids).astype("int64")
    features["scc_size"] = pd.Series(scc_size).reindex(gids).astype("int64")
    features["in_cycle"] = features["scc_size"].gt(1)
    reciprocal = {gid: 0 for gid in gids}
    for src, dst in graph.edges:
        if src != dst and graph.has_edge(dst, src):
            reciprocal[src] += 1
    features["reciprocal_relationship_count"] = pd.Series(reciprocal).reindex(gids).astype("int64")

    features = features.join(temporal_features(nodes, transactions).reindex(gids))
    # The seed observation window has incomplete inbound transactions, so
    # date-overlap and activity-share ratios for seeds are not comparable.
    features.loc[features["is_seed"], ["same_day_flow_ratio", "relay_2d_ratio", "peak_day_share"]] = np.nan
    features["peer_anomaly_score"] = _peer_anomaly(features)

    percentile_sources = [
        "in_deg", "out_deg", "in_kzt", "out_kzt", "in_tx", "out_tx", "total_kzt", "total_tx",
        "pagerank", "betweenness", "seed_reach_count", "seed_reach_fraction", "active_days",
        "same_day_flow_ratio", "relay_2d_ratio", "max_in_senders_day", "peak_day_share", "peer_anomaly_score",
    ]
    for column in percentile_sources:
        features[f"{column}_pct"] = percentile_rank(features[column])

    # Keep exact, stable node ordering from the validated input and index by gid.
    features.index.name = "gid"
    return features.reindex(gids)
