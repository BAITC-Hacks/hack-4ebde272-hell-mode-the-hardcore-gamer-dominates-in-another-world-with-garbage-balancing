"""Deterministic graph and temporal feature construction for the Money Graph."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd

from .schema import validate_inputs
from .temporal import temporal_features


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Build a canonical directed graph from validated int64 input tables.

    Numeric node order and (src, dst) edge order are independent of input row
    order. Isolates are retained; consumers must not rebuild an edge-only graph.
    Call ``validate_inputs`` first when accepting untrusted tables.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes["gid"].tolist()))
    for row in edges.sort_values(["src", "dst"], kind="stable").itertuples(index=False):
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

    Non-numeric/non-finite values remain missing. A singleton receives rank 1.
    For n equal valid values, every rank is (n + 1) / (2 * n), not zero.
    """
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    count = int(numeric.notna().sum())
    if count == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return numeric.rank(method="average", pct=True)


def weighted_pagerank(graph: nx.DiGraph, *, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> dict:
    """Compute amount-weighted PageRank without an implicit SciPy dependency."""
    nodes = sorted(graph.nodes)
    count = len(nodes)
    if count == 0:
        return {}
    rank = {node: 1.0 / count for node in nodes}
    outgoing = {node: [(target, float(graph[node][target].get("sum_kzt", 1.0)))
                       for target in sorted(graph.successors(node))] for node in nodes}
    out_weight = {node: math.fsum(weight for _, weight in outgoing[node]) for node in nodes}
    for _ in range(max_iter):
        dangling_rank = math.fsum(rank[node] for node in nodes if out_weight[node] == 0.0) / count
        next_rank = {node: (1.0 - alpha) / count + alpha * dangling_rank for node in nodes}
        for source in nodes:
            denominator = out_weight[source]
            if denominator:
                contribution = alpha * rank[source] / denominator
                for target, weight in outgoing[source]:
                    next_rank[target] += contribution * weight
        error = math.fsum(abs(next_rank[node] - rank[node]) for node in nodes)
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
        "seed_reach_fraction": pd.Series({gid: count / denominator if denominator else 0.0 for gid, count in reach_count.items()}, dtype="float64"),
        "min_seed_distance": pd.Series(min_distance, dtype=float),
    })


def _peer_anomaly_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose six equally weighted, normalized depth-peer deviations.

    With zero MAD the component is 0 at the median and 1 elsewhere. A singleton
    therefore has no within-group anomaly. Scores are deviations, not risk or
    probabilities, and include unusually low as well as high activity.
    """
    metrics = {
        "in_deg": frame["in_deg"].astype(float),
        "out_deg": frame["out_deg"].astype(float),
        "log1p_in_kzt": np.log1p(frame["in_kzt"]),
        "log1p_out_kzt": np.log1p(frame["out_kzt"]),
        "in_tx": frame["in_tx"].astype(float),
        "out_tx": frame["out_tx"].astype(float),
    }
    signals = {}
    for name, values in metrics.items():
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
        signals[f"peer_anomaly_{name}"] = signal
    return pd.DataFrame(signals, index=frame.index)


def _peer_anomaly(frame: pd.DataFrame) -> pd.Series:
    """Return the mean of the six documented depth-peer anomaly components."""
    return _peer_anomaly_components(frame).mean(axis=1).clip(0.0, 1.0)


def build_features(nodes: pd.DataFrame, edges: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Return one validated, deterministic row per gid in input node order.

    Observation flags distinguish missing/censored metrics from observed zeros.
    See documentation/feature-contract.md for formulas and denominator rules.
    """
    nodes, edges, transactions = validate_inputs(nodes, edges, transactions)
    graph = build_graph(nodes, edges)
    gids = nodes["gid"].tolist()
    features = nodes.set_index("gid")[["depth", "is_seed"]].copy()
    features["is_seed"] = features["is_seed"].astype(bool)
    features["boundary_censored"] = features["depth"].eq(4)

    try:
        in_amounts = {gid: math.fsum(edge["sum_kzt"] for _, _, edge in graph.in_edges(gid, data=True))
                      for gid in graph}
        out_amounts = {gid: math.fsum(edge["sum_kzt"] for _, _, edge in graph.out_edges(gid, data=True))
                       for gid in graph}
        total_amounts = {gid: math.fsum((in_amounts[gid], out_amounts[gid])) for gid in graph}
    except OverflowError as exc:
        raise ValueError("node amount totals exceed the finite float64 range") from exc

    for name, mapping in (
        ("in_deg", dict(graph.in_degree())),
        ("out_deg", dict(graph.out_degree())),
        ("in_kzt", in_amounts),
        ("out_kzt", out_amounts),
        ("in_tx", dict(graph.in_degree(weight="n_tx"))),
        ("out_tx", dict(graph.out_degree(weight="n_tx"))),
    ):
        features[name] = pd.Series(mapping).reindex(gids).fillna(0).to_numpy()
    features["in_deg"] = features["in_deg"].astype("int64")
    features["out_deg"] = features["out_deg"].astype("int64")
    features["in_tx"] = features["in_tx"].astype("int64")
    features["out_tx"] = features["out_tx"].astype("int64")
    features[["in_kzt", "out_kzt"]] = features[["in_kzt", "out_kzt"]].astype("float64")
    features["total_kzt"] = pd.Series(total_amounts, dtype="float64").reindex(gids)
    features["total_tx"] = features["in_tx"] + features["out_tx"]

    pagerank = weighted_pagerank(graph)
    features["pagerank"] = pd.Series(pagerank, dtype="float64").reindex(gids).fillna(0.0)
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight="distance")
    features["betweenness"] = pd.Series(betweenness, dtype="float64").reindex(gids).fillna(0.0)

    non_seed = ~features["is_seed"]
    ratio_eligible = non_seed & ~features["boundary_censored"]
    total_degree = features["in_deg"] + features["out_deg"]
    features["pass_through"] = np.nan
    valid_pass = ratio_eligible & features["in_kzt"].gt(0)
    features["pass_through_valid"] = valid_pass
    features["pass_through_invalid_reason"] = pd.array(np.select(
        [features["is_seed"], features["boundary_censored"], features["in_kzt"].le(0)],
        ["seed_inbound_incomplete", "boundary_outbound_incomplete", "no_inbound_activity"],
        default="",
    ), dtype="string")
    features.loc[valid_pass, "pass_through"] = features.loc[valid_pass, "out_kzt"] / features.loc[valid_pass, "in_kzt"]
    features["fanin_share"] = np.nan
    features["fanout_share"] = np.nan
    valid_share = ratio_eligible & total_degree.gt(0)
    features["flow_share_valid"] = valid_share
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
    # temporal_features owns its validity masks so direct callers receive the
    # same censored-window behavior as the full feature layer.
    components = _peer_anomaly_components(features)
    features = features.join(components)
    features["peer_anomaly_score"] = components.mean(axis=1).clip(0.0, 1.0)
    features["peer_group_size"] = features.groupby("depth")["depth"].transform("size").astype("int64")

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
