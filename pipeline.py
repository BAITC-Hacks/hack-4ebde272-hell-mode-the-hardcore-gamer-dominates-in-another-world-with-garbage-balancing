"""Local, deterministic AML decision and export pipeline."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.clustering import cluster_graph
from src.roles import assign_roles
from src.priority import score_priority
from src.explanations import add_evidence
from src.exports import cluster_summaries, write_exports
from src.resilience import resilience_analysis
from validate_submission import validate_submission


def load_inputs(data_dir: Path):
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    required = {"src", "dst", "sum_kzt", "n_tx"}
    if not required.issubset(edges.columns):
        raise ValueError(f"edges.parquet missing columns: {sorted(required - set(edges.columns))}")
    if not {"gid", "depth", "is_seed"}.issubset(nodes.columns):
        raise ValueError("nodes.parquet must contain gid, depth, is_seed")
    if not {"src", "dst", "date", "sum_kzt"}.issubset(tx.columns):
        raise ValueError("transactions.parquet must contain src, dst, date, sum_kzt")
    if nodes.gid.duplicated().any():
        raise ValueError("nodes.gid must be unique")
    known = set(nodes.gid)
    unknown_endpoints = (set(edges.src) | set(edges.dst)) - known
    if unknown_endpoints:
        raise ValueError(f"edges contain endpoints absent from nodes: {len(unknown_endpoints)}")
    edge_totals = edges.groupby(["src", "dst"], as_index=False).agg(edge_kzt=("sum_kzt", "sum"), edge_tx=("n_tx", "sum"))
    tx_totals = tx.groupby(["src", "dst"], as_index=False).agg(tx_kzt=("sum_kzt", "sum"), tx_count=("sum_kzt", "size"))
    consistency = edge_totals.merge(tx_totals, on=["src", "dst"], how="outer", indicator=True)
    if not consistency._merge.eq("both").all():
        raise ValueError("edges.parquet and transactions.parquet contain different directed pairs")
    if not np.allclose(consistency.edge_kzt, consistency.tx_kzt, rtol=1e-8, atol=0.01):
        raise ValueError("edge amounts do not reconcile with transactions")
    if not consistency.edge_tx.eq(consistency.tx_count).all():
        raise ValueError("edge transaction counts do not reconcile with transactions")
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce").dt.normalize()
    return edges, nodes, tx


def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.gid)
    for row in edges.itertuples(index=False):
        graph.add_edge(row.src, row.dst, sum_kzt=float(row.sum_kzt), n_tx=int(row.n_tx))
    return graph


def _temporal_metrics(tx: pd.DataFrame, gids, is_seed) -> pd.DataFrame:
    valid = tx.dropna(subset=["date"]).copy()
    day_in = valid.groupby(["dst", "date"], sort=False).sum_kzt.sum().to_dict()
    day_out = valid.groupby(["src", "date"], sort=False).sum_kzt.sum().to_dict()
    rows = []
    for gid in gids:
        inc = sorted((day, amount) for (node, day), amount in day_in.items() if node == gid)
        outs = sorted((day, amount) for (node, day), amount in day_out.items() if node == gid)
        total_in = sum(float(x[1]) for x in inc)
        total_out = sum(float(x[1]) for x in outs)
        same = sum(min(float(day_in.get((gid, day), 0)), float(day_out.get((gid, day), 0))) for day, _ in inc)
        relay = sum(min(amount, sum(out_amt for out_day, out_amt in outs if pd.Timedelta(0) <= out_day-day <= pd.Timedelta(days=2)))
                    for day, amount in inc)
        # Incoming graph volume is censored for seeds, making pass-through ratios unusable.
        valid_ratio = (not bool(is_seed.get(gid, False))) and total_in > 0
        pass_ratio = total_out / total_in if valid_ratio else np.nan
        same_ratio = same / total_in if valid_ratio else np.nan
        relay_ratio = min(1.0, relay / total_in) if valid_ratio else np.nan
        rows.append({"gid": gid, "pass_through": pass_ratio,
                     "retention": max(0.0, 1.0 - pass_ratio) if np.isfinite(pass_ratio) else np.nan,
                     "same_day_flow_ratio": same_ratio, "relay_2d_ratio": relay_ratio,
                     "balance_score": max(0.0, 1.0 - abs(1.0 - pass_ratio)) if np.isfinite(pass_ratio) else np.nan})
    return pd.DataFrame(rows)


def build_features(graph: nx.DiGraph, nodes: pd.DataFrame, edges: pd.DataFrame,
                   tx: pd.DataFrame) -> pd.DataFrame:
    """Derive reproducible graph, temporal and peer-relative measures locally."""
    gids = nodes.gid.tolist()
    is_seed = nodes.set_index("gid").is_seed.fillna(False).astype(bool).to_dict()
    in_deg, out_deg = dict(graph.in_degree()), dict(graph.out_degree())
    in_kzt, out_kzt = dict(graph.in_degree(weight="sum_kzt")), dict(graph.out_degree(weight="sum_kzt"))
    in_tx, out_tx = dict(graph.in_degree(weight="n_tx")), dict(graph.out_degree(weight="n_tx"))
    try:
        pagerank = nx.pagerank(graph, weight="sum_kzt")
    except nx.PowerIterationFailedConvergence:
        pagerank = nx.pagerank(graph, weight=None, max_iter=1000)
    # Approximation keeps runtime practical while remaining deterministic on this graph size.
    k = min(500, len(gids)) if len(gids) > 500 else None
    between = nx.betweenness_centrality(graph, normalized=True, k=k, seed=42) if k else nx.betweenness_centrality(graph, normalized=True)
    seed_ids = [g for g in gids if is_seed.get(g, False)]
    reach = {}
    for gid in gids:
        ancestors = nx.ancestors(graph, gid)
        reach[gid] = len((ancestors | ({gid} if is_seed.get(gid, False) else set())) & set(seed_ids))

    result = nodes[["gid", "depth", "is_seed"]].copy()
    for name, mapping in (("in_deg", in_deg), ("out_deg", out_deg), ("in_kzt", in_kzt),
                          ("out_kzt", out_kzt), ("in_tx", in_tx), ("out_tx", out_tx),
                          ("pagerank", pagerank), ("betweenness", between),
                          ("seed_reach_count", reach)):
        result[name] = result.gid.map(mapping).fillna(0)
    result["total_kzt"] = result.in_kzt + result.out_kzt
    result["truncated_by_depth"] = (result.depth >= 4) & result.out_deg.eq(0)
    result["fanout_share"] = result.out_deg / (result.in_deg + result.out_deg).replace(0, np.nan)
    result["fanout_share"] = result.fanout_share.fillna(0)
    temporal = _temporal_metrics(tx, gids, is_seed)
    result = result.merge(temporal, on="gid", validate="one_to_one")
    # Peer anomaly is an interpretable percentile average over existing graph-volume signals.
    peer_raw = np.log1p(result["in_kzt"]) + np.log1p(result["out_kzt"]) + np.log1p(result["in_deg"] + result["out_deg"])
    result["peer_anomaly_score"] = peer_raw.rank(method="average", pct=True).fillna(0).clip(0, 1)
    return result


def run(data_dir: Path, out_dir: Path) -> dict[str, int]:
    started = time.perf_counter()
    edges, nodes, tx = load_inputs(data_dir)
    if len(nodes) != 2248:
        raise ValueError(f"Expected supplied dataset to contain 2248 nodes, found {len(nodes)}")
    print(f"validation: {len(nodes)} nodes, {len(edges)} edges, {len(tx)} transactions")
    graph = build_graph(edges, nodes)
    features = build_features(graph, nodes, edges, tx)
    assignments, _ = cluster_graph(graph, nodes)
    features = features.merge(assignments, on="gid", validate="one_to_one")
    role_frame = assign_roles(features)
    features = features.merge(role_frame, on="gid", validate="one_to_one")
    features = score_priority(features, role_frame)
    # Add feature percentiles used by coordinator evidence strings.
    for name in ("betweenness", "pagerank"):
        features[f"{name}_percentile"] = features[name].rank(method="average", pct=True).fillna(0)
    features = add_evidence(features)
    clusters = cluster_summaries(features, graph, nodes)
    resilience = resilience_analysis(graph, features)
    counts = write_exports(features, clusters, out_dir)
    resilience.to_csv(out_dir / "resilience.csv", index=False)
    validate_submission(out_dir, expected_nodes=2248, source_nodes=nodes, features=features)
    print(f"output rows: nodes_roles={counts['nodes_roles']}, clusters={counts['clusters']}, top_nodes={counts['top_nodes']}, resilience={len(resilience)}")
    print(f"runtime: {time.perf_counter() - started:.2f}s")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("./data"))
    parser.add_argument("--out", type=Path, default=Path("./out"))
    args = parser.parse_args()
    run(args.data, args.out)


if __name__ == "__main__":
    main()
