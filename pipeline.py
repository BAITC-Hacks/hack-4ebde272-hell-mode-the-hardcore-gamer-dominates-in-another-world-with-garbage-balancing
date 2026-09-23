"""Local, deterministic AML decision and export pipeline."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import networkx as nx
import pandas as pd

from src.clustering import cluster_graph
from src.roles import assign_roles
from src.priority import score_priority
from src.explanations import add_evidence
from src.exports import cluster_summaries, write_exports
from src.resilience import resilience_analysis
from src.graph_features import build_features as graph_features, build_graph as directed_graph
from src.loader import load_inputs as validated_inputs
from validate_submission import validate_submission


def load_inputs(data_dir: Path):
    """Use the shared validated loader, preserving the pipeline argument order."""
    nodes, edges, tx = validated_inputs(data_dir)
    return edges, nodes, tx


def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame) -> nx.DiGraph:
    return directed_graph(nodes, edges)


def build_features(graph: nx.DiGraph, nodes: pd.DataFrame, edges: pd.DataFrame,
                   tx: pd.DataFrame) -> pd.DataFrame:
    """Adapt the shared feature layer to the decision engine's extra fields."""
    result = graph_features(nodes, edges, tx).reset_index()
    result["truncated_by_depth"] = result["boundary_censored"]
    result["retention"] = (1.0 - result["pass_through"]).clip(0, 1)
    result["balance_score"] = (1.0 - (1.0 - result["pass_through"]).abs()).clip(0, 1)
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
