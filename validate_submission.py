"""Mechanical checks for the required analyst-review exports."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.roles import ROLES


def validate_submission(out_dir: str | Path, expected_nodes=2248, source_nodes=None, features=None):
    out = Path(out_dir)
    nodes = pd.read_csv(out / "nodes_roles.csv")
    clusters = pd.read_csv(out / "clusters.csv")
    top = pd.read_csv(out / "top_nodes.csv")
    required = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
    assert len(nodes) == expected_nodes, f"expected {expected_nodes} node rows, found {len(nodes)}"
    assert nodes.gid.nunique() == expected_nodes, "gid values must be unique"
    assert not nodes[required].isna().any().any(), "required node fields contain nulls"
    assert nodes.role.isin(ROLES).all(), "invalid role found"
    assert np.isfinite(nodes[["role_score", "priority_score"]].to_numpy(dtype=float)).all()
    assert nodes[["role_score", "priority_score"]].ge(0).all().all()
    assert nodes[["role_score", "priority_score"]].le(1).all().all()
    assert nodes.cluster_id.notna().all()
    evidence = nodes.evidence.astype(str)
    assert evidence.str.len().gt(0).all()
    assert evidence.str.contains(r"\d").all(), "evidence must contain a numeric value"
    assert evidence.str.len().le(200).all()
    if {"depth", "out_deg"}.issubset(nodes.columns):
        cutoff = (pd.to_numeric(nodes.depth, errors="coerce") >= 4) & (pd.to_numeric(nodes.out_deg, errors="coerce") == 0)
        assert not (cutoff & nodes.role.eq("terminal")).any(), "depth-4 outflow cutoff classified terminal"
    assert set(clusters.columns) >= {"cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"}
    assert int(clusters.n_nodes.sum()) == expected_nodes, "cluster node counts must sum to node count"
    assert len(top) >= 20, "top list must contain at least 20 nodes"
    assert top["rank"].tolist() == list(range(1, len(top) + 1)), "top ranks must be contiguous"
    assert top.priority_score.is_monotonic_decreasing, "top list priority must descend"
    assert top.gid.is_unique, "top list contains duplicate gids"
    assert len(top) <= 50
    if source_nodes is not None:
        expected = set(source_nodes.gid)
        assert set(nodes.gid) == expected, "export gids do not match source nodes"
        if features is not None:
            # A hop-4 boundary with no observed outflow is censored; it cannot qualify as terminal.
            truncated = features.loc[(features.depth >= 4) & (features.out_deg == 0)]
            assert not truncated.role.eq("terminal").any(), "depth-4 outflow cutoff classified terminal"
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("./out"))
    parser.add_argument("--expected-nodes", type=int, default=2248)
    args = parser.parse_args()
    validate_submission(args.out, expected_nodes=args.expected_nodes)
    print("submission validation: OK")


if __name__ == "__main__":
    main()
