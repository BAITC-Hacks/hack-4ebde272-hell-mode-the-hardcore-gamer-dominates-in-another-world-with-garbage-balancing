"""Strict submission schemas and cross-file consistency checks.

The CSVs are authoritative for their published fields. The auxiliary Parquet
supplies observation context for seed and boundary checks. Pass raw edges (or a
directed graph) to independently reconcile internal turnover; the CLI does this
with ``--data``. No check relies on Python's removable ``assert`` statement.
"""
from __future__ import annotations

import argparse
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

from src.exports import CLUSTER_COLUMNS, NODE_COLUMNS, TOP_COLUMNS, exact_int64
from src.priority import WEIGHTS as PRIORITY_WEIGHTS
from src.roles import ROLES


class SubmissionValidationError(ValueError):
    """A submission violates a schema, observation rule or cross-file contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SubmissionValidationError(message)


def _integers(values: pd.Series, label: str, minimum: int | None = None) -> pd.Series:
    try:
        result = exact_int64(values, label)
    except ValueError as exc:
        raise SubmissionValidationError(str(exc)) from exc
    if minimum is not None:
        _require(result.ge(minimum).all(), f"{label} must be >= {minimum}")
    return result


def _numbers(values: pd.Series, label: str, *, bounded: bool = False) -> pd.Series:
    _require(not values.map(lambda value: isinstance(value, (bool, np.bool_))).any(),
             f"{label} must contain numbers, not booleans")
    result = pd.to_numeric(values, errors="coerce")
    _require(np.isfinite(result.to_numpy(dtype=float)).all(), f"{label} must contain finite numbers")
    _require(result.ge(0).all(), f"{label} must be nonnegative")
    if bounded:
        _require(result.le(1).all(), f"{label} must be in [0,1]")
    return result


def _text(values: pd.Series, label: str, *, numeric: bool = False, maximum: int | None = None) -> None:
    _require(values.map(lambda value: isinstance(value, str) and bool(value.strip())).all(),
             f"{label} must contain nonempty text")
    if numeric:
        _require(values.str.contains(r"[0-9]").all(), f"{label} must contain a numeric value")
    if maximum is not None:
        _require(values.str.len().le(maximum).all(), f"{label} must be <= {maximum} characters")


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        # String-first parsing protects large int64 gids and singleton top_gids.
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SubmissionValidationError(f"Cannot read {path.name}: {exc}") from exc
    _require(frame.columns.tolist() == columns, f"{path.name} columns must be exactly {columns}, in that order")
    _require(not frame.eq("").any().any(), f"{path.name} contains empty required fields")
    return frame


def _keyed(frame: pd.DataFrame, label: str, expected_gids: set[int]) -> pd.DataFrame:
    frame = frame.copy()
    if "gid" not in frame and frame.index.name == "gid":
        frame = frame.reset_index()
    _require("gid" in frame, f"{label} is missing gid")
    frame["gid"] = _integers(frame.gid, f"{label}.gid")
    _require(frame.gid.is_unique, f"{label}.gid must be unique")
    _require(set(frame.gid) == expected_gids, f"{label} gids do not match node exports")
    return frame.set_index("gid").sort_index()


def _seed_flags(values: pd.Series, label: str) -> pd.Series:
    _require(values.notna().all() and values.isin([True, False, 0, 1]).all(),
             f"{label} must contain boolean values")
    return values.astype(bool)


def _compare_fields(reference: pd.DataFrame, other: pd.DataFrame, label: str, columns: list[str]) -> None:
    for column in columns:
        if column not in other:
            continue
        if column in ("role_score", "priority_score", "internal_out_kzt"):
            values = _numbers(other[column], f"{label}.{column}", bounded=column.endswith("score"))
            equal = np.isclose(reference[column].to_numpy(dtype=float), values.to_numpy(dtype=float), rtol=1e-12, atol=1e-12).all()
        else:
            equal = reference[column].eq(other[column]).all()
        _require(equal, f"{label}.{column} disagrees with authoritative node data")


def _internal_turnover(source_edges, graph, node_table: pd.DataFrame) -> pd.Series | None:
    if source_edges is None and graph is None:
        return None
    _require(source_edges is None or graph is None, "Supply source_edges or graph, not both")
    if graph is not None:
        _require(graph.is_directed(), "Turnover verification requires a directed graph")
        _require(set(graph.nodes) == set(node_table.index), "graph gids do not match node exports")
        source_edges = pd.DataFrame([(src, dst, data.get("sum_kzt", np.nan))
                                     for src, dst, data in graph.edges(data=True)],
                                    columns=["src", "dst", "sum_kzt"])
    _require({"src", "dst", "sum_kzt"}.issubset(source_edges.columns),
             "source_edges requires src, dst, sum_kzt")
    edges = source_edges[["src", "dst", "sum_kzt"]].copy()
    edges["src"] = _integers(edges.src, "source_edges.src")
    edges["dst"] = _integers(edges.dst, "source_edges.dst")
    edges["sum_kzt"] = _numbers(edges.sum_kzt, "source_edges.sum_kzt")
    _require(set(edges.src).union(edges.dst).issubset(node_table.index), "source_edges contains unknown gids")
    cluster_of = node_table.cluster_id.to_dict()
    amounts = {gid: [] for gid in node_table.index}
    for src, dst, amount in edges.itertuples(index=False, name=None):
        if cluster_of[src] == cluster_of[dst]:
            amounts[src].append(float(amount))
    return pd.Series({gid: fsum(sorted(values)) for gid, values in amounts.items()}, dtype=float).reindex(node_table.index)


def validate_submission(out_dir: str | Path, expected_nodes=2248, source_nodes=None, features=None,
                        *, source_edges=None, graph=None):
    """Validate the full export contract; return True or raise a useful error.

    Without raw edges the turnover check reconciles ``internal_out_kzt`` from
    node_features. For an independent source-data check, supply source_edges or
    graph. Existing pipeline callers may still pass source_nodes and features.
    The brief requires at least 20 top rows; this validator imposes no 50-row cap.
    """
    out = Path(out_dir)
    nodes = _read_csv(out / "nodes_roles.csv", NODE_COLUMNS)
    clusters = _read_csv(out / "clusters.csv", CLUSTER_COLUMNS)
    top = _read_csv(out / "top_nodes.csv", TOP_COLUMNS)
    _require(len(nodes) == expected_nodes, f"expected {expected_nodes} node rows, found {len(nodes)}")
    nodes["gid"] = _integers(nodes.gid, "nodes.gid")
    nodes["cluster_id"] = _integers(nodes.cluster_id, "nodes.cluster_id", 0)
    _require(nodes.gid.is_unique, "nodes.gid values must be unique")
    _require(nodes.role.isin(ROLES).all(), "nodes.role contains an invalid role")
    for name in ("role_score", "priority_score"):
        nodes[name] = _numbers(nodes[name], f"nodes.{name}", bounded=True)
    _text(nodes.evidence, "nodes.evidence", numeric=True, maximum=200)
    table = nodes.set_index("gid").sort_index()

    try:
        auxiliary = pd.read_parquet(out / "node_features.parquet")
    except (OSError, ValueError) as exc:
        raise SubmissionValidationError(f"Cannot read node_features.parquet: {exc}") from exc
    _require("gid" in auxiliary and str(auxiliary.gid.dtype) == "int64", "node_features.gid must have int64 dtype")
    required_auxiliary = set(NODE_COLUMNS) | {"is_seed", "depth", "out_deg", "internal_out_kzt", "priority_explanation"}
    _require(required_auxiliary.issubset(auxiliary.columns),
             f"node_features is missing required columns: {sorted(required_auxiliary - set(auxiliary.columns))}")
    auxiliary = _keyed(auxiliary, "node_features", set(nodes.gid))
    _compare_fields(table, auxiliary, "node_features", NODE_COLUMNS[1:])
    auxiliary["is_seed"] = _seed_flags(auxiliary.is_seed, "node_features.is_seed")
    auxiliary["depth"] = _integers(auxiliary.depth, "node_features.depth", 0)
    _require(auxiliary.depth.le(4).all(), "node_features.depth must be <= 4")
    auxiliary["out_deg"] = _integers(auxiliary.out_deg, "node_features.out_deg", 0)
    auxiliary["internal_out_kzt"] = _numbers(auxiliary.internal_out_kzt, "node_features.internal_out_kzt")
    _text(auxiliary.priority_explanation, "node_features.priority_explanation", numeric=True)
    contribution_columns = [f"priority_{name}_contribution" for name in PRIORITY_WEIGHTS]
    if any(column in auxiliary for column in contribution_columns):
        _require(set(contribution_columns).issubset(auxiliary.columns),
                 "node_features must include every priority contribution")
        contributions = pd.DataFrame({column: _numbers(auxiliary[column], f"node_features.{column}", bounded=True)
                                      for column in contribution_columns})
        _require(np.isclose(contributions.sum(axis=1), table.priority_score, rtol=1e-12, atol=1e-12).all(),
                 "node_features priority contributions do not sum to priority_score")

    for source, label in ((source_nodes, "source_nodes"), (features, "features")):
        if source is not None:
            context = _keyed(source, label, set(nodes.gid))
            _compare_fields(table, context, label, NODE_COLUMNS[1:])
            _compare_fields(auxiliary, context, label, ["is_seed", "depth", "out_deg", "internal_out_kzt"])
    _require(not (table.role.eq("terminal") & auxiliary.depth.ge(4)).any(),
             "depth-4 outflow cutoff classified terminal")
    _require(not (table.role.eq("terminal") & auxiliary.is_seed).any(), "seed client classified terminal")

    for name, minimum in (("cluster_id", 0), ("n_nodes", 1), ("n_seed", 0)):
        clusters[name] = _integers(clusters[name], f"clusters.{name}", minimum)
    _require(clusters.cluster_id.is_unique, "clusters.cluster_id must be unique")
    clusters["sum_kzt_internal"] = _numbers(clusters.sum_kzt_internal, "clusters.sum_kzt_internal")
    _text(clusters.hypothesis, "clusters.hypothesis")
    _require(set(clusters.cluster_id) == set(nodes.cluster_id), "node cluster references and cluster rows must match exactly")
    _require(int(clusters.n_nodes.sum()) == expected_nodes, "cluster node counts must sum to node count")
    raw_internal = _internal_turnover(source_edges, graph, table)
    if raw_internal is not None:
        _require(np.isclose(raw_internal, auxiliary.internal_out_kzt, rtol=1e-12, atol=1e-6).all(),
                 "node_features.internal_out_kzt disagrees with source edges")
    for row in clusters.itertuples(index=False):
        members = table.loc[table.cluster_id.eq(row.cluster_id)]
        _require(row.n_nodes == len(members), f"cluster {row.cluster_id} n_nodes does not match members")
        _require(row.n_seed == int(auxiliary.loc[members.index, "is_seed"].sum()),
                 f"cluster {row.cluster_id} n_seed does not match seed members")
        expected_turnover = fsum(sorted(auxiliary.loc[members.index, "internal_out_kzt"].tolist()))
        _require(np.isclose(row.sum_kzt_internal, expected_turnover, rtol=1e-12, atol=1e-6),
                 f"cluster {row.cluster_id} internal turnover does not match member outflow")
        listed = _integers(pd.Series(row.top_gids.split(",")), f"cluster {row.cluster_id} top_gids").tolist()
        expected_top = members.reset_index().sort_values(["priority_score", "gid"], ascending=[False, True], kind="mergesort").gid.head(5).tolist()
        _require(listed == expected_top, f"cluster {row.cluster_id} top_gids must be its top five members in deterministic order")

    _require(len(top) >= 20, "top list must contain at least 20 nodes")
    _require(len(top) <= len(nodes), "top list cannot contain more rows than nodes")
    top["rank"] = _integers(top["rank"], "top.rank", 1)
    top["gid"] = _integers(top.gid, "top.gid")
    top["priority_score"] = _numbers(top.priority_score, "top.priority_score", bounded=True)
    _require(top.priority_score.is_monotonic_decreasing, "top priority scores must be monotonically descending")
    _require(top["rank"].tolist() == list(range(1, len(top) + 1)), "top ranks must be contiguous starting at 1")
    _require(top.gid.is_unique, "top list contains duplicate gids")
    _require(set(top.gid).issubset(table.index), "top list references unknown gids")
    _require(top.role.isin(ROLES).all(), "top.role contains an invalid role")
    _text(top.why, "top.why", numeric=True)
    expected_top = nodes.sort_values(["priority_score", "gid"], ascending=[False, True], kind="mergesort").head(len(top))
    _require(top.gid.tolist() == expected_top.gid.tolist(), "top gids must follow descending priority then ascending gid")
    actual_top = top.set_index("gid").sort_index()
    _compare_fields(table.loc[actual_top.index], actual_top, "top", ["role", "priority_score"])
    _require(actual_top.why.eq(auxiliary.loc[actual_top.index, "priority_explanation"]).all(),
             "top.why must match the independent priority explanation")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("./out"))
    parser.add_argument("--data", type=Path, default=Path("./data"), help="Raw supplied node/edge Parquet directory")
    parser.add_argument("--expected-nodes", type=int, default=2248)
    args = parser.parse_args()
    source_nodes = pd.read_parquet(args.data / "nodes.parquet")
    source_edges = pd.read_parquet(args.data / "edges.parquet")
    validate_submission(args.out, expected_nodes=args.expected_nodes, source_nodes=source_nodes, source_edges=source_edges)
    print("submission validation: OK")


if __name__ == "__main__":
    main()
