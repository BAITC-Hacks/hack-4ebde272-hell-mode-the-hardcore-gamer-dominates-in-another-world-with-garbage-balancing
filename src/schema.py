"""Input contracts and validation for the Money Graph feature layer."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


NODE_COLUMNS = ("gid", "depth", "is_seed")
EDGE_COLUMNS = ("src", "dst", "sum_kzt", "n_tx", "depth")
TRANSACTION_COLUMNS = ("src", "dst", "date", "sum_kzt")


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def validate_inputs(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate schemas and cross-file consistency, returning normalized copies.

    Transaction dates are parsed to pandas datetimes. Amount checks permit only
    finite, strictly positive values. Aggregated edge amounts are compared with
    transaction totals using ``rtol``/``atol`` and edge transaction counts.
    """
    if not all(isinstance(frame, pd.DataFrame) for frame in (nodes, edges, transactions)):
        raise TypeError("nodes, edges, and transactions must be pandas DataFrames")
    _require_columns(nodes, NODE_COLUMNS, "nodes")
    _require_columns(edges, EDGE_COLUMNS, "edges")
    _require_columns(transactions, TRANSACTION_COLUMNS, "transactions")

    nodes = nodes.loc[:, NODE_COLUMNS].copy()
    edges = edges.loc[:, EDGE_COLUMNS].copy()
    transactions = transactions.loc[:, TRANSACTION_COLUMNS].copy()

    if nodes["gid"].isna().any():
        raise ValueError("nodes.gid contains null values")
    if nodes["gid"].duplicated().any():
        duplicates = nodes.loc[nodes["gid"].duplicated(keep=False), "gid"].head(5).tolist()
        raise ValueError(f"nodes.gid must be unique; duplicate gid(s): {duplicates}")
    node_depth = pd.to_numeric(nodes["depth"], errors="coerce")
    if node_depth.isna().any() or not np.isfinite(node_depth.to_numpy(dtype=float)).all():
        raise ValueError("nodes.depth must contain finite integers from 0 through 4")
    if (node_depth % 1 != 0).any() or not node_depth.between(0, 4).all():
        raise ValueError("nodes.depth must contain finite integers from 0 through 4")
    nodes["depth"] = node_depth.astype("int64")
    if nodes["is_seed"].isna().any() or not nodes["is_seed"].isin([True, False, 0, 1]).all():
        raise ValueError("nodes.is_seed must contain only boolean values")
    nodes["is_seed"] = nodes["is_seed"].astype(bool)
    if edges[["src", "dst"]].isna().any().any():
        raise ValueError("edges.src and edges.dst must not contain null values")
    if transactions[["src", "dst"]].isna().any().any():
        raise ValueError("transactions.src and transactions.dst must not contain null values")
    known = set(nodes["gid"])
    if not set(edges["src"]).issubset(known) or not set(edges["dst"]).issubset(known):
        unknown = (set(edges["src"]) | set(edges["dst"])) - known
        raise ValueError(f"edges reference gid(s) absent from nodes: {sorted(unknown, key=str)[:5]}")
    if not set(transactions["src"]).issubset(known) or not set(transactions["dst"]).issubset(known):
        unknown = (set(transactions["src"]) | set(transactions["dst"])) - known
        raise ValueError(f"transactions reference gid(s) absent from nodes: {sorted(unknown, key=str)[:5]}")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("edges must contain one aggregated row per directed (src, dst) pair")

    for frame, column, label in (
        (edges, "sum_kzt", "edges.sum_kzt"),
        (transactions, "sum_kzt", "transactions.sum_kzt"),
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{label} must contain finite numeric amounts")
        if (values <= 0).any():
            raise ValueError(f"{label} must be strictly positive")
        frame[column] = values.astype(float)

    edge_counts = pd.to_numeric(edges["n_tx"], errors="coerce")
    if edge_counts.isna().any() or not np.isfinite(edge_counts.to_numpy(dtype=float)).all():
        raise ValueError("edges.n_tx must contain finite positive integers")
    if (edge_counts <= 0).any() or (edge_counts % 1 != 0).any():
        raise ValueError("edges.n_tx must contain finite positive integers")
    edges["n_tx"] = edge_counts.astype("int64")

    dates = pd.to_datetime(transactions["date"], errors="coerce", utc=True, format="mixed")
    if dates.isna().any():
        bad = transactions.loc[dates.isna(), "date"].head(5).tolist()
        raise ValueError(f"transactions.date contains invalid or missing date(s): {bad}")
    transactions["date"] = dates.dt.tz_convert(None).dt.normalize()

    if transactions.empty and not edges.empty:
        raise ValueError("edges contain rows but transactions are empty")
    tx_agg = (
        transactions.groupby(["src", "dst"], sort=False)
        .agg(tx_sum=("sum_kzt", "sum"), tx_count=("sum_kzt", "size"))
    )
    edge_agg = edges.set_index(["src", "dst"])[["sum_kzt", "n_tx"]]
    all_pairs = edge_agg.index.union(tx_agg.index)
    aligned_edges = edge_agg.reindex(all_pairs)
    aligned_tx = tx_agg.reindex(all_pairs)
    missing_edge = aligned_edges["n_tx"].isna()
    missing_tx = aligned_tx["tx_count"].isna()
    if missing_edge.any() or missing_tx.any():
        absent = all_pairs[missing_edge | missing_tx][:5].tolist()
        raise ValueError(f"aggregated edges and transactions have different directed pairs; examples: {absent}")
    count_mismatch = aligned_edges["n_tx"].to_numpy() != aligned_tx["tx_count"].to_numpy()
    if count_mismatch.any():
        pair = all_pairs[int(np.flatnonzero(count_mismatch)[0])]
        raise ValueError(
            f"edges.n_tx does not match transaction row count for directed pair {pair}: "
            f"edge={int(aligned_edges.loc[pair, 'n_tx'])}, "
            f"transactions={int(aligned_tx.loc[pair, 'tx_count'])}"
        )
    edge_sums = aligned_edges["sum_kzt"].to_numpy(dtype=float)
    tx_sums = aligned_tx["tx_sum"].to_numpy(dtype=float)
    close = np.isclose(edge_sums, tx_sums, rtol=rtol, atol=atol)
    if not close.all():
        idx = int(np.flatnonzero(~close)[0])
        pair = all_pairs[idx]
        raise ValueError(
            f"edges.sum_kzt does not match transaction total for directed pair {pair}: "
            f"edge={edge_sums[idx]:.12g}, transactions={tx_sums[idx]:.12g}"
        )

    return nodes, edges, transactions
