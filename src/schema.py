"""Input contracts and exact-identifier validation for Money Graph inputs."""

from __future__ import annotations

from collections.abc import Iterable
import math
from numbers import Integral, Number, Real
import re

import numpy as np
import pandas as pd


NODE_COLUMNS = ("gid", "depth", "is_seed")
EDGE_COLUMNS = ("src", "dst", "sum_kzt", "n_tx", "depth")
TRANSACTION_COLUMNS = ("src", "dst", "date", "sum_kzt")
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_SAFE_FLOAT_LIMIT = 2**53
_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+\Z")


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    if frame.columns.duplicated().any():
        raise ValueError(f"{name} contains duplicate column names")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def _integer_column(
    values: pd.Series,
    label: str,
    *,
    minimum: int = _INT64_MIN,
    maximum: int = _INT64_MAX,
) -> pd.Series:
    """Parse integers without rounding identifiers through a float dtype.

    Integral floats below 2**53 are accepted for small synthetic/CSV inputs.
    Larger floats are rejected even when integral: their original integer may
    already have been rounded before reaching this function. Exact large gids
    must arrive as integers or decimal integer strings.
    """
    parsed = []
    for position, value in enumerate(values.tolist()):
        reason = None
        if isinstance(value, (bool, np.bool_)):
            reason = "boolean values are not integers"
        elif isinstance(value, Integral):
            integer = int(value)
        elif isinstance(value, str) and _INTEGER_TEXT.fullmatch(value.strip()):
            integer = int(value.strip())
        elif isinstance(value, (float, np.floating)):
            if not np.isfinite(value) or not value.is_integer():
                reason = "value must be a finite integer"
            elif abs(value) >= _SAFE_FLOAT_LIMIT:
                reason = "unsafe floating-point integer; use an exact integer or decimal integer string"
            else:
                integer = int(value)
        else:
            reason = "value must be a non-null integer or decimal integer string"
        if reason is None and not minimum <= integer <= maximum:
            reason = f"integer must be between {minimum} and {maximum}"
        if reason is not None:
            raise ValueError(f"{label} has invalid integer at row position {position}: {value!r}; {reason}")
        parsed.append(integer)
    return pd.Series(parsed, index=values.index, dtype="int64", name=values.name)


def _positive_amounts(values: pd.Series, label: str) -> pd.Series:
    """Normalize strictly positive finite amounts; booleans are not amounts."""
    invalid_type = values.map(
        lambda value: not isinstance(value, (Number, str))
        or isinstance(value, (bool, np.bool_, complex, np.complexfloating))
    )
    if invalid_type.any():
        raise ValueError(f"{label} must contain finite numeric amounts, not booleans or other nonnumeric values")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain finite numeric amounts")
    if (numeric <= 0).any():
        raise ValueError(f"{label} must be strictly positive")
    return numeric.astype(float)


def _transaction_totals(transactions: pd.DataFrame) -> pd.DataFrame:
    """Reconcile each pair using a canonical, accurately summed amount order."""
    pairs = []
    totals = []
    counts = []
    for pair, amounts in transactions.groupby(["src", "dst"], sort=True)["sum_kzt"]:
        try:
            total = math.fsum(sorted(amounts.tolist()))
        except OverflowError as error:
            raise ValueError(f"transactions.sum_kzt total overflows for directed pair {pair}") from error
        pairs.append(pair)
        totals.append(total)
        counts.append(len(amounts))
    return pd.DataFrame(
        {"tx_sum": pd.Series(totals, dtype=float).to_numpy(),
         "tx_count": pd.Series(counts, dtype="int64").to_numpy()},
        index=pd.MultiIndex.from_tuples(pairs, names=["src", "dst"]),
    )


def validate_inputs(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate cross-file consistency, returning normalized copies in row order.

    Gids/endpoints and counts become exact signed int64 columns; identifiers may
    be negative or zero. Node depths are 0..4 and edge sampling depths are 1..4.
    Extra columns are omitted. Supplied-dataset counts, date window and minimum
    transfer threshold are deliberately not enforced on reusable small inputs.

    Dates are parsed as UTC and normalized to timezone-naive calendar dates;
    numeric epoch values are rejected. Amounts must be finite and positive.
    Canonical transaction sums match edge amounts under ``rtol``/``atol`` and
    counts match exactly. Inputs are never mutated.
    """
    if not all(isinstance(frame, pd.DataFrame) for frame in (nodes, edges, transactions)):
        raise TypeError("nodes, edges, and transactions must be pandas DataFrames")
    for tolerance, label in ((rtol, "rtol"), (atol, "atol")):
        if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real):
            raise ValueError(f"{label} must be a finite nonnegative number")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"{label} must be a finite nonnegative number")
    _require_columns(nodes, NODE_COLUMNS, "nodes")
    _require_columns(edges, EDGE_COLUMNS, "edges")
    _require_columns(transactions, TRANSACTION_COLUMNS, "transactions")

    nodes = nodes.loc[:, NODE_COLUMNS].copy()
    edges = edges.loc[:, EDGE_COLUMNS].copy()
    transactions = transactions.loc[:, TRANSACTION_COLUMNS].copy()
    for frame, columns, name in (
        (nodes, ("gid",), "nodes"),
        (edges, ("src", "dst"), "edges"),
        (transactions, ("src", "dst"), "transactions"),
    ):
        for column in columns:
            frame[column] = _integer_column(frame[column], f"{name}.{column}")
    if nodes["gid"].duplicated().any():
        duplicates = nodes.loc[nodes["gid"].duplicated(keep=False), "gid"].head(5).tolist()
        raise ValueError(f"nodes.gid must be unique; duplicate gid(s): {duplicates}")
    nodes["depth"] = _integer_column(nodes["depth"], "nodes.depth", minimum=0, maximum=4)
    edges["depth"] = _integer_column(edges["depth"], "edges.depth", minimum=1, maximum=4)
    if nodes["is_seed"].isna().any() or not nodes["is_seed"].isin([True, False, 0, 1]).all():
        raise ValueError("nodes.is_seed must contain only boolean values")
    nodes["is_seed"] = nodes["is_seed"].astype(bool)
    known = set(nodes["gid"])
    for frame, name in ((edges, "edges"), (transactions, "transactions")):
        unknown = (set(frame["src"]) | set(frame["dst"])) - known
        if unknown:
            raise ValueError(f"{name} reference gid(s) absent from nodes: {sorted(unknown)[:5]}")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("edges must contain one aggregated row per directed (src, dst) pair")

    edges["sum_kzt"] = _positive_amounts(edges["sum_kzt"], "edges.sum_kzt")
    transactions["sum_kzt"] = _positive_amounts(transactions["sum_kzt"], "transactions.sum_kzt")
    edges["n_tx"] = _integer_column(edges["n_tx"], "edges.n_tx", minimum=1)

    raw_dates = transactions["date"]
    numeric_dates = pd.Series(
        [isinstance(value, (Number, np.bool_)) for value in raw_dates],
        index=raw_dates.index,
        dtype=bool,
    )
    if numeric_dates.any():
        bad = raw_dates.loc[numeric_dates].head(5).tolist()
        raise ValueError(f"transactions.date contains invalid numeric date(s): {bad}; use calendar dates")
    try:
        dates = pd.to_datetime(raw_dates, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("transactions.date contains invalid or missing date(s)") from error
    if dates.isna().any():
        bad = raw_dates.loc[dates.isna()].head(5).tolist()
        raise ValueError(f"transactions.date contains invalid or missing date(s): {bad}")
    transactions["date"] = dates.dt.tz_convert(None).dt.normalize()

    if transactions.empty and not edges.empty:
        raise ValueError("edges contain rows but transactions are empty")
    tx_agg = _transaction_totals(transactions)
    edge_agg = edges.set_index(["src", "dst"])[["sum_kzt", "n_tx"]]
    all_pairs = edge_agg.index.union(tx_agg.index).sort_values()
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
