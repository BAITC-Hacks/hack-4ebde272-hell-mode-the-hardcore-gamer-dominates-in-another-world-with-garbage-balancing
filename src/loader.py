"""Parquet loading entry points for Money Graph inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import validate_inputs


def load_inputs(data_dir: str | Path, *, validate: bool = True):
    """Load nodes, edges, and transactions from a dataset directory."""
    root = Path(data_dir)
    paths = {name: root / f"{name}.parquet" for name in ("nodes", "edges", "transactions")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required parquet file(s): {', '.join(missing)}")
    nodes = pd.read_parquet(paths["nodes"])
    edges = pd.read_parquet(paths["edges"])
    transactions = pd.read_parquet(paths["transactions"])
    if validate:
        return validate_inputs(nodes, edges, transactions)
    return nodes, edges, transactions


# A short alias is convenient for callers that use the original starter name.
load = load_inputs
