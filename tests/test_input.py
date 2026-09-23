"""Input validation contract tests."""

import pandas as pd
import pytest

from src.schema import validate_inputs


@pytest.fixture
def valid_data():
    nodes = pd.DataFrame({"gid": [1, 2, 3], "depth": [0, 1, 4], "is_seed": [True, False, False]})
    edges = pd.DataFrame({
        "src": [1, 2], "dst": [2, 1], "sum_kzt": [12.5, 2.0], "n_tx": [2, 1], "depth": [1, 1]
    })
    transactions = pd.DataFrame({
        "src": [1, 1, 2], "dst": [2, 2, 1], "date": ["2026-01-01", "2026-01-02", "2026-01-01"],
        "sum_kzt": [5.0, 7.5, 2.0],
    })
    return nodes, edges, transactions


def test_accepts_consistent_inputs_and_normalizes_dates(valid_data):
    nodes, edges, transactions = validate_inputs(*valid_data)

    assert len(nodes) == 3
    assert edges["n_tx"].tolist() == [2, 1]
    assert str(transactions["date"].dtype).startswith("datetime64")


def test_rejects_duplicate_node_gid(valid_data):
    nodes, edges, transactions = valid_data
    nodes = pd.concat([nodes, nodes.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="gid must be unique"):
        validate_inputs(nodes, edges, transactions)


def test_rejects_unknown_edge_endpoint(valid_data):
    nodes, edges, transactions = valid_data
    edges.loc[0, "dst"] = 999
    transactions.loc[0, "dst"] = 999

    with pytest.raises(ValueError, match="edges reference gid"):
        validate_inputs(nodes, edges, transactions)


def test_rejects_nonpositive_transaction_amount(valid_data):
    nodes, edges, transactions = valid_data
    transactions.loc[0, "sum_kzt"] = 0

    with pytest.raises(ValueError, match="strictly positive"):
        validate_inputs(nodes, edges, transactions)


def test_rejects_invalid_transaction_date(valid_data):
    nodes, edges, transactions = valid_data
    transactions.loc[0, "date"] = "not-a-date"

    with pytest.raises(ValueError, match="invalid or missing date"):
        validate_inputs(nodes, edges, transactions)


def test_rejects_edge_transaction_total_mismatch(valid_data):
    nodes, edges, transactions = valid_data
    edges.loc[0, "sum_kzt"] = 99

    with pytest.raises(ValueError, match="does not match transaction total"):
        validate_inputs(nodes, edges, transactions)


def test_rejects_edge_transaction_count_mismatch(valid_data):
    nodes, edges, transactions = valid_data
    edges.loc[0, "n_tx"] = 3

    with pytest.raises(ValueError, match="row count"):
        validate_inputs(nodes, edges, transactions)
