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


IDENTIFIER_COLUMNS = [(0, "gid"), (1, "src"), (1, "dst"), (2, "src"), (2, "dst")]


@pytest.mark.parametrize("frame_number,column", IDENTIFIER_COLUMNS)
@pytest.mark.parametrize("bad", [True, False, 1.5, float("nan"), float("inf"), None, pd.NA,
                                  "1.5", "1e0", "garbage", "", 2**63, -(2**63)-1,
                                  str(2**63), str(-(2**63)-1), float(2**53), -float(2**53)])
def test_rejects_invalid_identifier_in_every_column(valid_data, frame_number, column, bad):
    frames = list(valid_data)
    frames[frame_number][column] = frames[frame_number][column].astype(object)
    frames[frame_number].loc[0, column] = bad
    label = ("nodes", "edges", "transactions")[frame_number]
    with pytest.raises(ValueError, match=rf"{label}\.{column} has invalid integer"):
        validate_inputs(*frames)


@pytest.mark.parametrize("as_text", [False, True])
def test_preserves_int64_extremes_and_large_identifiers_exactly(valid_data, as_text):
    frames = list(valid_data)
    mapping = {1: -(2**63), 2: 2**63-1, 3: 2**53+1}
    for frame_number, column in IDENTIFIER_COLUMNS:
        frames[frame_number][column] = frames[frame_number][column].map(mapping)
        if as_text:
            frames[frame_number][column] = frames[frame_number][column].map(str)
    expected = {(number, column): [int(value) for value in frames[number][column]]
                for number, column in IDENTIFIER_COLUMNS}
    normalized = validate_inputs(*frames)
    for frame_number, column in IDENTIFIER_COLUMNS:
        assert str(normalized[frame_number][column].dtype) == "int64"
        assert normalized[frame_number][column].tolist() == expected[(frame_number, column)]
    assert normalized[0]["gid"].tolist() == [-(2**63), 2**63-1, 2**53+1]
    assert normalized[1]["src"].tolist() == [-(2**63), 2**63-1]
    assert normalized[2]["dst"].tolist() == [2**63-1, 2**63-1, -(2**63)]


def test_accepts_safe_integral_float_and_integer_text_identifiers(valid_data):
    nodes, edges, transactions = valid_data
    nodes["gid"] = ["+01", " 2 ", "3"]
    edges["src"] = edges["src"].astype(float)
    transactions["dst"] = transactions["dst"].astype(float)
    normalized = validate_inputs(nodes, edges, transactions)
    assert normalized[0]["gid"].tolist() == [1, 2, 3]
    assert all(str(normalized[number][column].dtype) == "int64"
               for number, column in IDENTIFIER_COLUMNS)


def test_checks_gid_uniqueness_after_normalization(valid_data):
    nodes, edges, transactions = valid_data
    nodes["gid"] = [1, "01", 3]
    with pytest.raises(ValueError, match="gid must be unique"):
        validate_inputs(nodes, edges, transactions)


@pytest.mark.parametrize("frame_number,invalid", [(0, -1), (0, 5), (0, 0.5), (0, True),
                                                    (0, None), (1, 0), (1, -1), (1, 5),
                                                    (1, 1.5), (1, True), (1, None)])
def test_validates_node_and_edge_sampling_depth(valid_data, frame_number, invalid):
    frames = list(valid_data)
    frames[frame_number]["depth"] = frames[frame_number]["depth"].astype(object)
    frames[frame_number].loc[0, "depth"] = invalid
    with pytest.raises(ValueError, match=rf"{('nodes', 'edges')[frame_number]}\.depth"):
        validate_inputs(*frames)


@pytest.mark.parametrize("frame_number,column", [(1, "src"), (1, "dst"), (2, "src"), (2, "dst")])
def test_checks_every_endpoint_membership(valid_data, frame_number, column):
    frames = list(valid_data)
    frames[frame_number].loc[0, column] = 1234
    with pytest.raises(ValueError, match=rf"{('nodes', 'edges', 'transactions')[frame_number]} reference gid"):
        validate_inputs(*frames)


@pytest.mark.parametrize("frame_number", [1, 2])
@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"), -float("inf"), True, "bad", 1+0j, pd.Timestamp("2026-01-01")])
def test_validates_edge_and_transaction_amounts(valid_data, frame_number, invalid):
    frames = list(valid_data)
    frames[frame_number]["sum_kzt"] = frames[frame_number]["sum_kzt"].astype(object)
    frames[frame_number].loc[0, "sum_kzt"] = invalid
    with pytest.raises(ValueError, match=rf"{('nodes', 'edges', 'transactions')[frame_number]}\.sum_kzt"):
        validate_inputs(*frames)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, None, float("inf"), 2**63, "2.0"])
def test_validates_transaction_counts(valid_data, invalid):
    nodes, edges, transactions = valid_data
    edges["n_tx"] = edges["n_tx"].astype(object)
    edges.loc[0, "n_tx"] = invalid
    with pytest.raises(ValueError, match=r"edges\.n_tx"):
        validate_inputs(nodes, edges, transactions)


@pytest.mark.parametrize("invalid", [None, "2026-02-30", "not-a-date", 20260701, 1.0, True])
def test_rejects_missing_invalid_and_numeric_dates(valid_data, invalid):
    nodes, edges, transactions = valid_data
    transactions["date"] = transactions["date"].astype(object)
    transactions.loc[0, "date"] = invalid
    with pytest.raises(ValueError, match=r"transactions\.date contains invalid"):
        validate_inputs(nodes, edges, transactions)


def test_normalizes_calendar_dates_in_utc(valid_data):
    nodes, edges, transactions = valid_data
    transactions["date"] = ["2026-01-02T00:30:00+02:00", "2026-01-02", "2026-01-01T15:00:00Z"]
    result = validate_inputs(nodes, edges, transactions)[2]
    assert result["date"].tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"),
                                       pd.Timestamp("2026-01-01")]


def test_rejects_duplicate_aggregated_edges(valid_data):
    nodes, edges, transactions = valid_data
    edges = pd.concat([edges, edges.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one aggregated row per directed"):
        validate_inputs(nodes, edges, transactions)


@pytest.mark.parametrize("remove_from", ["edges", "transactions"])
def test_rejects_missing_directed_pairs(valid_data, remove_from):
    nodes, edges, transactions = valid_data
    if remove_from == "edges":
        edges = edges.iloc[:1]
    else:
        transactions = transactions.iloc[:2]
    with pytest.raises(ValueError, match="different directed pairs"):
        validate_inputs(nodes, edges, transactions)


def test_aggregate_comparison_honors_floating_tolerance(valid_data):
    nodes, edges, transactions = valid_data
    edges.loc[0, "sum_kzt"] += 1e-7
    validate_inputs(nodes, edges, transactions)
    with pytest.raises(ValueError, match="does not match transaction total"):
        validate_inputs(nodes, edges, transactions, rtol=0, atol=0)


@pytest.mark.parametrize("label", ["rtol", "atol"])
@pytest.mark.parametrize("invalid", [-1, float("inf"), float("nan"), True, "0.1", 1j, 1+0j])
def test_rejects_invalid_tolerances(valid_data, label, invalid):
    with pytest.raises(ValueError, match=f"{label} must be a finite nonnegative number"):
        validate_inputs(*valid_data, **{label: invalid})


def test_aggregate_sums_are_stable_under_transaction_permutations():
    nodes = pd.DataFrame({"gid": [1, 2], "depth": [0, 1], "is_seed": [True, False]})
    edges = pd.DataFrame({"src": [1], "dst": [2], "sum_kzt": [1e16+4], "n_tx": [5], "depth": [1]})
    transactions = pd.DataFrame({"src": [1]*5, "dst": [2]*5, "date": ["2026-01-01"]*5,
                                 "sum_kzt": [1e16, 1, 1, 1, 1]})
    for random_state in range(6):
        validate_inputs(nodes, edges, transactions.sample(frac=1, random_state=random_state), rtol=0, atol=0)


def test_rejects_overflowing_transaction_aggregate():
    nodes = pd.DataFrame({"gid": [1, 2], "depth": [0, 1], "is_seed": [True, False]})
    edges = pd.DataFrame({"src": [1], "dst": [2], "sum_kzt": [1e308], "n_tx": [2], "depth": [1]})
    transactions = pd.DataFrame({"src": [1]*2, "dst": [2]*2, "date": ["2026-01-01"]*2,
                                 "sum_kzt": [1e308, 1e308]})
    with pytest.raises(ValueError, match="total overflows for directed pair"):
        validate_inputs(nodes, edges, transactions)


def test_does_not_mutate_inputs_or_row_order(valid_data):
    frames = [frame.iloc[::-1].copy() for frame in valid_data]
    frames[0]["unused"] = "kept in source only"
    before = [frame.copy(deep=True) for frame in frames]
    normalized = validate_inputs(*frames)
    for original, snapshot, result in zip(frames, before, normalized):
        pd.testing.assert_frame_equal(original, snapshot)
        assert result.index.tolist() == original.index.tolist()
    assert "unused" not in normalized[0]
    assert normalized[0]["gid"].tolist() == frames[0]["gid"].tolist()


def test_accepts_isolates_empty_transactions_and_empty_datasets(valid_data):
    nodes, edges, transactions = valid_data
    for node_input in (nodes, nodes.iloc[:0]):
        result = validate_inputs(node_input, edges.iloc[:0], transactions.iloc[:0])
        assert len(result[0]) == len(node_input)
        assert result[1].empty and result[2].empty
        assert str(result[0]["gid"].dtype) == "int64"


def test_nonempty_edges_require_transactions(valid_data):
    nodes, edges, transactions = valid_data
    with pytest.raises(ValueError, match="transactions are empty"):
        validate_inputs(nodes, edges, transactions.iloc[:0])


@pytest.mark.parametrize("frame_number", [0, 1, 2])
def test_reports_missing_required_columns(valid_data, frame_number):
    frames = list(valid_data)
    frames[frame_number] = frames[frame_number].drop(columns=frames[frame_number].columns[0])
    with pytest.raises(ValueError, match="missing required column"):
        validate_inputs(*frames)


def test_rejects_duplicate_column_names(valid_data):
    nodes, edges, transactions = valid_data
    nodes = pd.concat([nodes, nodes[["gid"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate column names"):
        validate_inputs(nodes, edges, transactions)


def test_requires_dataframe_inputs(valid_data):
    with pytest.raises(TypeError, match="must be pandas DataFrames"):
        validate_inputs(valid_data[0].to_dict(), *valid_data[1:])


def test_loader_validates_and_preserves_large_ids(tmp_path, valid_data):
    from src.loader import load, load_inputs

    frames = list(valid_data)
    mapping = {1: 2**63-1, 2: 2**53+1, 3: -10}
    for number, column in IDENTIFIER_COLUMNS:
        frames[number][column] = frames[number][column].map(mapping)
    for name, frame in zip(("nodes", "edges", "transactions"), frames):
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    loaded = load_inputs(tmp_path)
    assert loaded[0]["gid"].tolist() == [2**63-1, 2**53+1, -10]
    assert loaded[2]["date"].dt.hour.eq(0).all()
    assert load is load_inputs
    raw = load_inputs(tmp_path, validate=False)
    assert raw[2]["date"].tolist() == frames[2]["date"].tolist()


def test_loader_reports_all_missing_paths(tmp_path):
    from src.loader import load_inputs

    with pytest.raises(FileNotFoundError) as error:
        load_inputs(tmp_path)
    assert all(f"{name}.parquet" in str(error.value) for name in ("nodes", "edges", "transactions"))


@pytest.mark.parametrize("empty", [False, True])
def test_validation_is_idempotent_with_normalized_dates(valid_data, empty):
    nodes, edges, transactions = valid_data
    if empty:
        edges = edges.iloc[:0]
        transactions = transactions.iloc[:0]
    once = validate_inputs(nodes, edges, transactions)
    twice = validate_inputs(*once)
    for first, second in zip(once, twice):
        pd.testing.assert_frame_equal(first, second)
