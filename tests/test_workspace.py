"""Uploaded cases and review downloads keep exact IDs and completed-run provenance."""
from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path
import zipfile

import pandas as pd
import pytest

import pipeline
from src import workspace


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def uploaded():
    contents = {name: (ROOT / "data" / name).read_bytes() for name in workspace.INPUT_NAMES}
    case = workspace.run_uploaded_case(contents)
    yield case
    case.cleanup()


def test_uploaded_case_uses_production_pipeline_without_mutating_supplied_files(uploaded):
    assert uploaded.metadata["status"] == "complete"
    assert uploaded.metadata["inputs"]["nodes.parquet"]["rows"] == 2248
    assert uploaded.data_dir != ROOT / "data"
    assert uploaded.out_dir != ROOT / "out"
    assert pipeline.validate_completed_run(uploaded.data_dir, uploaded.out_dir)
    for name in workspace.INPUT_NAMES:
        assert (uploaded.data_dir / name).read_bytes() == (ROOT / "data" / name).read_bytes()
    assert len(pd.read_parquet(uploaded.out_dir / "node_features.parquet")) == 2248


def test_submission_download_has_exact_schemas_hashes_and_stable_bytes(uploaded):
    content = workspace.build_submission_zip(uploaded.out_dir, uploaded.data_dir)
    assert content == workspace.build_submission_zip(uploaded.out_dir, uploaded.data_dir)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert archive.namelist() == [*pipeline.SUBMISSION_COLUMNS, "release_metadata.json"]
        metadata = json.loads(archive.read("release_metadata.json"))
        assert metadata["run_id"] == uploaded.metadata["run_id"]
        assert metadata["package_type"] == "submission"
        assert set(metadata["outputs"]) == set(pipeline.SUBMISSION_COLUMNS)
        for name, columns in pipeline.SUBMISSION_COLUMNS.items():
            payload = archive.read(name)
            assert payload == (uploaded.out_dir / name).read_bytes()
            assert pd.read_csv(io.BytesIO(payload)).columns.tolist() == columns
            assert hashlib.sha256(payload).hexdigest() == metadata["outputs"][name]["sha256"]


@pytest.mark.parametrize("changed", ["run_id", "inputs", "outputs"])
def test_submission_download_rejects_a_different_displayed_snapshot(uploaded, changed):
    displayed = deepcopy(uploaded.metadata)
    if changed == "run_id":
        displayed["run_id"] = "earlier-displayed-run"
    else:
        name = next(iter(displayed[changed]))
        displayed[changed][name]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed since the displayed case"):
        workspace.build_submission_zip(uploaded.out_dir, uploaded.data_dir,
                                       expected_metadata=displayed)
    matching = workspace.build_submission_zip(uploaded.out_dir, uploaded.data_dir,
                                              expected_metadata=uploaded.metadata)
    assert matching == workspace.build_submission_zip(uploaded.out_dir, uploaded.data_dir)


def test_submission_download_rejects_a_changed_run(uploaded, monkeypatch):
    actual = pipeline.verify_run_metadata
    calls = 0

    def changed(*args):
        nonlocal calls
        calls += 1
        metadata = actual(*args)
        return metadata if calls == 1 else {**metadata, "run_id": "another-run"}

    monkeypatch.setattr(pipeline, "verify_run_metadata", changed)
    with pytest.raises(ValueError, match="Run changed"):
        workspace.build_submission_zip(uploaded.out_dir, uploaded.data_dir)


@pytest.fixture
def review_nodes():
    return pd.DataFrame({
        "gid": [2**63 - 1, 2**53 + 1, 7], "role": ["peripheral", "consolidator", "terminal"],
        "role_score": [0, .8, .7], "cluster_id": [0, 1, 1], "priority_score": [.1, .9, .9],
        "evidence": ["No observed activity", "Several senders", "Observed retention"],
        "priority_explanation": ["No observed signals", "Incoming concentration", "Graph centrality"],
        "depth": [0, 4, 2], "is_seed": [True, False, False], "in_deg": [0, 8, 1],
        "out_deg": [0, 0, 0], "relay_2d_censored_days": [0, 1, 0],
    })


def test_shortlist_copies_decisions_exact_ids_and_observation_guidance(review_nodes):
    ids = [str(2**63 - 1), str(2**53 + 1), "7", "7"]
    content = workspace.build_review_csv(review_nodes, ids, {"run_id": "case-123"})
    frame = pd.read_csv(io.BytesIO(content), dtype={"gid": "string"})
    assert frame.columns.tolist() == workspace.REVIEW_COLUMNS
    assert frame.gid.tolist() == ["7", str(2**53 + 1), str(2**63 - 1)]
    assert frame.why.tolist() == ["Graph centrality", "Incoming concentration", "No observed signals"]
    assert frame.run_id.eq("case-123").all()
    assert "Hop-4" in frame.iloc[1].limitations
    assert "beyond hop 4" in frame.iloc[1].next_request
    assert "two days past" in frame.iloc[1].next_request
    assert "Seed inflows are incomplete" in frame.iloc[2].limitations
    assert "isolated gid" in frame.iloc[2].next_request
    assert frame.limitations.str.contains("trace the same money").all()
    assert content == workspace.build_review_csv(review_nodes.iloc[::-1], ids[::-1], {"run_id": "case-123"})


@pytest.mark.parametrize("selected, message", [([1.0], "never floats"), ([True], "exact integers"),
    (["1.0"], "exact integers"), ([str(2**63)], "outside"), ([999], "absent")])
def test_shortlist_rejects_lossy_invalid_or_unknown_ids(review_nodes, selected, message):
    with pytest.raises(ValueError, match=message):
        workspace.build_review_csv(review_nodes, selected, {"run_id": "case"})


def test_shortlist_requires_unique_nodes_and_run_id(review_nodes):
    with pytest.raises(ValueError, match="run_id"):
        workspace.build_review_csv(review_nodes, [], {})
    with pytest.raises(ValueError, match="unique gids"):
        workspace.build_review_csv(pd.concat([review_nodes, review_nodes]), [], {"run_id": "case"})
    empty = workspace.build_review_csv(review_nodes, [], {"run_id": "case"})
    assert pd.read_csv(io.BytesIO(empty)).empty


@pytest.mark.parametrize("files", [{}, {"../../nodes.parquet": b"x"},
    {"nodes.parquet": b"x", "edges.parquet": b"x", "transactions.parquet": b"x", "extra": b"x"}])
def test_upload_requires_exact_fixed_input_names(files):
    with pytest.raises(ValueError, match="exactly"):
        workspace.run_uploaded_case(files)


@pytest.mark.parametrize("content, message", [(b"", "nonempty"), (b"not parquet", "metadata"),
    (b"x" * (workspace.MAX_UPLOAD_BYTES + 1), "16 MiB")])
def test_upload_reports_empty_malformed_or_oversized_files(content, message):
    files = {name: content for name in workspace.INPUT_NAMES}
    with pytest.raises(ValueError, match=message):
        workspace.run_uploaded_case(files)


def test_upload_profile_check_is_separate_from_generic_feature_validation():
    content = io.BytesIO()
    pd.DataFrame({"gid": [1], "depth": [0], "is_seed": [True]}).to_parquet(content, index=False)
    files = {name: content.getvalue() for name in workspace.INPUT_NAMES}
    with pytest.raises(ValueError, match="exactly 2248"):
        workspace.run_uploaded_case(files)


@pytest.mark.parametrize("date, amount, message", [("2026-08-01", 5000, "July 2026"),
    ("2026-06-30", 5000, "July 2026"), ("2026-07-15", 4999, "5,000 KZT")])
def test_upload_scope_cannot_misstate_dates_or_threshold(date, amount, message):
    transactions = pd.DataFrame({"date": pd.to_datetime([date]), "sum_kzt": [amount]})
    with pytest.raises(ValueError, match=message):
        workspace._check_case_scope(transactions)


def test_failed_upload_cleans_new_files_and_keeps_prior_case(uploaded, monkeypatch):
    before = uploaded.metadata["run_id"]
    created = []

    def fail(data_dir, out_dir):
        created.append(data_dir.parent)
        raise ValueError("transaction reconciliation failed")

    monkeypatch.setattr(pipeline, "run", fail)
    files = {name: (ROOT / "data" / name).read_bytes() for name in workspace.INPUT_NAMES}
    with pytest.raises(ValueError, match="reconciliation failed"):
        workspace.run_uploaded_case(files)
    assert created and not created[0].exists()
    assert uploaded.data_dir.is_dir()
    assert uploaded.metadata["run_id"] == before


def test_uploaded_case_explicit_cleanup_removes_only_owned_directory(tmp_path):
    import tempfile
    private = tempfile.TemporaryDirectory(dir=tmp_path, prefix="case-")
    case = workspace.UploadedCase(private, {})
    owned = Path(private.name)
    case.data_dir.mkdir()
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep")
    case.cleanup()
    assert not owned.exists()
    assert unrelated.read_text() == "keep"
