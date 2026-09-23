"""Release boundary: exact identifiers, strict schemas and complete snapshots."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd
import pytest

import pipeline


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("release")
    destination, submission = workspace / "out", workspace / "submission"
    source_before = {name: pipeline.file_hash(ROOT / "data" / name) for name in pipeline.INPUT_NAMES}
    started = time.perf_counter()
    pipeline.run(ROOT / "data", destination, submission)
    assert time.perf_counter() - started < 300
    assert source_before == {name: pipeline.file_hash(ROOT / "data" / name) for name in pipeline.INPUT_NAMES}
    return destination, submission


def test_strict_submission_and_feature_contract(completed_run):
    destination, submission = completed_run
    nodes = pd.read_parquet(ROOT / "data/nodes.parquet")
    edges = pd.read_parquet(ROOT / "data/edges.parquet")
    counts = pipeline.validate_artifacts(destination, nodes, edges)
    assert counts["nodes_roles"] == counts["node_features"] == 2248
    assert counts["top_nodes"] >= 20
    rich = pd.read_parquet(destination / "node_features.parquet")
    decisions = pd.read_csv(destination / "nodes_roles.csv", dtype={"gid": "int64"})
    assert rich.gid.dtype == "int64"
    assert set(rich.gid) == set(nodes.gid) == set(decisions.gid)
    isolated = set(nodes.gid) - set(edges.src) - set(edges.dst)
    assert len(isolated) == 19 and isolated.issubset(set(rich.gid))
    assert len(rich.columns) > 6
    assert {"in_deg", "out_deg", "in_kzt", "out_kzt", "seed_reach_count",
            "pagerank", "betweenness", "relay_2d_ratio"}.issubset(rich.columns)
    for name, columns in pipeline.SUBMISSION_COLUMNS.items():
        assert pd.read_csv(destination / name, nrows=0).columns.tolist() == columns
        assert (submission / name).read_bytes() == (destination / name).read_bytes()


def test_metadata_associates_all_artifacts_with_inputs(completed_run):
    destination, submission = completed_run
    metadata = pipeline.verify_run_metadata(ROOT / "data", destination)
    assert 0 < metadata["runtime_seconds"] < 300
    assert metadata["environment"]["python"].startswith("3.11.")
    assert metadata["environment"]["packages"]["networkx"]
    assert metadata["git"]["source"] in {"git", "environment", "unavailable"}
    assert metadata["schema_versions"]["submission"] == "1"
    assert metadata["schema_versions"]["node_features"].startswith("sha256:")
    for name in ("src/roles.py", "src/priority.py", "src/graph_features.py", "src/temporal.py"):
        assert metadata["rule_versions"][name] == "sha256:" + pipeline.file_hash(ROOT / name)
    assert metadata["inputs"]["nodes.parquet"]["rows"] == 2248
    assert metadata["inputs"]["edges.parquet"]["rows"] == 3119
    assert metadata["inputs"]["transactions.parquet"]["rows"] == 4840
    release = json.loads((submission / "release_metadata.json").read_text(encoding="utf-8"))
    assert release["run_id"] == metadata["run_id"]
    assert set(release["outputs"]) == set(pipeline.SUBMISSION_COLUMNS)
    assert release["analytics_run_metadata_sha256"] == pipeline.file_hash(destination / "run_metadata.json")
    for name, record in release["outputs"].items():
        assert record["sha256"] == pipeline.file_hash(submission / name)


def test_native_decision_reasons_and_observation_flags(completed_run):
    destination, _ = completed_run
    rich = pd.read_parquet(destination / "node_features.parquet").set_index("gid")
    top = pd.read_csv(destination / "top_nodes.csv", dtype={"gid": "int64"}).set_index("gid")
    ranked = rich.reindex(top.index)
    assert top.why.tolist() == ranked.priority_explanation.tolist()
    assert top.why.ne(ranked.evidence).all()
    assert rich.role_rule.str.strip().str.len().gt(0).all()
    assert rich.role_rule_details.str.strip().str.len().gt(0).all()
    censored = rich.is_seed | rich.depth.eq(4)
    assert rich.loc[censored, "relay_2d_ratio"].isna().all()
    assert not rich.loc[censored, "relay_2d_valid"].any()
    assert rich.loc[censored, "relay_2d_invalid_reason"].str.len().gt(0).all()
    assert rich.loc[~rich.relay_2d_valid, "relay_2d_ratio"].isna().all()
    valid = rich.loc[rich.relay_2d_valid]
    pd.testing.assert_series_equal(
        valid.relay_2d_ratio,
        valid.relay_2d_matched_days / valid.relay_2d_eligible_days,
        check_names=False,
    )
    assert not rich.loc[censored, "role"].eq("terminal").any()


def test_standalone_raw_edge_validator_survives_optimized_python(completed_run, tmp_path):
    destination, _ = completed_run
    command = [sys.executable, "-O", str(ROOT / "validate_submission.py"),
               "--data", str(ROOT / "data"), "--out", str(destination)]
    valid = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert valid.returncode == 0, valid.stdout + valid.stderr
    copied = tmp_path / "out"
    shutil.copytree(destination, copied)
    rich = pd.read_parquet(copied / "node_features.parquet")
    gid = rich.loc[rich.internal_out_kzt.gt(0), "gid"].iloc[0]
    member = rich.gid.eq(gid)
    cluster_id = rich.loc[member, "cluster_id"].iloc[0]
    rich.loc[member, "internal_out_kzt"] += 1000
    rich.to_parquet(copied / "node_features.parquet", index=False)
    clusters = pd.read_csv(copied / "clusters.csv")
    clusters.loc[clusters.cluster_id.eq(cluster_id), "sum_kzt_internal"] += 1000
    clusters.to_csv(copied / "clusters.csv", index=False)
    # The two artifacts agree with each other, but independently supplied raw
    # edges must still invalidate their fabricated internal turnover under -O.
    invalid = subprocess.run([*command[:-1], str(copied)], cwd=ROOT,
                             capture_output=True, text=True, timeout=120)
    assert invalid.returncode != 0
    assert "disagrees with source edges" in invalid.stderr


def test_equivalent_input_permutations_preserve_integrated_decisions(completed_run, tmp_path):
    destination, _ = completed_run
    permuted_data = tmp_path / "permuted_data"
    permuted_data.mkdir()
    for name in pipeline.INPUT_NAMES:
        original = pd.read_parquet(ROOT / "data" / name)
        original.sample(frac=1, random_state=7).reset_index(drop=True).to_parquet(permuted_data / name, index=False)
    permuted_out = tmp_path / "permuted_out"
    pipeline.run(permuted_data, permuted_out)
    for name, key in (("nodes_roles.csv", "gid"), ("clusters.csv", "cluster_id"), ("top_nodes.csv", "rank")):
        first = pd.read_csv(destination / name).sort_values(key).reset_index(drop=True)
        second = pd.read_csv(permuted_out / name).sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(first, second, check_exact=False, rtol=1e-12, atol=1e-12)
    first = pd.read_parquet(destination / "node_features.parquet").set_index("gid").sort_index()
    second = pd.read_parquet(permuted_out / "node_features.parquet").set_index("gid").sort_index()
    pd.testing.assert_frame_equal(first, second, check_exact=False, rtol=1e-12, atol=1e-12)
    assert pipeline.verify_run_metadata(ROOT / "data", destination)["inputs"] != pipeline.verify_run_metadata(permuted_data, permuted_out)["inputs"]


def test_identical_regeneration_keeps_artifact_hashes_and_changes_run(completed_run, tmp_path):
    destination, _ = completed_run
    first = pipeline.verify_run_metadata(ROOT / "data", destination)
    repeated = tmp_path / "out"
    pipeline.run(ROOT / "data", repeated)
    second = pipeline.verify_run_metadata(ROOT / "data", repeated)
    assert first["run_id"] != second["run_id"]
    assert first["inputs"] == second["inputs"]
    assert first["outputs"] == second["outputs"]


@pytest.mark.parametrize("name", pipeline.OUTPUT_NAMES)
def test_modified_outputs_reject_provenance(completed_run, tmp_path, name):
    destination, _ = completed_run
    copied = tmp_path / "out"
    shutil.copytree(destination, copied)
    with (copied / name).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        pipeline.verify_run_metadata(ROOT / "data", copied)


def test_changed_source_and_incomplete_run_are_rejected(completed_run, tmp_path):
    destination, _ = completed_run
    source = tmp_path / "source"
    shutil.copytree(ROOT / "data", source)
    with (source / "nodes.parquet").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="nodes.parquet"):
        pipeline.verify_run_metadata(source, destination)
    copied = tmp_path / "out"
    shutil.copytree(destination, copied)
    (copied / "run_metadata.json").unlink()
    with pytest.raises(ValueError, match="completed run metadata"):
        pipeline.verify_run_metadata(ROOT / "data", copied)


def test_validation_failure_preserves_completed_snapshot(completed_run, tmp_path, monkeypatch):
    destination, _ = completed_run
    copied = tmp_path / "out"
    shutil.copytree(destination, copied)
    previous = pipeline.verify_run_metadata(ROOT / "data", copied)

    def fail(*_args, **_kwargs):
        raise ValueError("deliberate export validation failure")

    monkeypatch.setattr(pipeline, "validate_artifacts", fail)
    with pytest.raises(ValueError, match="deliberate"):
        pipeline.run(ROOT / "data", copied)
    assert pipeline.verify_run_metadata(ROOT / "data", copied) == previous
    assert not (copied / ".pipeline.lock").exists()
    assert not list(copied.glob(".pipeline-stage-*"))


def test_late_source_swap_cannot_relabel_old_outputs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    shutil.copytree(ROOT / "data", source)
    original_hashes = {name: pipeline.file_hash(source / name) for name in pipeline.INPUT_NAMES}
    original_environment = pipeline._environment

    def swap_after_final_source_check():
        # Simulate replacement after the pipeline's last source consistency
        # check but before it serializes run metadata and publishes artifacts.
        with (source / "nodes.parquet").open("ab") as handle:
            handle.write(b"replacement")
        return original_environment()

    monkeypatch.setattr(pipeline, "_environment", swap_after_final_source_check)
    destination = tmp_path / "out"
    pipeline.run(source, destination)
    metadata = json.loads((destination / "run_metadata.json").read_text(encoding="utf-8"))
    assert {name: entry["sha256"] for name, entry in metadata["inputs"].items()} == original_hashes
    with pytest.raises(ValueError, match="nodes.parquet"):
        pipeline.verify_run_metadata(source, destination)


def test_validation_detects_regeneration_between_reads(completed_run, tmp_path, monkeypatch):
    destination, _ = completed_run
    copied = tmp_path / "out"
    shutil.copytree(destination, copied)
    original_validate = pipeline.validate_artifacts

    def regenerate_during_validation(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        marker = copied / "run_metadata.json"
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        metadata["run_id"] = "a-concurrent-regeneration-with-identical-data"
        marker.write_text(json.dumps(metadata), encoding="utf-8")
        return result

    monkeypatch.setattr(pipeline, "validate_artifacts", regenerate_during_validation)
    with pytest.raises(ValueError, match="changed during validation"):
        pipeline.validate_completed_run(ROOT / "data", copied)


def test_publication_failure_cannot_leave_valid_completion_marker(completed_run, tmp_path, monkeypatch):
    destination, _ = completed_run
    copied = tmp_path / "out"
    stage = tmp_path / "stage"
    shutil.copytree(destination, copied)
    shutil.copytree(destination, stage)
    original = pipeline.os.replace

    def fail_on_second(source, target):
        if Path(source).name == "clusters.csv":
            raise OSError("simulated interrupted publication")
        return original(source, target)

    monkeypatch.setattr(pipeline.os, "replace", fail_on_second)
    with pytest.raises(OSError, match="interrupted"):
        pipeline._publish(stage, copied, pipeline.OUTPUT_NAMES, "run_metadata.json")
    assert not (copied / "run_metadata.json").exists()


def test_parallel_writer_and_source_overwrite_are_rejected(tmp_path):
    target = tmp_path / "out"
    with pipeline._output_lock(target):
        with pytest.raises(RuntimeError, match="locked"):
            pipeline.run(ROOT / "data", target)
    with pytest.raises(ValueError, match="non-overlapping"):
        pipeline.run(ROOT / "data", ROOT / "data" / "out")


def test_large_integer_identifiers_never_pass_through_float(tmp_path):
    path = tmp_path / "ids.csv"
    path.write_text("gid\n9007199254740993\n9223372036854775807\n-9223372036854775808\n", encoding="utf-8")
    frame = pipeline._read_strict_csv(path, ["gid"], ("gid",))
    assert frame.gid.tolist() == [2**53 + 1, 2**63 - 1, -(2**63)]
    assert frame.gid.dtype == "int64"


@pytest.mark.parametrize("value", ["9007199254740993.0", "9e15", "True", "", "9223372036854775808"])
def test_malformed_identifiers_rejected(tmp_path, value):
    path = tmp_path / "ids.csv"
    path.write_text(f'gid\n"{value}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="int64"):
        pipeline._read_strict_csv(path, ["gid"], ("gid",))


@pytest.mark.parametrize("damage", ["extra_column", "cluster_count", "top_score", "feature_role"])
def test_strict_validator_catches_cross_file_inconsistency(completed_run, tmp_path, damage):
    destination, _ = completed_run
    copied = tmp_path / "out"
    shutil.copytree(destination, copied)
    if damage == "extra_column":
        frame = pd.read_csv(copied / "nodes_roles.csv")
        frame["extra"] = 0
        frame.to_csv(copied / "nodes_roles.csv", index=False)
    elif damage == "cluster_count":
        frame = pd.read_csv(copied / "clusters.csv")
        frame.loc[0, "n_seed"] += 1
        frame.to_csv(copied / "clusters.csv", index=False)
    elif damage == "top_score":
        frame = pd.read_csv(copied / "top_nodes.csv")
        frame.loc[0, "priority_score"] = 1.0
        frame.to_csv(copied / "top_nodes.csv", index=False)
    else:
        frame = pd.read_parquet(copied / "node_features.parquet")
        frame.loc[0, "role"] = "coordinator" if frame.loc[0, "role"] != "coordinator" else "transit"
        frame.to_parquet(copied / "node_features.parquet", index=False)
    with pytest.raises(ValueError):
        pipeline.validate_artifacts(copied, pd.read_parquet(ROOT / "data/nodes.parquet"),
                                    pd.read_parquet(ROOT / "data/edges.parquet"))
