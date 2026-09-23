"""Local, deterministic AML decision and export pipeline."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import networkx as nx
import numpy as np
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


ROOT = Path(__file__).resolve().parent
INPUT_NAMES = ("nodes.parquet", "edges.parquet", "transactions.parquet")
SUBMISSION_COLUMNS = {
    "nodes_roles.csv": ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"],
    "clusters.csv": ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"],
    "top_nodes.csv": ["rank", "gid", "role", "priority_score", "why"],
}
OUTPUT_NAMES = (*SUBMISSION_COLUMNS, "node_features.parquet", "resilience.csv")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, rows: int) -> dict:
    return {"sha256": file_hash(path), "size_bytes": path.stat().st_size, "rows": int(rows)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_information() -> dict:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return {"revision": revision, "dirty": dirty, "source": "git"}
    except (OSError, subprocess.CalledProcessError):
        revision = os.environ.get("MONEY_GRAPH_GIT_REVISION") or None
        return {"revision": revision, "dirty": None,
                "source": "environment" if revision else "unavailable"}


def _environment() -> dict:
    versions = {}
    for name in ("pandas", "numpy", "scipy", "pyarrow", "networkx", "streamlit", "pyvis", "openai"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "packages": versions}


def _source_versions() -> dict:
    names = ("src/schema.py", "src/loader.py", "src/graph_features.py", "src/temporal.py",
             "src/clustering.py", "src/roles.py", "src/priority.py", "src/explanations.py",
             "src/exports.py", "src/resilience.py", "validate_submission.py", "pipeline.py",
             "main.py", "documentation/feature-contract.md", "documentation/data-quality.md",
             "documentation/decision-rules.md", "requirements.txt", "requirements-dev.txt",
             "constraints.txt")
    return {name: "sha256:" + file_hash(ROOT / name) for name in names if (ROOT / name).is_file()}


def _disjoint_paths(source: Path, destination: Path) -> None:
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and output directories must be separate, non-overlapping locations")


@contextmanager
def _output_lock(out_dir: Path):
    """Serialize publishers; a crashed process leaves a visible, diagnosable lock."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".pipeline.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Output is locked: {lock}. Check for an active pipeline before removing a stale lock.") from exc
    try:
        with handle:
            json.dump({"pid": os.getpid(), "started_at": _utc_now()}, handle)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _publish(stage: Path, destination: Path, names: tuple, marker: str) -> None:
    """The marker is absent throughout publication; consumers verify its hashes."""
    (destination / marker).unlink(missing_ok=True)
    for name in names:
        os.replace(stage / name, destination / name)
    os.replace(stage / marker, destination / marker)


def _read_strict_csv(path: Path, columns: list[str], integer_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.columns.tolist() != columns:
        raise ValueError(f"{path.name}: expected exact ordered columns {columns}")
    limits = np.iinfo(np.int64)
    for column in integer_columns:
        values = frame[column]
        if not values.str.fullmatch(r"-?(?:0|[1-9][0-9]*)").all():
            raise ValueError(f"{path.name}.{column}: expected exact decimal int64 identifiers/counts")
        exact = values.map(int)
        if not exact.map(lambda value: limits.min <= value <= limits.max).all():
            raise ValueError(f"{path.name}.{column}: outside int64 range")
        frame[column] = exact.astype("int64")
    return frame


def validate_artifacts(out_dir: Path, source_nodes: pd.DataFrame, edges: pd.DataFrame,
                       *, expected_nodes: int = 2248) -> dict[str, int]:
    """Strict packaging checks, independent of removable Python assertions.

    Consistency is checked here; analytical interpretation belongs to the
    decision modules. This does not certify the quality of their explanations.
    """
    tables = {
        name: _read_strict_csv(out_dir / name, columns, {
            "nodes_roles.csv": ("gid", "cluster_id"),
            "clusters.csv": ("cluster_id", "n_nodes", "n_seed"),
            "top_nodes.csv": ("rank", "gid"),
        }[name]) for name, columns in SUBMISSION_COLUMNS.items()
    }
    nodes, clusters, top = (tables[name] for name in SUBMISSION_COLUMNS)
    features = pd.read_parquet(out_dir / "node_features.parquet")
    if "gid" not in features and features.index.name == "gid":
        features = features.reset_index()
    if not pd.api.types.is_integer_dtype(source_nodes.gid.dtype) or pd.api.types.is_bool_dtype(source_nodes.gid.dtype):
        raise ValueError("Source gids must be stored as exact int64-compatible integers")
    if len(nodes) != expected_nodes or nodes.gid.duplicated().any() or set(nodes.gid) != set(source_nodes.gid):
        raise ValueError("Node output must contain every source gid exactly once")
    if ("gid" not in features or not pd.api.types.is_integer_dtype(features.gid.dtype)
            or pd.api.types.is_bool_dtype(features.gid.dtype) or features.gid.duplicated().any()
            or set(features.gid) != set(nodes.gid)):
        raise ValueError("Feature artifact must contain exactly one exact integer gid per node")
    for frame, label in ((nodes, "nodes_roles"), (top, "top_nodes")):
        for column in ("role_score", "priority_score"):
            if column in frame:
                values = pd.to_numeric(frame[column], errors="coerce")
                if not np.isfinite(values).all() or not values.between(0, 1).all():
                    raise ValueError(f"{label}.{column}: expected finite score in [0,1]")
                frame[column] = values
    from src.roles import ROLES
    if not nodes.role.isin(ROLES).all():
        raise ValueError("Invalid node role")
    for frame, field in ((nodes, "evidence"), (clusters, "hypothesis"), (clusters, "top_gids"), (top, "why")):
        if not frame[field].str.strip().str.len().gt(0).all():
            raise ValueError(f"Required {field} contains empty text")
    if not nodes.evidence.str.len().le(200).all():
        raise ValueError("Node evidence exceeds 200 characters")
    if not nodes.evidence.str.contains(r"\d").all():
        raise ValueError("Node evidence must contain numeric support")
    by_gid = nodes.set_index("gid")
    rich = features.set_index("gid").reindex(by_gid.index)
    for field in SUBMISSION_COLUMNS["nodes_roles.csv"][1:]:
        if field not in rich:
            continue
        if field in ("role_score", "priority_score"):
            matches = np.allclose(rich[field], by_gid[field], rtol=1e-12, atol=1e-12)
        else:
            matches = rich[field].equals(by_gid[field])
        if not matches:
            raise ValueError(f"Feature artifact disagrees with authoritative CSV field {field}")
    if not {"depth", "out_deg", "is_seed"}.issubset(rich):
        raise ValueError("Feature artifact lacks observation-limit flags")
    if (by_gid.role.eq("terminal") & (rich.depth.ge(4) | rich.is_seed)).any():
        raise ValueError("Seed or depth-4 node classified as terminal")
    for field in ("pass_through", "relay_2d_ratio", "same_day_flow_ratio"):
        if field in rich and rich.loc[rich.is_seed, field].notna().any():
            raise ValueError(f"Seed {field} must be unavailable for censored inflows")
    contributions = [name for name in rich if name.startswith("priority_") and name.endswith("_contribution")]
    if contributions and not np.allclose(rich[contributions].sum(axis=1), by_gid.priority_score, rtol=1e-10, atol=1e-12):
        raise ValueError("Priority contributions do not reconcile with the published score")
    if clusters.cluster_id.duplicated().any() or set(clusters.cluster_id) != set(nodes.cluster_id):
        raise ValueError("Cluster ids must match the node membership table exactly")
    seed_of = source_nodes.set_index("gid").is_seed.astype(bool)
    cluster_of = by_gid.cluster_id.to_dict()
    internal = {cid: 0.0 for cid in clusters.cluster_id}
    for edge in edges.itertuples(index=False):
        if cluster_of[edge.src] == cluster_of[edge.dst]:
            internal[cluster_of[edge.src]] += float(edge.sum_kzt)
    for row in clusters.itertuples(index=False):
        members = by_gid.index[by_gid.cluster_id.eq(row.cluster_id)]
        if row.n_nodes != len(members) or row.n_seed != int(seed_of.reindex(members).sum()):
            raise ValueError(f"Cluster {row.cluster_id} has inconsistent node/seed counts")
        try:
            amount = float(row.sum_kzt_internal)
        except ValueError as exc:
            raise ValueError("Cluster internal turnover must be numeric") from exc
        if not np.isfinite(amount) or not np.isclose(amount, internal[row.cluster_id], rtol=1e-10, atol=0.01):
            raise ValueError(f"Cluster {row.cluster_id} has inconsistent internal turnover")
        if not re.fullmatch(r"-?[0-9]+(?:,-?[0-9]+)*", row.top_gids):
            raise ValueError("Cluster top_gids must be comma-separated exact integer identifiers")
        identifiers = [int(value) for value in row.top_gids.split(",")]
        if len(identifiers) != len(set(identifiers)) or not set(identifiers).issubset(set(members)):
            raise ValueError(f"Cluster {row.cluster_id} references invalid top gids")
    if len(top) < 20 or top.gid.duplicated().any() or not set(top.gid).issubset(set(nodes.gid)):
        raise ValueError("Top list requires at least 20 unique known gids")
    if top["rank"].tolist() != list(range(1, len(top) + 1)) or not top.priority_score.is_monotonic_decreasing:
        raise ValueError("Top ranks must be contiguous and scores descending")
    expected_top = nodes.sort_values(["priority_score", "gid"], ascending=[False, True], kind="mergesort").head(len(top))
    if top.gid.tolist() != expected_top.gid.tolist():
        raise ValueError("Top gids differ from the ranked node table")
    selected = by_gid.reindex(top.gid)
    if selected.role.tolist() != top.role.tolist() or not np.allclose(selected.priority_score, top.priority_score, rtol=1e-12, atol=1e-12):
        raise ValueError("Top roles/scores disagree with the node table")
    resilience = pd.read_csv(out_dir / "resilience.csv")
    required_scenarios = {"baseline", "remove_top_1", "remove_top_5", "remove_top_10", "remove_top_20"}
    if "scenario" not in resilience or set(resilience.scenario) != required_scenarios or len(resilience) != 5:
        raise ValueError("Resilience artifact must contain baseline and four removal scenarios")
    return {"nodes_roles": len(nodes), "clusters": len(clusters), "top_nodes": len(top),
            "node_features": len(features), "resilience": len(resilience)}


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


def verify_run_metadata(data_dir: Path, out_dir: Path) -> dict:
    """Reject incomplete snapshots, changed source files and modified exports."""
    path = out_dir / "run_metadata.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("No readable completed run metadata; regenerate the pipeline outputs") from exc
    if metadata.get("schema_version") != 1 or metadata.get("status") != "complete":
        raise ValueError("Unsupported or incomplete run metadata")
    for key, directory, names in (("inputs", data_dir, INPUT_NAMES), ("outputs", out_dir, OUTPUT_NAMES)):
        entries = metadata.get(key, {})
        if set(entries) != set(names):
            raise ValueError(f"Run metadata {key} do not match the required artifacts")
        for name in names:
            artifact = directory / name
            if not artifact.is_file() or file_hash(artifact) != entries[name].get("sha256"):
                raise ValueError(f"Run metadata hash mismatch: {name}")
    if json.loads(path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Outputs changed during verification; retry after regeneration completes")
    return metadata


def validate_completed_run(data_dir: Path, out_dir: Path) -> dict[str, int]:
    """Validate one stable completed run, detecting concurrent regeneration."""
    before = verify_run_metadata(data_dir, out_dir)
    edges, nodes, _ = load_inputs(data_dir)
    counts = validate_artifacts(out_dir, nodes, edges)
    after = verify_run_metadata(data_dir, out_dir)
    if before != after:
        raise ValueError("Completed run changed during validation; retry after regeneration completes")
    return counts


def _package_submission(stage: Path, destination: Path, metadata: dict) -> None:
    with _output_lock(destination), tempfile.TemporaryDirectory(prefix=".submission-stage-", dir=destination) as temporary:
        package = Path(temporary)
        for name in SUBMISSION_COLUMNS:
            (package / name).write_bytes((stage / name).read_bytes())
        release = {**metadata, "package_type": "submission",
                   "outputs": {name: metadata["outputs"][name] for name in SUBMISSION_COLUMNS},
                   "analytics_run_metadata_sha256": file_hash(stage / "run_metadata.json")}
        (package / "release_metadata.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
        _publish(package, destination, tuple(SUBMISSION_COLUMNS), "release_metadata.json")


def _run_staged(data_dir: Path, out_dir: Path, stage: Path, submission_dir: Path | None) -> dict[str, int]:
    started = time.perf_counter()
    started_at = _utc_now()
    # Compute against an immutable private copy. Metadata always describes these
    # exact bytes, even if a user replaces source files during publication.
    input_hashes = {name: file_hash(data_dir / name) for name in INPUT_NAMES}
    snapshot = stage / ".inputs"
    snapshot.mkdir()
    for name in INPUT_NAMES:
        shutil.copyfile(data_dir / name, snapshot / name)
    snapshot_records = {name: _file_record(snapshot / name, 0) for name in INPUT_NAMES}
    if input_hashes != {name: record["sha256"] for name, record in snapshot_records.items()}:
        raise ValueError("Source files changed while making the input snapshot; no outputs were published")
    rule_versions = _source_versions()
    edges, nodes, tx = load_inputs(snapshot)
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
    write_exports(features, clusters, stage)
    # Compatibility seam for the current rich CSV exporter. The decision engine
    # remains authoritative; only serialization is split into the agreed files.
    features[SUBMISSION_COLUMNS["nodes_roles.csv"]].to_csv(stage / "nodes_roles.csv", index=False)
    detailed = features.copy()
    detailed.attrs = {}
    detailed.to_parquet(stage / "node_features.parquet", index=False)
    resilience.to_csv(stage / "resilience.csv", index=False)
    validate_submission(stage, expected_nodes=2248, source_nodes=nodes, features=features)
    counts = validate_artifacts(stage, nodes, edges)
    if input_hashes != {name: file_hash(data_dir / name) for name in INPUT_NAMES}:
        raise ValueError("Source files changed during computation; no outputs were published")
    row_counts = {"nodes.parquet": len(nodes), "edges.parquet": len(edges), "transactions.parquet": len(tx)}
    feature_schema = [(name, str(dtype)) for name, dtype in detailed.dtypes.items()]
    schema_hash = hashlib.sha256(json.dumps(feature_schema, separators=(",", ":")).encode()).hexdigest()
    command = ["python", "main.py", "--data", str(data_dir), "--out", str(out_dir)]
    if submission_dir is not None:
        command += ["--submission", str(submission_dir)]
    metadata = {
        "schema_version": 1, "status": "complete", "run_id": str(uuid.uuid4()),
        "started_at": started_at, "completed_at": _utc_now(),
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "command": command, "invocation": [sys.executable, *sys.argv],
        "inputs": {name: {**snapshot_records[name], "rows": row_counts[name]} for name in INPUT_NAMES},
        "outputs": {name: _file_record(stage / name, counts[Path(name).stem]) for name in OUTPUT_NAMES},
        "schema_versions": {"submission": "1", "node_features": "sha256:" + schema_hash},
        "rule_versions": rule_versions, "environment": _environment(), "git": _git_information(),
        "validation": "strict artifact consistency; analytical interpretation remains a separate review",
    }
    # Mark completion only after every artifact has been produced and validated.
    (stage / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if submission_dir is not None:
        _package_submission(stage, submission_dir, metadata)
    _publish(stage, out_dir, OUTPUT_NAMES, "run_metadata.json")
    print(f"output rows: nodes_roles={counts['nodes_roles']}, clusters={counts['clusters']}, top_nodes={counts['top_nodes']}, resilience={len(resilience)}")
    print(f"runtime: {time.perf_counter() - started:.2f}s")
    print(f"completed run: {metadata['run_id']}; metadata: {out_dir / 'run_metadata.json'}")
    return counts


def run(data_dir: Path, out_dir: Path, submission_dir: Path | None = None) -> dict[str, int]:
    data_dir, out_dir = Path(data_dir).resolve(), Path(out_dir).resolve()
    _disjoint_paths(data_dir, out_dir)
    if submission_dir is not None:
        submission_dir = Path(submission_dir).resolve()
        _disjoint_paths(data_dir, submission_dir)
        _disjoint_paths(out_dir, submission_dir)
    with _output_lock(out_dir), tempfile.TemporaryDirectory(prefix=".pipeline-stage-", dir=out_dir) as temporary:
        return _run_staged(data_dir, out_dir, Path(temporary), submission_dir)


def main():
    parser = argparse.ArgumentParser(description="Validate source Parquets and publish a complete, hash-associated run.")
    parser.add_argument("--data", type=Path, default=Path("./data"))
    parser.add_argument("--out", type=Path, default=Path("./out"))
    parser.add_argument("--submission", type=Path, help="Also publish the three strict CSVs and release_metadata.json here")
    parser.add_argument("--validate-only", action="store_true", help="Verify existing hashes and strict artifact consistency")
    args = parser.parse_args()
    if args.validate_only:
        if args.submission is not None:
            parser.error("--submission cannot be combined with --validate-only")
        validate_completed_run(args.data, args.out)
        print("strict artifact and provenance validation: OK")
    else:
        run(args.data, args.out, args.submission)


if __name__ == "__main__":
    main()
