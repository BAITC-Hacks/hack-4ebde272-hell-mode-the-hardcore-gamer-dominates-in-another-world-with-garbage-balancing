"""Local case uploads and analyst handoff files; no independent decision rules.

Uploads run the same validated production pipeline in a private temporary
directory. Review exports copy existing decisions and add observation guidance,
without changing the three strict submission schemas or the underlying data.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path
import re
import tempfile
import zipfile

import pandas as pd
import pyarrow.parquet as pq


INPUT_NAMES = ("nodes.parquet", "edges.parquet", "transactions.parquet")
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_UPLOAD_ROWS = {"nodes.parquet": 2248, "edges.parquet": 50000, "transactions.parquet": 100000}
REVIEW_COLUMNS = ["gid", "role", "role_score", "cluster_id", "priority_score",
                  "evidence", "why", "limitations", "next_request", "run_id"]


@dataclass
class UploadedCase:
    """Own a private case directory for the lifetime of one viewer session.

    Keep this object in session state. Explicit cleanup on replacement/reset and
    TemporaryDirectory's finalizer remove the private files. Browser disconnects
    alone do not guarantee immediate cleanup; download needed results first.
    """

    _temporary: tempfile.TemporaryDirectory
    metadata: dict

    @property
    def data_dir(self) -> Path:
        return Path(self._temporary.name) / "data"

    @property
    def out_dir(self) -> Path:
        return Path(self._temporary.name) / "out"

    def cleanup(self) -> None:
        self._temporary.cleanup()


def _check_upload(name: str, content: bytes) -> None:
    if not isinstance(content, bytes) or not content:
        raise ValueError(f"{name}: upload a nonempty Parquet file")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"{name}: upload exceeds the 16 MiB per-file limit")
    try:
        parquet = pq.ParquetFile(io.BytesIO(content))
        meta = parquet.metadata
    except Exception as exc:
        raise ValueError(f"{name}: cannot read Parquet metadata: {exc}") from exc
    if meta.num_rows > MAX_UPLOAD_ROWS[name]:
        raise ValueError(f"{name}: exceeds this local viewer's {MAX_UPLOAD_ROWS[name]:,}-row limit")
    if name == "nodes.parquet" and meta.num_rows != 2248:
        raise ValueError("nodes.parquet: this supplied-case pipeline expects exactly 2248 nodes")
    size = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
    if size > MAX_UNCOMPRESSED_BYTES:
        raise ValueError(f"{name}: expanded Parquet data exceeds the 128 MiB limit")


def run_uploaded_case(files: Mapping[str, bytes]) -> UploadedCase:
    """Validate three uploaded files and return a successfully completed case.

    Mapping keys must be the three fixed input names; client filenames are never
    used as paths. Failed runs remove their private directory and cannot replace
    a previously loaded case. Uploaded bytes are retained exactly for provenance.
    This release's production pipeline targets the supplied 2,248-node profile;
    reusable feature/input functions continue to support small synthetic cases.
    """
    if set(files) != set(INPUT_NAMES):
        raise ValueError("Upload exactly nodes.parquet, edges.parquet and transactions.parquet")
    for name in INPUT_NAMES:
        _check_upload(name, files[name])
    temporary = tempfile.TemporaryDirectory(prefix="money-graph-case-")
    try:
        root = Path(temporary.name)
        data_dir, out_dir = root / "data", root / "out"
        data_dir.mkdir()
        for name in INPUT_NAMES:
            (data_dir / name).write_bytes(files[name])
        # Viewer language is explicitly scoped to the hackathon's July sample.
        # Do not accept another period or threshold and silently describe it as
        # July / >= 5,000 KZT. Generic validation remains reusable on other data.
        from src.loader import load_inputs
        _, _, transactions = load_inputs(data_dir)
        _check_case_scope(transactions)
        # Lazy import keeps pure review/download helpers independent of execution.
        from pipeline import run, verify_run_metadata
        run(data_dir, out_dir)
        return UploadedCase(temporary, verify_run_metadata(data_dir, out_dir))
    except BaseException:
        temporary.cleanup()
        raise


def _check_case_scope(transactions: pd.DataFrame) -> None:
    if not transactions.empty and not transactions["date"].between("2026-07-01", "2026-07-31").all():
        raise ValueError("This case viewer is scoped to July 2026; uploaded transaction dates must be in that month")
    if transactions["sum_kzt"].lt(5000).any():
        raise ValueError("This case viewer assumes transfers of at least 5,000 KZT; lower amounts need a revised observation contract")


def _present(row: Mapping, name: str, default=None):
    result = row.get(name, default)
    return default if result is None or pd.isna(result) else result


def _flag(row: Mapping, name: str) -> bool:
    item = _present(row, name, False)
    return item.strip().lower() in {"true", "1", "yes"} if isinstance(item, str) else bool(item)


def observation_guidance(row: Mapping) -> tuple[list[str], list[str]]:
    """Return observed-data limitations and suggested follow-up, never guilt claims."""
    limits = ["Observed intrabank sample only; full balances are unknown. "
              "Transfers below 5,000 KZT are absent. Roles are hypotheses for review."]
    requests = []
    if _present(row, "depth") == 4 or _flag(row, "boundary_censored"):
        limits.append("Hop-4 outgoing transfers are censored; zero outflow is not confirmed retention.")
        requests.append("Extend outgoing transaction history for this gid beyond hop 4, including counterparties and dates.")
    if _flag(row, "is_seed"):
        limits.append("Seed inflows are incomplete; inbound-dependent flow ratios are unavailable.")
        requests.append("Retrieve incoming transfers and opening balance context for this seed; seed inflows are incomplete in this sample.")
    if _present(row, "in_deg") == 0 and _present(row, "out_deg") == 0:
        limits.append("No transfers are observed for this isolated gid; absence is not proof of inactivity.")
        requests.append("Confirm extract completeness for this isolated gid and request a longer history before interpreting absent activity.")
    if _present(row, "relay_2d_censored_days", 0) > 0:
        limits.append("Some incoming dates have incomplete two-day follow-up windows.")
        requests.append("Extend transaction history at least two days past the observation end to assess the omitted partial-follow-up incoming dates.")
    limits.append("Date overlap does not establish intraday order or trace the same money.")
    if not requests:
        requests.append("Retrieve a longer observation window and KYC / counterparty context for the highest-value adjacent flows.")
    return limits, requests


def _exact_gid(item: object) -> int:
    if isinstance(item, bool) or not (isinstance(item, Integral) or
            isinstance(item, str) and re.fullmatch(r"-?(?:0|[1-9][0-9]*)", item)):
        raise ValueError("Review gids must be exact integers or decimal strings, never floats")
    gid = int(item)
    if not -(2**63) <= gid < 2**63:
        raise ValueError("Review gid is outside signed int64")
    return gid


def build_review_csv(nodes: pd.DataFrame, selected_gids: Sequence, metadata: dict) -> bytes:
    """Export a deterministic, exact-gid shortlist of existing decisions.

    One row per selected gid, sorted by exported priority descending then gid.
    ``why`` copies the exported priority explanation; no score is recalculated.
    The CSV is separate from the required six-column nodes_roles submission.
    ``run_id`` binds it to the active run; users should import gid as text in
    spreadsheets to prevent automatic rounding of 64-bit identifiers.
    """
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Review export requires a verified run_id")
    required = {"gid", "role", "role_score", "cluster_id", "priority_score", "evidence"}
    if not required.issubset(nodes.columns):
        raise ValueError(f"Review export is missing decision columns: {sorted(required - set(nodes.columns))}")
    frame = nodes.copy()
    frame["gid"] = pd.Series([_exact_gid(x) for x in frame.gid], index=frame.index, dtype="int64")
    if frame.gid.duplicated().any():
        raise ValueError("Review export requires unique gids")
    selected = {_exact_gid(item) for item in selected_gids}
    missing = selected - set(frame.gid)
    if missing:
        raise ValueError(f"Review selection contains gids absent from this run: {sorted(missing)[:5]}")
    frame = frame[frame.gid.isin(selected)].sort_values(["priority_score", "gid"], ascending=[False, True], kind="stable")
    records = []
    for row in frame.to_dict("records"):
        limits, requests = observation_guidance(row)
        records.append({**{name: row[name] for name in required},
            "why": _present(row, "priority_explanation", _present(row, "why", row["evidence"])),
            "limitations": " ".join(limits), "next_request": " ".join(requests), "run_id": run_id})
    result = pd.DataFrame(records, columns=REVIEW_COLUMNS)
    return result.to_csv(index=False, lineterminator="\n").encode("utf-8")


def build_submission_zip(out_dir: str | Path, data_dir: str | Path,
                         *, expected_metadata: dict | None = None) -> bytes:
    """Package strict CSVs and a release manifest from one unchanged verified run.

    Read and recheck bytes before packaging to reject concurrent publication.
    When called by the viewer, require the displayed snapshot's full metadata;
    a new run published since the page loaded must not replace its download.
    No source data, private upload files or auxiliary features enter the ZIP.
    Fixed member times and order make identical snapshots byte-reproducible.
    """
    from pipeline import SUBMISSION_COLUMNS, verify_run_metadata
    out_dir, data_dir = Path(out_dir), Path(data_dir)
    metadata = verify_run_metadata(data_dir, out_dir)
    if expected_metadata is not None and metadata != expected_metadata:
        raise ValueError("Run changed since the displayed case was loaded; reload before downloading")
    manifest_bytes = (out_dir / "run_metadata.json").read_bytes()
    if json.loads(manifest_bytes) != metadata:
        raise ValueError("Run changed while preparing download; reload before downloading")
    payload = {name: (out_dir / name).read_bytes() for name in SUBMISSION_COLUMNS}
    for name, content in payload.items():
        if hashlib.sha256(content).hexdigest() != metadata["outputs"][name]["sha256"]:
            raise ValueError(f"{name} changed while preparing download; reload before downloading")
        if pd.read_csv(io.BytesIO(content), nrows=0).columns.tolist() != SUBMISSION_COLUMNS[name]:
            raise ValueError(f"{name} does not have the strict submission schema")
    if verify_run_metadata(data_dir, out_dir) != metadata:
        raise ValueError("Run changed while preparing download; reload before downloading")
    release = {**metadata, "package_type": "submission",
        "outputs": {name: metadata["outputs"][name] for name in SUBMISSION_COLUMNS},
        "analytics_run_metadata_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    payload["release_metadata.json"] = (json.dumps(release, indent=2) + "\n").encode("utf-8")
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in payload.items():
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, content)
    return result.getvalue()
