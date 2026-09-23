"""Money Graph — Streamlit investigation workspace.

The app is deliberately a *consumer* of the analytics exports.  It does not
assign roles, recalibrate scores, or cluster accounts; those decisions remain in
the pipeline owned by the analytics team.
"""

from __future__ import annotations

import html
import hashlib
import json
import colorsys
import math
import os
import re
from numbers import Integral, Real
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ROLE_COLORS = {
    "coordinator": "#7c3aed", "consolidator": "#dc2626", "transit": "#0284c7",
    "distributor": "#ea580c", "terminal": "#16a34a", "peripheral": "#64748b",
}
MISSING = "Not exported"
UNAVAILABLE_REASONS = {
    "seed_inbound_incomplete": "seed incoming transfers are incomplete",
    "boundary_outbound_incomplete": "outgoing transfers beyond hop 4 are unobserved",
    "no_inbound_activity": "no incoming activity is present in the sample",
    "no_outgoing_activity": "no outgoing activity is present in the sample",
    "no_complete_followup_window": "no incoming date has a complete two-day follow-up window",
}
NODE_COLUMNS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
SOURCE_FILES = ("nodes.parquet", "edges.parquet", "transactions.parquet")
EXPORT_FILES = ("nodes_roles.csv", "clusters.csv", "top_nodes.csv", "node_features.parquet", "resilience.csv")


@dataclass
class InvestigationData:
    roles: pd.DataFrame
    clusters: pd.DataFrame
    top_nodes: pd.DataFrame
    edges: pd.DataFrame
    nodes: pd.DataFrame
    transactions: pd.DataFrame
    resilience: pd.DataFrame
    output_dir: Path
    data_dir: Path
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)


def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Serialize account IDs as text so browsers cannot round 64-bit integers."""
    result = frame.copy()
    for name in ("gid", "src", "dst"):
        if name in result:
            result[name] = result[name].astype("string")
    return result


def first_column(frame: pd.DataFrame, *names: str) -> str | None:
    """Return the first matching column, case-insensitively."""
    lookup = {str(c).lower(): str(c) for c in frame.columns}
    return next((lookup[name.lower()] for name in names if name.lower() in lookup), None)


def value(row: pd.Series, *names: str, default: object = None) -> object:
    col = first_column(row.to_frame().T, *names)
    if col is None:
        return default
    candidate = row.get(col, default)
    return default if pd.isna(candidate) else candidate


def as_number(item: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([item]), errors="coerce").iloc[0]
    return default if pd.isna(parsed) else float(parsed)


def as_bool(item: object) -> bool:
    """Handle boolean columns consistently when CSV loaders return strings."""
    if isinstance(item, str):
        return item.strip().lower() in {"true", "1", "yes", "y"}
    return False if item is None or pd.isna(item) else bool(item)


def fmt_kzt(item: object) -> str:
    return f"{as_number(item):,.0f} KZT"


def fmt_metric(item: object, digits: int = 3) -> str:
    if item is None or pd.isna(item):
        return MISSING
    if isinstance(item, bool):
        return "Yes" if item else "No"
    if isinstance(item, (int, float)):
        return f"{item:,.{digits}f}" if isinstance(item, float) else f"{item:,}"
    return str(item)


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else APP_DIR / path


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"gid": "string", "top_gids": "string"}) if path.exists() else pd.DataFrame()


def exact_ids(series: pd.Series, label: str) -> pd.Series:
    """Reject lossy/fractional IDs before converting, including integral floats."""
    parsed = []
    for item in series:
        if isinstance(item, bool) or pd.isna(item):
            raise ValueError(f"{label} must contain non-null int64 identifiers")
        if isinstance(item, Integral):
            number = int(item)
        elif isinstance(item, str) and re.fullmatch(r"-?(0|[1-9][0-9]*)", item):
            number = int(item)
        else:
            raise ValueError(f"{label} contains a non-integer identifier; floating-point gids are not safe")
        if not -(2**63) <= number < 2**63:
            raise ValueError(f"{label} contains an identifier outside int64")
        parsed.append(number)
    return pd.Series(parsed, index=series.index, dtype="int64", name=series.name)


def file_fingerprints(out_dir: Path, data_dir: Path) -> tuple:
    """Hash bytes, not just timestamps: same-path/same-size regeneration is fresh."""
    paths = {data_dir / name for name in SOURCE_FILES}
    paths.update(out_dir / name for name in (*EXPORT_FILES, "run_metadata.json"))
    if out_dir.is_dir():
        paths.update(path for path in out_dir.iterdir() if path.suffix in {".csv", ".parquet", ".json"})
    result = []
    for path in sorted(paths, key=str):
        digest = None
        if path.is_file():
            hasher = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
        result.append((str(path), digest))
    return tuple(result)


def verify_provenance(out_dir: Path, data_dir: Path, fingerprints: tuple) -> dict:
    """A complete manifest prevents mixing new graph context with old scores."""
    hashes = dict(fingerprints)
    if not any(hashes.get(str(out_dir / name)) for name in EXPORT_FILES):
        return {}
    metadata_path = out_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Exports have no run_metadata.json provenance. Regenerate with the production pipeline.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("status") != "complete":
        raise ValueError("Export run metadata is unsupported or incomplete. Wait for the pipeline to finish, then reload.")
    if not isinstance(metadata.get("run_id"), str) or not metadata["run_id"].strip():
        raise ValueError("Export run metadata must identify a nonempty run_id. Regenerate the pipeline outputs.")
    for section, directory, required in (("inputs", data_dir, SOURCE_FILES), ("outputs", out_dir, EXPORT_FILES)):
        entries = metadata.get(section, {})
        for name in required:
            entry = entries.get(name, {})
            expected, actual = entry.get("sha256"), hashes.get(str(directory / name))
            if not expected or actual != expected:
                raise ValueError(f"Dataset provenance mismatch: {section}/{name}. Use source data and exports from the same completed run.")
    return metadata


def load_data(output_location: str, source_location: str) -> InvestigationData:
    out_dir, data_dir = resolve_path(output_location), resolve_path(source_location)
    fingerprints = file_fingerprints(out_dir, data_dir)
    return _load_data(str(out_dir), str(data_dir), fingerprints)


@st.cache_data(show_spinner="Loading investigation exports…")
def _load_data(output_location: str, source_location: str, fingerprints: tuple) -> InvestigationData:
    out_dir, data_dir = resolve_path(output_location), resolve_path(source_location)
    metadata = verify_provenance(out_dir, data_dir, fingerprints)
    roles = read_csv_if_exists(out_dir / "nodes_roles.csv")
    clusters = read_csv_if_exists(out_dir / "clusters.csv")
    top_nodes = read_csv_if_exists(out_dir / "top_nodes.csv")
    resilience = pd.DataFrame()
    for filename in ("resilience.csv", "resilience_metrics.csv", "resilience_analysis.csv"):
        candidate = out_dir / filename
        if candidate.exists():
            resilience = read_csv_if_exists(candidate)
            break

    def parquet(name: str) -> pd.DataFrame:
        path = data_dir / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    edges, source_nodes, transactions = parquet("edges"), parquet("nodes"), parquet("transactions")
    feature_path = out_dir / "node_features.parquet"
    features = pd.read_parquet(feature_path) if feature_path.exists() else pd.DataFrame()
    if "gid" not in features and features.index.name == "gid":
        features = features.reset_index()
    if not roles.empty and list(roles.columns) != NODE_COLUMNS:
        raise ValueError("nodes_roles.csv must have exactly: " + ",".join(NODE_COLUMNS) + ". Regenerate production exports.")
    for name, frame, required in (
        ("nodes_roles.csv", roles, {"gid", "role", "priority_score"}),
        ("clusters.csv", clusters, {"cluster_id"}),
        ("top_nodes.csv", top_nodes, {"gid", "priority_score"}),
        ("nodes.parquet", source_nodes, {"gid", "depth", "is_seed"}),
        ("edges.parquet", edges, {"src", "dst", "sum_kzt", "n_tx"}),
        ("transactions.parquet", transactions, {"src", "dst", "date", "sum_kzt"}),
        ("node_features.parquet", features, {"gid"}),
    ):
        if not frame.empty and not required.issubset(frame.columns):
            raise ValueError(f"{name} is missing columns: {sorted(required - set(frame.columns))}")
        for identifier in ("gid", "src", "dst"):
            if identifier in frame:
                frame[identifier] = exact_ids(frame[identifier], f"{name}.{identifier}")
        if "gid" in frame and frame["gid"].duplicated().any():
            raise ValueError(f"{name} must contain unique, non-null gids")
    if not roles.empty:
        expected = set(roles["gid"])
        if set(features.get("gid", [])) != expected or set(source_nodes.get("gid", [])) != expected:
            raise ValueError("Node gids must match one-to-one in nodes_roles.csv, node_features.parquet and nodes.parquet")
        if not set(top_nodes.get("gid", [])).issubset(expected):
            raise ValueError("top_nodes.csv contains gids absent from nodes_roles.csv")
    known = set(source_nodes.get("gid", []))
    for name, frame in (("edges.parquet", edges), ("transactions.parquet", transactions)):
        if not frame.empty and not (set(frame["src"]) | set(frame["dst"])).issubset(known):
            raise ValueError(f"{name} refers to gids absent from nodes.parquet")
    if file_fingerprints(out_dir, data_dir) != fingerprints:
        raise ValueError("Files changed during loading. Wait for the export run to finish, then reload.")
    return InvestigationData(roles, clusters, top_nodes, edges, source_nodes,
                             transactions, resilience, out_dir, data_dir, features, metadata)


# Preserve the explicit reload action alongside automatic content invalidation.
load_data.clear = _load_data.clear


def merged_nodes(data: InvestigationData) -> pd.DataFrame:
    """Keep the six submission fields authoritative; join exact gid-keyed metrics."""
    roles = data.roles.copy()
    if roles.empty and not data.nodes.empty:
        roles = data.nodes.copy()
    if roles.empty:
        return roles
    gid_col = first_column(roles, "gid")
    if gid_col != "gid" and gid_col:
        roles = roles.rename(columns={gid_col: "gid"})
    roles["gid"] = exact_ids(roles["gid"], "nodes_roles.gid")
    if not data.features.empty:
        features = data.features.copy()
        features["gid"] = exact_ids(features["gid"], "node_features.gid")
        if features["gid"].duplicated().any() or set(features["gid"]) != set(roles["gid"]):
            raise ValueError("node_features.parquet must match submission gids one-to-one")
        aligned = features.set_index("gid").reindex(roles["gid"])
        for name in set(NODE_COLUMNS[1:]) & set(features.columns):
            left, right = roles[name].reset_index(drop=True), aligned[name].reset_index(drop=True)
            if name in {"role_score", "priority_score"}:
                agree = ((left - right).abs() <= 1e-12) | (left.isna() & right.isna())
            else:
                agree = (left.astype("string") == right.astype("string")).fillna(False)
            if not agree.all():
                raise ValueError(f"node_features.parquet conflicts with authoritative submission field {name}")
        missing = [name for name in features if name != "gid" and name not in roles]
        roles = roles.merge(features[["gid", *missing]], on="gid", how="left", validate="one_to_one")
    if not data.nodes.empty and "gid" in data.nodes:
        source = data.nodes.copy()
        source["gid"] = exact_ids(source["gid"], "nodes.gid")
        missing = [c for c in source.columns if c != "gid" and c not in roles.columns]
        roles = roles.merge(source[["gid", *missing]], on="gid", how="left", validate="one_to_one")
    # These are direct, observed edge aggregates for presentation only.  They do
    # not alter the analytics-owned role, cluster, PageRank, anomaly or priority
    # outputs, and are only added when an export did not already supply them.
    if not data.edges.empty and {"src", "dst"}.issubset(data.edges.columns):
        edges = data.edges.copy()
        for endpoint in ("src", "dst"):
            edges[endpoint] = exact_ids(edges[endpoint], f"edges.{endpoint}")
        amount = pd.to_numeric(edges["sum_kzt"], errors="coerce").fillna(0.0) if "sum_kzt" in edges else pd.Series(0.0, index=edges.index)
        tx_count = pd.to_numeric(edges["n_tx"], errors="coerce").fillna(0.0) if "n_tx" in edges else pd.Series(0.0, index=edges.index)
        incoming = pd.DataFrame({"gid": edges["dst"], "in_deg": 1, "in_kzt": amount, "in_tx": tx_count}) \
            .groupby("gid", as_index=False).agg(in_deg=("in_deg", "sum"), in_kzt=("in_kzt", "sum"), in_tx=("in_tx", "sum"))
        outgoing = pd.DataFrame({"gid": edges["src"], "out_deg": 1, "out_kzt": amount, "out_tx": tx_count}) \
            .groupby("gid", as_index=False).agg(out_deg=("out_deg", "sum"), out_kzt=("out_kzt", "sum"), out_tx=("out_tx", "sum"))
        direct = incoming.merge(outgoing, on="gid", how="outer").fillna(0)
        additions = [c for c in direct.columns if c != "gid" and c not in roles.columns]
        if additions:
            roles = roles.merge(direct[["gid", *additions]], on="gid", how="left")
            roles[additions] = roles[additions].fillna(0)
    if first_column(roles, "turnover_kzt", "turnover", "in_out_kzt") is None and {"in_kzt", "out_kzt"}.issubset(roles.columns):
        roles["turnover_kzt"] = pd.to_numeric(roles["in_kzt"], errors="coerce").fillna(0) + pd.to_numeric(roles["out_kzt"], errors="coerce").fillna(0)
    if "truncated_by_depth" not in roles and {"depth", "out_deg"}.issubset(roles.columns):
        roles["truncated_by_depth"] = (pd.to_numeric(roles["depth"], errors="coerce") == 4) & (roles["out_deg"] == 0)
    return roles


def priority_table(data: InvestigationData, nodes: pd.DataFrame) -> pd.DataFrame:
    """Build a presentation table without altering the exported ranking."""
    table = nodes.copy() if not data.roles.empty else data.top_nodes.copy()
    if table.empty:
        return table
    gid_col = first_column(table, "gid")
    if gid_col != "gid" and gid_col:
        table = table.rename(columns={gid_col: "gid"})
    table["gid"] = exact_ids(table["gid"], "priority_table.gid")
    enrich = [c for c in nodes.columns if c != "gid" and c not in table.columns]
    if enrich:
        table = table.merge(nodes[["gid", *enrich]], on="gid", how="left")
    if "why" in data.top_nodes and "gid" in data.top_nodes and "why" not in table:
        reasons = data.top_nodes[["gid", "why"]].copy()
        reasons["gid"] = exact_ids(reasons["gid"], "top_nodes.gid")
        table = table.merge(reasons, on="gid", how="left", validate="one_to_one")
    score = first_column(table, "priority_score", "priority")
    if score is None:
        table["priority_score"] = 0.0
        score = "priority_score"
    table[score] = pd.to_numeric(table[score], errors="coerce").fillna(0.0)
    if score != "priority_score":
        table["priority_score"] = table[score]
    table = table.sort_values(["priority_score", "gid"], ascending=[False, True], kind="stable").reset_index(drop=True)
    if "rank" not in table:
        table.insert(0, "rank", range(1, len(table) + 1))
    return table


def column_or_default(frame: pd.DataFrame, candidates: Iterable[str], default: object = MISSING) -> pd.Series:
    col = first_column(frame, *candidates)
    return frame[col] if col else pd.Series([default] * len(frame), index=frame.index)


def queue_view(table: pd.DataFrame) -> pd.DataFrame:
    reasons = column_or_default(table, ["priority_explanation", "why", "evidence"])
    for name in ("why", "evidence"):
        if name in table:
            reasons = reasons.fillna(table[name])
    queue = pd.DataFrame({
        "rank": column_or_default(table, ["rank"]),
        "gid": column_or_default(table, ["gid"]),
        "role": column_or_default(table, ["role"]),
        "priority_score": column_or_default(table, ["priority_score", "priority"]),
        "cluster": column_or_default(table, ["cluster_id", "cluster"]),
        "seed_reach": column_or_default(table, ["seed_reach", "n_seed_reach", "seed_reach_count"]),
        "turnover_kzt": column_or_default(table, ["turnover_kzt", "turnover", "in_out_kzt", "in_kzt"]),
        "why": reasons,
        "depth": column_or_default(table, ["depth"]),
        "is_seed": column_or_default(table, ["is_seed", "seed"]),
    })
    queue["priority_score"] = pd.to_numeric(queue["priority_score"], errors="coerce").fillna(0.0)
    return queue.sort_values("priority_score", ascending=False, kind="stable")


def nav_to_node(gid: int) -> None:
    st.session_state["pending_navigation"] = {"page": "Node card", "gid": int(gid)}


def metric_cards(data: InvestigationData, nodes: pd.DataFrame) -> None:
    roles_col = first_column(nodes, "role")
    seed_col = first_column(nodes, "is_seed", "seed")
    cluster_col = first_column(nodes, "cluster_id", "cluster")
    turnover = data.edges["sum_kzt"].sum() if "sum_kzt" in data.edges else 0
    cards = st.columns(6)
    cards[0].metric("Nodes", f"{len(data.nodes) or len(nodes):,}")
    cards[1].metric("Directed edges", f"{len(data.edges):,}")
    cards[2].metric("Transactions", f"{len(data.transactions):,}")
    cards[3].metric("Observed turnover", fmt_kzt(turnover))
    cards[4].metric("Seeds", f"{int(nodes[seed_col].map(as_bool).sum()) if seed_col else 0:,}")
    cards[5].metric("Clusters", f"{len(data.clusters) if not data.clusters.empty else nodes[cluster_col].nunique() if cluster_col else 0:,}")
    left, right = st.columns(2)
    with left:
        st.subheader("Roles")
        if roles_col:
            st.bar_chart(nodes[roles_col].fillna("unassigned").astype(str).value_counts())
        else:
            st.info("Role labels will appear after `nodes_roles.csv` is exported.")
    with right:
        st.subheader("Nodes by sampled depth")
        depth_col = first_column(nodes, "depth")
        if depth_col:
            st.bar_chart(nodes[depth_col].value_counts().sort_index())
        else:
            st.info("Depth is not present in the loaded export or source nodes file.")
    score_col = first_column(nodes, "priority_score", "priority")
    if score_col:
        st.subheader("Priority distribution")
        scores = pd.to_numeric(nodes[score_col], errors="coerce").dropna()
        if not scores.empty:
            bins = pd.cut(scores, bins=min(12, max(2, scores.nunique())))
            counts = bins.value_counts(sort=False)
            counts.index = counts.index.astype(str)
            st.bar_chart(counts, height=180)


def overview_page(data: InvestigationData, nodes: pd.DataFrame) -> None:
    st.title("Money Graph · Analyst workspace")
    st.caption("Triage sampled payment flows, document evidence, and request the next data slice.")
    metric_cards(data, nodes)
    seed_count = int(data.nodes["is_seed"].map(as_bool).sum()) if "is_seed" in data.nodes else 0
    st.warning(f"Scope note: this is a four-hop, outgoing expansion from {seed_count} observed seeds. Scores and roles identify candidates for review; they are not findings of wrongdoing.")
    st.subheader("What to do next")
    st.markdown("1. Start in **Investigation queue**.  \n2. Open a candidate card to examine directed flow and evidence.  \n3. Use **Network explorer** for a small, legible neighborhood—not a full-network hairball.")


def queue_page(data: InvestigationData, nodes: pd.DataFrame) -> None:
    st.title("Investigation queue")
    table = priority_table(data, nodes)
    if table.empty:
        st.error(f"No `nodes_roles.csv` found in `{data.output_dir}`. Run the analytics pipeline first.")
        return
    queue = queue_view(table)
    a, b, c, d, e = st.columns(5)
    roles = sorted(x for x in queue["role"].dropna().astype(str).unique() if x != MISSING)
    selected_roles = a.multiselect("Role", roles)
    clusters = sorted(queue["cluster"].dropna().astype(str).unique())
    selected_clusters = b.multiselect("Cluster", clusters)
    depths = sorted(pd.to_numeric(queue["depth"], errors="coerce").dropna().astype(int).unique().tolist())
    selected_depths = c.multiselect("Depth", depths)
    seed_filter = d.selectbox("Seed status", ["All", "Seed", "Non-seed"])
    minimum = e.number_input("Minimum priority", min_value=0.0, max_value=1.0, value=0.0)
    filtered = queue[queue["priority_score"] >= minimum]
    if selected_roles:
        filtered = filtered[filtered["role"].astype(str).isin(selected_roles)]
    if selected_clusters:
        filtered = filtered[filtered["cluster"].astype(str).isin(selected_clusters)]
    if selected_depths:
        filtered = filtered[pd.to_numeric(filtered["depth"], errors="coerce").isin(selected_depths)]
    if seed_filter != "All":
        desired = seed_filter == "Seed"
        filtered = filtered[filtered["is_seed"].map(as_bool) == desired]
    st.caption(f"{len(filtered):,} candidates shown · sorted by exported priority score")
    display = filtered.copy()
    display["turnover_kzt"] = display["turnover_kzt"].map(lambda x: fmt_kzt(x) if x != MISSING else MISSING)
    st.dataframe(display_frame(display[["rank", "gid", "role", "priority_score", "cluster", "seed_reach", "turnover_kzt", "why"]]),
                 width="stretch", hide_index=True, height=440)
    if filtered.empty:
        return
    option_map = {f"#{row.rank} · gid {row.gid} · {row.role}": int(row.gid) for row in filtered.itertuples()}
    chosen = st.selectbox("Open selected queue row", list(option_map), key="queue_selection")
    if st.button("Open node card", type="primary"):
        nav_to_node(option_map[chosen])
        st.rerun()


def percentile(frame: pd.DataFrame, col: str, gid: int) -> str:
    if col not in frame:
        return MISSING
    values = pd.to_numeric(frame[col], errors="coerce")
    if values.notna().sum() < 2:
        return MISSING
    row_index = frame.index[frame["gid"] == gid]
    if row_index.empty:
        return MISSING
    return f"{values.rank(pct=True).loc[row_index[0]] * 100:.1f}th percentile"


def observed_ratio(row: pd.Series, metric: str, flag: str, reason: str | None = None) -> str:
    """Present explicit availability; never turn an unobserved ratio into zero."""
    invalid = flag in row.index and not as_bool(value(row, flag, default=False))
    if invalid or value(row, metric) is None:
        code = value(row, reason, default="") if reason else ""
        explanation = UNAVAILABLE_REASONS.get(str(code), str(code).replace("_", " "))
        return "Unavailable" + (f": {explanation}" if explanation else "")
    return fmt_metric(value(row, metric))


def evidence_records(raw: object) -> list[dict]:
    """Decode optional evidence while keeping nested int64 gids as browser text."""
    records = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise ValueError("Evidence must be a JSON list of records")

    def browser_safe(item, key=None):
        if key in {"gid", "src", "via", "dst"}:
            return str(exact_ids(pd.Series([item], dtype="object"), f"evidence.{key}").iloc[0])
        if key == "recipient_gids":
            if not isinstance(item, list):
                raise ValueError("Evidence recipient_gids must be a list")
            return exact_ids(pd.Series(item, dtype="object"), "evidence.recipient_gids").astype(str).tolist()
        if isinstance(item, dict):
            return {name: browser_safe(part, name) for name, part in item.items()}
        if isinstance(item, list):
            return [browser_safe(part) for part in item]
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Evidence amounts must be finite")
        return item

    return browser_safe(records)


def render_evidence_records(row: pd.Series, field: str, title: str) -> None:
    st.markdown(f"**{title}**")
    raw = value(row, field)
    if raw is None:
        st.caption("No evidence artifact was exported for this observation.")
        return
    try:
        records = evidence_records(raw)
    except (ValueError, TypeError) as exc:
        st.warning(f"Cannot display {title.lower()}: {exc}")
        return
    if records:
        st.json(records, expanded=False)
    else:
        st.caption("No matching pattern was exported within the supplied sample and search scope.")


def render_role_diagnostics(row: pd.Series) -> None:
    winning_rule = value(row, "role_rule", "role_rule_explanation", "winning_rule", "rule_explanation")
    if winning_rule is not None:
        st.write(winning_rule)
    details = value(row, "role_rule_details")
    if details is not None:
        st.write(details)
    candidates = []
    for role in ROLE_COLORS:
        if f"role_{role}_score" not in row.index:
            continue
        candidates.append({"Candidate": role, "Strength": value(row, f"role_{role}_score"),
            "Structural gate passed": value(row, f"role_{role}_gate"),
            "Eligible at threshold": value(row, f"role_{role}_eligible"),
            "Available weight": value(row, f"role_{role}_available_weight"),
            "Gate comparison": value(row, f"role_{role}_gate_reason", default=MISSING)})
    if candidates:
        with st.expander("Evaluated role candidates"):
            st.dataframe(pd.DataFrame(candidates), hide_index=True, width="stretch")
            st.caption("Exported rule diagnostics; missing components are excluded from each role's available-weight denominator.")


def node_card(data: InvestigationData, nodes: pd.DataFrame) -> None:
    st.title("Node card")
    if nodes.empty:
        st.info("No nodes are loaded. Check the source and export directories.")
        return
    gids = sorted(nodes["gid"].astype(int).unique().tolist())
    current = st.session_state.get("selected_gid", gids[0] if gids else None)
    if "node_gid_input" not in st.session_state:
        st.session_state["node_gid_input"] = str(current) if current is not None else ""
    entered = st.text_input("Search arbitrary gid", key="node_gid_input", placeholder="e.g. 123456")
    try:
        gid = int(entered)
    except ValueError:
        st.info("Enter an integer gid.")
        return
    matched = nodes[nodes["gid"] == gid]
    if matched.empty:
        st.warning("That gid is not in the loaded `nodes_roles.csv`.")
        return
    st.session_state["selected_gid"] = gid
    row = matched.iloc[0]
    role = value(row, "role", default=MISSING)
    cluster = value(row, "cluster_id", "cluster", default=MISSING)
    depth = value(row, "depth", default=MISSING)
    is_seed = as_bool(value(row, "is_seed", "seed", default=False))
    boundary = depth == 4 or as_bool(value(row, "boundary_censored", "truncated_by_depth", default=False))
    st.subheader(f"gid {gid} · {role}")
    top = st.columns(6)
    top[0].metric("Role strength", fmt_metric(value(row, "role_score", "role_confidence")))
    top[1].metric("Priority", fmt_metric(value(row, "priority_score", "priority")))
    top[2].metric("Cluster", fmt_metric(cluster))
    top[3].metric("Seed", "Yes" if is_seed else "No")
    top[4].metric("Boundary", "Hop-4" if boundary else "Within sample")
    top[5].metric("Depth", fmt_metric(depth, 0))
    st.caption("Role strength is a heuristic evidence score, not a calibrated probability. Priority is a separate review-ranking score.")
    if boundary:
        st.warning("Outgoing transfers beyond hop 4 are not present in the supplied sample. Do not interpret out_deg=0 as confirmed retention.")
    if is_seed:
        st.info("Seed inflows are incomplete. Flow ratios that depend on them are unavailable.")
    if value(row, "in_deg") == 0 and value(row, "out_deg") == 0:
        st.info("Known isolated account: no transfers involving this gid are present in the supplied sample. The peripheral role is a fallback for absent observed activity.")
    st.subheader("Flow and graph evidence")
    fields = [
        ("In / out degree", value(row, "in_deg"), value(row, "out_deg"), "count"),
        ("In / out turnover", value(row, "in_kzt"), value(row, "out_kzt"), "kzt"),
        ("In / out transactions", value(row, "in_tx"), value(row, "out_tx"), "count"),
        ("Seed reach", value(row, "seed_reach", "n_seed_reach", "seed_reach_count"), None, "count"),
        ("PageRank percentile", value(row, "pagerank_percentile", "pagerank_pct"), None, "raw"),
        ("Betweenness percentile", value(row, "betweenness_percentile", "betweenness_pct"), None, "raw"),
        ("Temporal relay", value(row, "relay_2d_ratio", "temporal_relay", "temporal_relay_score"), None, "raw"),
        ("Cross-cluster degree", value(row, "cross_cluster_degree"), None, "count"),
        ("Anomaly score", value(row, "peer_anomaly_score", "anomaly_score"), None, "raw"),
    ]
    columns = st.columns(3)
    for index, (label, left, right, kind) in enumerate(fields):
        if label == "Temporal relay":
            left = observed_ratio(row, "relay_2d_ratio", "relay_2d_valid", "relay_2d_invalid_reason")
            if (boundary or is_seed) and "relay_2d_valid" not in row.index:
                left = "Unavailable: sampled flow is incomplete"
        if label == "PageRank percentile" and left is None:
            left = percentile(nodes, "pagerank", gid) if "pagerank" in nodes else MISSING
        if label == "Betweenness percentile" and left is None:
            left = percentile(nodes, "betweenness", gid) if "betweenness" in nodes else MISSING
        if label.endswith("percentile") and isinstance(left, Real):
            left = f"{left * 100:.1f}th percentile"
        shown = f"{fmt_kzt(left)} / {fmt_kzt(right)}" if kind == "kzt" and right is not None else \
            f"{fmt_metric(left, 0)} / {fmt_metric(right, 0)}" if right is not None else fmt_metric(left)
        columns[index % 3].metric(label, shown)
    left, right = st.columns(2)
    with left:
        st.subheader("Role evidence")
        st.write(value(row, "evidence", "role_evidence", default="No evidence text was exported."))
        render_role_diagnostics(row)
    with right:
        st.subheader("Investigation-priority explanation")
        detail = value(row, "priority_explanation", "priority_decomposition", "why", "evidence",
                       default="No decomposition was exported; the UI does not reconstruct the priority formula.")
        st.write(detail)
        contributions = [name for name in nodes.columns if name.startswith("priority_") and name.endswith("_contribution")]
        if contributions:
            breakdown = pd.DataFrame({
                "Signal": [name.removeprefix("priority_").removesuffix("_contribution").replace("_", " ") for name in contributions],
                "Input value": [value(row, name.removesuffix("_contribution") + "_value") for name in contributions],
                "Available": [value(row, name.removesuffix("_contribution") + "_available") for name in contributions],
                "Score contribution": [row[name] for name in contributions],
            }).sort_values("Score contribution", ascending=False, kind="stable")
            st.dataframe(breakdown, hide_index=True, width="stretch")
            st.caption("Exported contributions sum to the priority score. Unavailable terms contribute zero with fixed weights; they are not measured zeros.")
    st.subheader("Suggested next data request")
    requests = []
    if depth == 4:
        requests.append("Extend outgoing transaction history for this gid beyond hop 4, including counterparties and dates.")
    if is_seed:
        requests.append("Retrieve incoming transfers and opening balance context for this seed; seed inflows are incomplete in this sample.")
    if value(row, "in_deg") == 0 and value(row, "out_deg") == 0:
        requests.append("Confirm extract completeness for this isolated gid and request a longer history before interpreting absent activity.")
    if as_number(value(row, "relay_2d_censored_days")) > 0:
        requests.append("Extend transaction history at least two days past the observation end to assess the omitted partial-follow-up incoming dates.")
    if not requests:
        requests.append("Retrieve a longer observation window and KYC / counterparty context for the highest-value adjacent flows.")
    for request in requests:
        st.markdown(f"- {request}")
    st.caption("The sample only covers intrabank transfers of at least 5,000 KZT during July 2026. Request other banks, smaller transfers and a longer period to assess missing context.")
    render_temporal_evidence(data, row, gid)
    render_pattern_evidence(row)
    render_anomaly_evidence(row)
    render_counterparties(data, gid)


def daily_activity(transactions: pd.DataFrame, gid: int) -> pd.DataFrame:
    """Aggregate dates with the feature contract's UTC-midnight normalization."""
    selected = transactions.loc[transactions["src"].eq(gid) | transactions["dst"].eq(gid)].copy()
    selected["date"] = pd.to_datetime(selected["date"], utc=True, format="mixed").dt.tz_convert(None).dt.normalize()
    incoming = selected.loc[selected["dst"].eq(gid)].groupby("date")["sum_kzt"].sum().rename("Incoming KZT")
    outgoing = selected.loc[selected["src"].eq(gid)].groupby("date")["sum_kzt"].sum().rename("Outgoing KZT")
    return pd.concat([incoming, outgoing], axis=1, sort=False).fillna(0).sort_index()


def render_temporal_evidence(data: InvestigationData, row: pd.Series, gid: int) -> None:
    with st.expander("Date-level activity and structural evidence"):
        st.caption("Dates do not establish intraday order or prove that the same funds moved onward. The two-day relay ratio uses only incoming dates with complete follow-up through the latest supplied transaction date.")
        start, end = value(row, "temporal_observation_start"), value(row, "temporal_observation_end")
        if start is not None and end is not None:
            st.write(f"Observed date window: {pd.Timestamp(start).date().isoformat()} — {pd.Timestamp(end).date().isoformat()}")
        st.write("Two-day relay ratio: " + observed_ratio(row, "relay_2d_ratio", "relay_2d_valid", "relay_2d_invalid_reason"))
        if "relay_2d_eligible_days" in row.index:
            st.write(f"Relay evidence: {fmt_metric(value(row, 'relay_2d_matched_days'))} matched / "
                     f"{fmt_metric(value(row, 'relay_2d_eligible_days'))} eligible incoming dates; "
                     f"{fmt_metric(value(row, 'relay_2d_censored_days'))} partial-follow-up dates excluded.")
        st.write("Same-day outgoing amount overlap: " + observed_ratio(row, "same_day_flow_ratio", "same_day_flow_valid", "same_day_flow_invalid_reason"))
        st.write("Busiest-date share of observed activity: " + observed_ratio(row, "peak_day_share", "peak_day_share_valid"))
        fields = [("Active dates", "active_days"), ("Incoming-active dates", "inbound_active_days"),
                  ("Outgoing-active dates", "outbound_active_days"), ("Most distinct senders on one date", "max_in_senders_day")]
        for label, name in fields:
            st.write(f"{label}: {fmt_metric(value(row, name))}")
        for label, name in (("Date of sender maximum", "max_in_senders_date"), ("Peak activity date", "peak_activity_date")):
            item = value(row, name)
            st.write(f"{label}: {pd.Timestamp(item).date().isoformat() if item is not None else 'Unavailable'}")
        peak_amount = value(row, "peak_activity_kzt")
        st.write(f"Peak daily incoming + outgoing activity: {float(peak_amount):,.2f} KZT" if peak_amount is not None else "Peak daily activity: Unavailable")
        st.caption("Sender maxima and peak dates describe sampled activity; no universal burst or coordination threshold is implied. Incoming + outgoing amounts count self-transfers twice.")
        if data.transactions.empty:
            st.info("No dated transactions are loaded.")
            return
        daily = daily_activity(data.transactions, gid)
        if daily.empty:
            st.info("No sampled dated activity for this gid.")
            return
        st.bar_chart(daily, stack=False)


def render_pattern_evidence(row: pd.Series) -> None:
    with st.expander("Observed routes and return candidates"):
        st.write(f"Strongly connected component size: {fmt_metric(value(row, 'scc_size'))}; "
                 f"reciprocal counterparties: {fmt_metric(value(row, 'reciprocal_relationship_count'))}.")
        st.caption("A strongly connected component indicates possible directed return paths. Reciprocal date patterns are observations of transfers, not tracing of the original funds.")
        for title, prefix in (("Repeated A → B → C routes", "repeated_route"), ("Observed A → B → A returns", "temporal_return")):
            st.write(f"{title}: {fmt_metric(value(row, prefix + '_count'))} candidates; "
                     f"maximum support: {fmt_metric(value(row, prefix + '_max_support_days'))} incoming dates.")
            if as_bool(value(row, prefix + "_truncated", default=False)):
                st.warning("Search limit reached: this count and support are lower bounds within the sampled data.")
            render_evidence_records(row, prefix + "_evidence", title + " · exact records")
        st.caption("Each record contains exact string gids, incoming/outgoing ISO date pairs, support-day count and observed KZT volumes. Routes require at least two incoming dates; returns require one. Outgoing volume counts unique matched dates once. Same-day order is unknown.")
        st.caption("Searches examine at most 512 candidate pairs and 20,000 date probes per node and pattern type. Only three candidates and three earliest date pairs per candidate are exported; absent evidence does not rule out other activity.")
    with st.expander("Observed amount patterns"):
        fields = [("Transactions in repeated exact-amount groups", "outgoing_repeated_amount_tx_count"),
                  ("Share of outgoing transactions in exact repeats", "outgoing_repeated_amount_tx_share"),
                  ("Transactions in same-day similar-amount groups", "outgoing_similar_amount_tx_count"),
                  ("Share of outgoing transactions in similar groups", "outgoing_similar_amount_tx_share"),
                  ("Same-day similar-amount groups", "similar_amount_group_count")]
        for label, field in fields:
            shown = "Unavailable" if field in row.index and value(row, field) is None else fmt_metric(value(row, field), 4)
            st.write(f"{label}: {shown}")
        render_evidence_records(row, "amount_pattern_evidence", "Observed amount groups · exact records")
        st.caption("Amounts are KZT; n_tx counts transaction rows. Similar groups require at least three same-day transfers to two recipients, with amounts no more than 5% above the group's minimum. Exact and similar shares may overlap and must not be added.")
        st.caption("These descriptors do not establish intentional splitting. Transfers omitted below the extract's 5,000 KZT threshold cannot be assessed; positive observed groups do not establish completeness.")


def render_anomaly_evidence(row: pd.Series) -> None:
    components = [name for name in row.index if name.startswith("peer_anomaly_") and name != "peer_anomaly_score" and not name.endswith("_pct")]
    if not components:
        return
    with st.expander("Depth-peer anomaly details"):
        st.write(f"Comparable depth: {fmt_metric(value(row, 'depth'))}; peer group size: {fmt_metric(value(row, 'peer_group_size'))}; "
                 f"mean component score: {fmt_metric(value(row, 'peer_anomaly_score'))}.")
        frame = pd.DataFrame({"Observed signal": [name.removeprefix("peer_anomaly_").replace("_", " ") for name in components],
                              "Deviation component": [row[name] for name in components]})
        st.dataframe(frame, hide_index=True, width="stretch")
        st.caption("Each of six exported components has weight 1/6. Values measure absolute deviations from the median at the same sampled depth, so unusually low and high activity can both contribute. Amount components use log(1 + KZT).")
        st.caption("The bounded median/MAD score is not a probability. When MAD is zero the component is 0 at the median and 1 elsewhere; no threshold here establishes wrongdoing.")


def render_counterparties(data: InvestigationData, gid: int) -> None:
    if data.edges.empty:
        return
    st.subheader("Largest sampled counterparties")
    incoming = data.edges[data.edges["dst"] == gid].sort_values("sum_kzt", ascending=False).head(8)
    outgoing = data.edges[data.edges["src"] == gid].sort_values("sum_kzt", ascending=False).head(8)
    a, b = st.columns(2)
    with a:
        st.caption("Incoming")
        st.dataframe(display_frame(incoming), hide_index=True, width="stretch")
    with b:
        st.caption("Outgoing")
        st.dataframe(display_frame(outgoing), hide_index=True, width="stretch")


def neighborhood(edges: pd.DataFrame, gid: int, hops: int) -> pd.DataFrame:
    """Bidirectional neighborhood for visual investigation; no score calculation."""
    frontier, seen = {gid}, {gid}
    for _ in range(hops):
        related = edges[edges["src"].isin(frontier) | edges["dst"].isin(frontier)]
        frontier = set(related["src"]) | set(related["dst"])
        frontier -= seen
        seen |= frontier
    return edges[edges["src"].isin(seen) & edges["dst"].isin(seen)].copy()


def selected_neighborhood(edges: pd.DataFrame, nodes: pd.DataFrame, gid: int,
                          hops: int, edge_limit: int = 500) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Bound canvas edges while retaining the selected account, even if isolated."""
    if gid not in set(nodes["gid"]):
        raise ValueError("Unknown gid: this identifier is not in the supplied nodes.")
    selected_edges = neighborhood(edges, gid, hops)
    truncated = len(selected_edges) > edge_limit
    if truncated:
        selected_edges = selected_edges.nlargest(edge_limit, "sum_kzt")
    members = {gid} | set(selected_edges["src"]) | set(selected_edges["dst"])
    return selected_edges, nodes[nodes["gid"].isin(members)], truncated


def cluster_color(cluster: object) -> str:
    hue = int(hashlib.sha256(str(cluster).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    red, green, blue = colorsys.hls_to_rgb(hue, 0.48, 0.65)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def pyvis_html(edges: pd.DataFrame, nodes: pd.DataFrame, focus_gid: int | None = None,
               color_by: str = "Role", highlight_cluster: str | None = None) -> str | None:
    """Render exactly the selected node records, including disconnected members."""
    try:
        from pyvis.network import Network
    except ImportError:
        return None
    selected = set(nodes["gid"])
    attributes = nodes.set_index("gid", drop=False)
    graph = Network(height="650px", width="100%", directed=True, bgcolor="#ffffff", font_color="#172033", cdn_resources="in_line")
    graph.set_options("""{"physics":{"stabilization":{"iterations":150},"barnesHut":{"gravitationalConstant":-5000}},"edges":{"smooth":false,"arrows":{"to":{"enabled":true,"scaleFactor":0.7}}}}""")
    for gid in sorted(selected):
        row = attributes.loc[gid] if gid in attributes.index else pd.Series(dtype=object)
        role = str(value(row, "role", default="unassigned")).lower()
        cluster = value(row, "cluster_id", "cluster", default=MISSING)
        seed = as_bool(value(row, "is_seed", "seed", default=False))
        priority = as_number(value(row, "priority_score", "priority", default=0))
        label = str(gid) + (" ★" if seed else "")
        title = "<br>".join([f"<b>gid {html.escape(str(gid))}</b>", f"role: {html.escape(role)}",
                               f"cluster: {html.escape(str(cluster))}", f"priority: {priority:.3f}", f"seed: {seed}"])
        background = cluster_color(cluster) if color_by == "Cluster" else ROLE_COLORS.get(role, "#94a3b8")
        highlighted = highlight_cluster is not None and str(cluster) == highlight_cluster
        if highlight_cluster is not None and not highlighted:
            background = "#e2e8f0"
        graph.add_node(str(gid), label=label, title=title, color={"background": background,
                       "border": "#111827" if seed else "#f59e0b" if highlighted else "#ffffff"}, borderWidth=4 if seed or highlighted else 1,
                       size=12 + min(24, priority * 18) + (7 if gid == focus_gid else 0))
    for edge in edges.itertuples(index=False):
        if edge.src not in selected or edge.dst not in selected:
            raise ValueError("Selected graph edges must refer to displayed node records")
        amount, count = as_number(getattr(edge, "sum_kzt", 0)), getattr(edge, "n_tx", "?")
        graph.add_edge(str(edge.src), str(edge.dst), width=max(1, min(10, math.log1p(amount) / 2)),
                       title=f"{amount:,.0f} KZT · {count} transaction(s)")
    content = graph.generate_html(notebook=False)
    # PyVis embeds vis-network but its template still adds unused Bootstrap CDN
    # tags. No filter/select menus use Bootstrap; remove these runtime requests.
    content = re.sub(r'<script\b[^>]*\bsrc\s*=\s*["\'][^"\']+["\'][^>]*>\s*</script>', "", content, flags=re.I)
    content = re.sub(r'<link\b[^>]*\bhref\s*=\s*["\'][^"\']+["\'][^>]*>', "", content, flags=re.I)
    return content


def graph_legend(nodes: pd.DataFrame, color_by: str) -> None:
    roles = " &nbsp; ".join(f'<span style="color:{color}">●</span> {role}' for role, color in ROLE_COLORS.items())
    st.markdown("Role legend: " + roles, unsafe_allow_html=True)
    if color_by == "Cluster" and "cluster_id" in nodes:
        clusters = sorted(nodes["cluster_id"].astype(str).unique())
        legend = " &nbsp; ".join(f'<span style="color:{cluster_color(cluster)}">●</span> {html.escape(cluster)}' for cluster in clusters)
        with st.expander(f"Cluster colours ({len(clusters)})", expanded=len(clusters) <= 12):
            st.markdown(legend, unsafe_allow_html=True)
    st.caption("★ and a dark border mark seeds. Larger nodes have higher exported priority; arrows show payment direction.")


def network_page(data: InvestigationData, nodes: pd.DataFrame) -> None:
    st.title("Network explorer")
    st.caption("Start with one account or one community, then inspect sampled counterparties.")
    if nodes.empty:
        st.info("No supplied nodes are loaded.")
        return
    if not {"src", "dst"}.issubset(data.edges.columns):
        st.warning("Source `edges.parquet` is required for network exploration.")
        return
    mode = st.radio("View", ["Selected gid · 1 hop", "Selected gid · 2 hops", "Selected cluster", "Full network (optional)"], horizontal=True, key="network_view")
    gid = st.session_state.get("selected_gid", int(nodes["gid"].iloc[0]) if not nodes.empty else None)
    graph_edges = pd.DataFrame()
    graph_nodes = nodes.iloc[:0]
    if mode.startswith("Selected gid"):
        if "network_gid_input" not in st.session_state:
            st.session_state["network_gid_input"] = str(gid) if gid is not None else ""
        entered = st.text_input("Selected gid", key="network_gid_input")
        try:
            gid = int(entered)
        except ValueError:
            st.info("Enter a valid gid.")
            return
        if gid not in set(nodes["gid"]):
            st.warning("Unknown gid: this identifier is not in the supplied nodes.")
            return
        st.session_state["selected_gid"] = gid
        graph_edges, graph_nodes, truncated = selected_neighborhood(data.edges, nodes, gid, 1 if "1 hop" in mode else 2)
        if truncated:
            st.info("This neighborhood is large; showing the 500 highest-turnover sampled edges. The selected gid is retained even if its edges fall outside that limit.")
        if graph_edges.empty:
            st.info(f"Known isolated gid {gid}: no incoming or outgoing transfers are present in the supplied sample.")
    elif mode == "Selected cluster":
        cluster_col = first_column(nodes, "cluster_id", "cluster")
        if not cluster_col:
            st.warning("Cluster labels have not been exported yet.")
            return
        choices = sorted(nodes[cluster_col].dropna().astype(str).unique())
        if not choices:
            st.info("No clusters are available in the loaded data.")
            return
        if st.session_state.get("network_cluster_selection") not in choices:
            st.session_state["network_cluster_selection"] = choices[0]
        selected = st.selectbox("Cluster", choices, key="network_cluster_selection")
        members = set(nodes.loc[nodes[cluster_col].astype(str) == selected, "gid"])
        graph_nodes = nodes[nodes["gid"].isin(members)]
        graph_edges = data.edges[data.edges["src"].isin(members) & data.edges["dst"].isin(members)]
        gid = None
    else:
        confirm = st.checkbox("I understand the full network may be unreadable", value=False)
        if not confirm:
            st.info("Use a neighborhood or cluster for a decision-ready view.")
            return
        graph_edges, graph_nodes, gid = data.edges, nodes, None
    color_by = st.radio("Colour nodes by", ["Role", "Cluster"], horizontal=True)
    cluster_col = first_column(nodes, "cluster_id", "cluster")
    highlight = None
    if cluster_col:
        cluster_options = sorted(graph_nodes[cluster_col].dropna().astype(str).unique())
        chosen_highlight = st.selectbox("Highlight cluster", ["All", *cluster_options])
        highlight = None if chosen_highlight == "All" else chosen_highlight
    graph_legend(graph_nodes, color_by)
    st.caption(f"Showing {len(graph_nodes):,} nodes and {len(graph_edges):,} directed edges.")
    if graph_edges.empty and not mode.startswith("Selected gid"):
        st.info("These supplied nodes have no internal transfers in the current selection; each member is still displayed.")
    html_graph = pyvis_html(graph_edges, graph_nodes, gid, color_by, highlight)
    if html_graph is None:
        st.error("Network view needs the optional `pyvis` package: `pip install pyvis`.")
        return
    st.iframe(html_graph, height=670)


def cluster_page(data: InvestigationData, nodes: pd.DataFrame) -> None:
    st.title("Cluster review")
    cluster_col = first_column(nodes, "cluster_id", "cluster")
    if not cluster_col:
        st.warning("No cluster labels are available. Run the clustering stage first.")
        return
    choices = sorted(nodes[cluster_col].dropna().astype(str).unique())
    if not choices:
        st.info("No clusters are available in the loaded data.")
        return
    selected = st.selectbox("Cluster", choices)
    members = nodes[nodes[cluster_col].astype(str) == selected]
    summary = data.clusters[data.clusters["cluster_id"].astype(str) == selected] if "cluster_id" in data.clusters else pd.DataFrame()
    internal = data.edges[data.edges["src"].isin(members["gid"]) & data.edges["dst"].isin(members["gid"])] if not data.edges.empty else pd.DataFrame()
    seed_col, role_col = first_column(members, "is_seed", "seed"), first_column(members, "role")
    cards = st.columns(4)
    cards[0].metric("Nodes", f"{len(members):,}")
    cards[1].metric("Seeds", f"{int(members[seed_col].map(as_bool).sum()) if seed_col else 0:,}")
    cards[2].metric("Internal turnover", fmt_kzt(internal["sum_kzt"].sum() if "sum_kzt" in internal else 0))
    cards[3].metric("Internal edges", f"{len(internal):,}")
    st.subheader("Hypothesis")
    hypothesis = value(summary.iloc[0], "hypothesis", default="No hypothesis was exported for this cluster.") if not summary.empty else "No hypothesis was exported for this cluster."
    st.write(hypothesis)
    left, right = st.columns(2)
    with left:
        st.subheader("Top gids")
        score = first_column(members, "priority_score", "priority")
        st.dataframe(display_frame(members.sort_values(score, ascending=False).head(15) if score else members.head(15)), hide_index=True, width="stretch")
    with right:
        st.subheader("Role composition")
        if role_col:
            st.bar_chart(members[role_col].fillna("unassigned").astype(str).value_counts())
        else:
            st.info("Roles are not exported.")
    if st.button("Open this cluster in Network explorer"):
        st.session_state["pending_navigation"] = {"page": "Network explorer", "cluster": selected}
        st.rerun()


def resilience_page(data: InvestigationData) -> None:
    st.title("Resilience · structural concentration")
    st.caption("This is a structural sensitivity analysis, not proof that blocking accounts would destroy a criminal network.")
    if data.resilience.empty:
        st.info("No resilience dataframe found. Place `resilience.csv` (or `resilience_metrics.csv`) in `out/` to enable this page.")
        st.code("scenario,largest_component_size,n_components,fraction_remaining\nbaseline,...\nremove_top_1,...")
        return
    frame = data.resilience.copy()
    scenario = first_column(frame, "scenario", "removed_top_n", "removal", "step")
    largest = first_column(frame, "largest_component_size", "largest_weak_component_size", "largest_component")
    components_col = first_column(frame, "n_components", "n_weak_components", "number_of_components", "components")
    fraction = first_column(frame, "fraction_remaining", "fraction_baseline_largest", "remaining_fraction")
    st.dataframe(frame, width="stretch", hide_index=True)
    if scenario and largest:
        st.subheader("Largest component after ranked removals")
        st.bar_chart(frame.set_index(scenario)[largest], sort=False)
    if scenario and components_col:
        st.subheader("Number of components")
        st.bar_chart(frame.set_index(scenario)[components_col], sort=False)
    if scenario and fraction:
        st.subheader("Largest component as a fraction of its baseline size")
        st.bar_chart(frame.set_index(scenario)[fraction], sort=False)


def ai_page(data: InvestigationData, nodes: pd.DataFrame) -> None:
    st.title("Optional AI analyst")
    st.caption("Available only with `OPENAI_API_KEY`. It can explain deterministic tool results; it cannot create graph facts or make guilt claims.")
    if not os.getenv("OPENAI_API_KEY"):
        st.info("Core investigation features work without an API key. Set `OPENAI_API_KEY` to enable this optional panel.")
        return
    question = st.text_area("Ask about exported evidence", placeholder="Compare gid 101 and gid 202, then explain what to review next.")
    context = hashlib.sha256(json.dumps({
        "run_id": data.metadata.get("run_id"), "inputs": data.metadata.get("inputs"), "outputs": data.metadata.get("outputs"),
    }, sort_keys=True).encode("utf-8")).hexdigest()
    if st.button("Ask grounded assistant", type="primary", disabled=not question.strip()):
        st.session_state.pop("grounded_answer", None)
        try:
            from src.ai_assistant import GraphInvestigationTools, answer_question_result
            tools = GraphInvestigationTools(nodes, data.top_nodes, data.clusters, data.edges)
            with st.spinner("Calling deterministic graph tools…"):
                answer = answer_question_result(question, tools)
            st.session_state["grounded_answer"] = {"context": context, "question": question.strip(), "answer": answer}
        except Exception as exc:
            st.error(f"AI analyst unavailable: {exc}")
    saved = st.session_state.get("grounded_answer")
    if isinstance(saved, dict) and saved.get("context") == context:
        answer = saved["answer"]
        st.caption(f"Answer to: {saved['question']}")
        st.markdown(answer.text)
        known = set(nodes["gid"].astype(str))
        for gid in answer.node_gids:
            if gid in known and st.button(f"Open gid {gid}", key=f"ai_gid_{gid}"):
                nav_to_node(int(gid))
                st.rerun()
        if answer.sources:
            with st.expander("Supporting deterministic results"):
                st.json(list(answer.sources))


def main() -> None:
    st.set_page_config(page_title="Money Graph", page_icon="◌", layout="wide")
    st.sidebar.title("Money Graph")
    output_location = st.sidebar.text_input("Exports directory", os.getenv("MONEY_GRAPH_OUTPUT_DIR", "out"))
    source_location = st.sidebar.text_input("Source data directory", os.getenv("MONEY_GRAPH_DATA_DIR", "data"))
    if st.sidebar.button("Reload data"):
        load_data.clear()
    pages = ["Overview", "Investigation queue", "Node card", "Network explorer", "Cluster review", "Resilience", "AI analyst"]
    navigation = st.session_state.pop("pending_navigation", {})
    if navigation:
        st.session_state["page"] = navigation["page"]
        if "gid" in navigation:
            st.session_state["selected_gid"] = navigation["gid"]
            st.session_state["node_gid_input"] = str(navigation["gid"])
            st.session_state["network_gid_input"] = str(navigation["gid"])
        if "cluster" in navigation:
            st.session_state["network_view"] = "Selected cluster"
            st.session_state["network_cluster_selection"] = navigation["cluster"]
    page = st.sidebar.radio("Workspace", pages, key="page")
    try:
        data = load_data(output_location, source_location)
        nodes = merged_nodes(data)
    except Exception as exc:
        st.error(f"Could not load the supplied files: {exc}")
        st.stop()
    if data.metadata:
        st.sidebar.caption(f"Verified run: {data.metadata['run_id']}")
        st.sidebar.caption(
            f"Completed: {data.metadata.get('completed_at', 'not recorded')} · "
            f"Runtime: {data.metadata.get('runtime_seconds', 'not recorded')} s"
        )
        with st.sidebar.expander("Run provenance"):
            st.json(data.metadata)
    if page == "Overview":
        overview_page(data, nodes)
    elif page == "Investigation queue":
        queue_page(data, nodes)
    elif page == "Node card":
        node_card(data, nodes)
    elif page == "Network explorer":
        network_page(data, nodes)
    elif page == "Cluster review":
        cluster_page(data, nodes)
    elif page == "Resilience":
        resilience_page(data)
    else:
        ai_page(data, nodes)


if __name__ == "__main__":
    main()
