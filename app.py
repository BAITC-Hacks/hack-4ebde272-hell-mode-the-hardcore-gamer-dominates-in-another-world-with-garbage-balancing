"""Money Graph — Streamlit investigation workspace.

The app is deliberately a *consumer* of the analytics exports.  It does not
assign roles, recalibrate scores, or cluster accounts; those decisions remain in
the pipeline owned by the analytics team.
"""

from __future__ import annotations

import html
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
ROLE_COLORS = {
    "coordinator": "#7c3aed", "consolidator": "#dc2626", "transit": "#0284c7",
    "distributor": "#ea580c", "terminal": "#16a34a", "peripheral": "#64748b",
}
MISSING = "Not exported"


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
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner="Loading investigation exports…")
def load_data(output_location: str, source_location: str) -> InvestigationData:
    out_dir, data_dir = resolve_path(output_location), resolve_path(source_location)
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

    return InvestigationData(roles, clusters, top_nodes, parquet("edges"), parquet("nodes"),
                             parquet("transactions"), resilience, out_dir, data_dir)


def merged_nodes(data: InvestigationData) -> pd.DataFrame:
    """Use exported role rows as authority; enrich only with source-node fields."""
    roles = data.roles.copy()
    if roles.empty and not data.nodes.empty:
        roles = data.nodes.copy()
    if roles.empty:
        return roles
    gid_col = first_column(roles, "gid")
    if gid_col != "gid" and gid_col:
        roles = roles.rename(columns={gid_col: "gid"})
    roles["gid"] = pd.to_numeric(roles["gid"], errors="coerce")
    roles = roles.dropna(subset=["gid"])
    roles["gid"] = roles["gid"].astype("int64")
    if not data.nodes.empty and "gid" in data.nodes:
        source = data.nodes.copy()
        source["gid"] = pd.to_numeric(source["gid"], errors="coerce")
        source = source.dropna(subset=["gid"])
        source["gid"] = source["gid"].astype("int64")
        missing = [c for c in source.columns if c != "gid" and c not in roles.columns]
        roles = roles.merge(source[["gid", *missing]], on="gid", how="left")
    # These are direct, observed edge aggregates for presentation only.  They do
    # not alter the analytics-owned role, cluster, PageRank, anomaly or priority
    # outputs, and are only added when an export did not already supply them.
    if not data.edges.empty and {"src", "dst"}.issubset(data.edges.columns):
        edges = data.edges.copy()
        for endpoint in ("src", "dst"):
            edges[endpoint] = pd.to_numeric(edges[endpoint], errors="coerce")
        edges = edges.dropna(subset=["src", "dst"])
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
    if first_column(roles, "turnover_kzt", "turnover", "in_out_kzt") is None and {"in_kzt", "out_kzt"}.issubset(roles.columns):
        roles["turnover_kzt"] = pd.to_numeric(roles["in_kzt"], errors="coerce").fillna(0) + pd.to_numeric(roles["out_kzt"], errors="coerce").fillna(0)
    if "truncated_by_depth" not in roles and {"depth", "out_deg"}.issubset(roles.columns):
        roles["truncated_by_depth"] = (pd.to_numeric(roles["depth"], errors="coerce") == 4) & (roles["out_deg"] == 0)
    return roles


def priority_table(data: InvestigationData, nodes: pd.DataFrame) -> pd.DataFrame:
    """Build a presentation table without altering the exported ranking."""
    table = data.top_nodes.copy() if not data.top_nodes.empty else nodes.copy()
    if table.empty:
        return table
    gid_col = first_column(table, "gid")
    if gid_col != "gid" and gid_col:
        table = table.rename(columns={gid_col: "gid"})
    table["gid"] = pd.to_numeric(table["gid"], errors="coerce")
    table = table.dropna(subset=["gid"])
    table["gid"] = table["gid"].astype("int64")
    enrich = [c for c in nodes.columns if c != "gid" and c not in table.columns]
    if enrich:
        table = table.merge(nodes[["gid", *enrich]], on="gid", how="left")
    score = first_column(table, "priority_score", "priority")
    if score is None:
        table["priority_score"] = 0.0
        score = "priority_score"
    table[score] = pd.to_numeric(table[score], errors="coerce").fillna(0.0)
    if score != "priority_score":
        table["priority_score"] = table[score]
    table = table.sort_values("priority_score", ascending=False, kind="stable").reset_index(drop=True)
    if "rank" not in table:
        table.insert(0, "rank", range(1, len(table) + 1))
    return table


def column_or_default(frame: pd.DataFrame, candidates: Iterable[str], default: object = MISSING) -> pd.Series:
    col = first_column(frame, *candidates)
    return frame[col] if col else pd.Series([default] * len(frame), index=frame.index)


def queue_view(table: pd.DataFrame) -> pd.DataFrame:
    queue = pd.DataFrame({
        "rank": column_or_default(table, ["rank"]),
        "gid": column_or_default(table, ["gid"]),
        "role": column_or_default(table, ["role"]),
        "priority_score": column_or_default(table, ["priority_score", "priority"]),
        "cluster": column_or_default(table, ["cluster_id", "cluster"]),
        "seed_reach": column_or_default(table, ["seed_reach", "n_seed_reach", "seed_reach_count"]),
        "turnover_kzt": column_or_default(table, ["turnover_kzt", "turnover", "in_out_kzt", "in_kzt"]),
        "why": column_or_default(table, ["why", "evidence", "priority_explanation"]),
        "depth": column_or_default(table, ["depth"]),
        "is_seed": column_or_default(table, ["is_seed", "seed"]),
    })
    queue["priority_score"] = pd.to_numeric(queue["priority_score"], errors="coerce").fillna(0.0)
    return queue.sort_values("priority_score", ascending=False, kind="stable")


def nav_to_node(gid: int) -> None:
    st.session_state["selected_gid"] = int(gid)
    st.session_state["page"] = "Node card"


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
            st.bar_chart(bins.value_counts(sort=False), height=180)


def overview_page(data: InvestigationData, nodes: pd.DataFrame) -> None:
    st.title("Money Graph · Analyst workspace")
    st.caption("Triage sampled payment flows, document evidence, and request the next data slice.")
    metric_cards(data, nodes)
    st.warning("Scope note: this is a four-hop, outgoing expansion from 81 seeds. Scores and roles identify candidates for review; they are not findings of wrongdoing.")
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
    minimum = e.number_input("Minimum priority", min_value=0.0, max_value=float(queue["priority_score"].max() or 1), value=0.0)
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
    st.dataframe(display[["rank", "gid", "role", "priority_score", "cluster", "seed_reach", "turnover_kzt", "why"]],
                 use_container_width=True, hide_index=True, height=440)
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


def node_card(data: InvestigationData, nodes: pd.DataFrame) -> None:
    st.title("Node card")
    gids = sorted(nodes["gid"].astype(int).unique().tolist()) if not nodes.empty else []
    current = st.session_state.get("selected_gid", gids[0] if gids else None)
    entered = st.text_input("Search arbitrary gid", value=str(current) if current is not None else "", placeholder="e.g. 123456")
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
    boundary = depth == 4 or bool(value(row, "truncated_by_depth", default=False))
    st.subheader(f"gid {gid:,} · {role}")
    top = st.columns(6)
    top[0].metric("Role confidence", fmt_metric(value(row, "role_score", "role_confidence")))
    top[1].metric("Priority", fmt_metric(value(row, "priority_score", "priority")))
    top[2].metric("Cluster", fmt_metric(cluster))
    top[3].metric("Seed", "Yes" if is_seed else "No")
    top[4].metric("Boundary", "Hop-4" if boundary else "Within sample")
    top[5].metric("Depth", fmt_metric(depth, 0))
    if depth == 4:
        st.warning("Outgoing transfers beyond hop 4 are not present in the supplied sample. Do not interpret out_deg=0 as confirmed retention.")
    st.subheader("Flow and graph evidence")
    fields = [
        ("In / out degree", value(row, "in_deg"), value(row, "out_deg"), "count"),
        ("In / out turnover", value(row, "in_kzt"), value(row, "out_kzt"), "kzt"),
        ("In / out transactions", value(row, "in_tx"), value(row, "out_tx"), "count"),
        ("Seed reach", value(row, "seed_reach", "n_seed_reach", "seed_reach_count"), None, "count"),
        ("PageRank percentile", value(row, "pagerank_percentile"), None, "raw"),
        ("Betweenness percentile", value(row, "betweenness_percentile"), None, "raw"),
        ("Temporal relay", value(row, "temporal_relay", "temporal_relay_score"), None, "raw"),
        ("Cross-cluster degree", value(row, "cross_cluster_degree"), None, "count"),
        ("Anomaly score", value(row, "anomaly_score"), None, "raw"),
    ]
    columns = st.columns(3)
    for index, (label, left, right, kind) in enumerate(fields):
        if label == "PageRank percentile" and left is None:
            left = percentile(nodes, "pagerank", gid) if "pagerank" in nodes else MISSING
        if label == "Betweenness percentile" and left is None:
            left = percentile(nodes, "betweenness", gid) if "betweenness" in nodes else MISSING
        shown = f"{fmt_kzt(left)} / {fmt_kzt(right)}" if kind == "kzt" and right is not None else \
            f"{fmt_metric(left, 0)} / {fmt_metric(right, 0)}" if right is not None else fmt_metric(left)
        columns[index % 3].metric(label, shown)
    left, right = st.columns(2)
    with left:
        st.subheader("Role evidence")
        st.write(value(row, "evidence", "role_evidence", default="No evidence text was exported."))
    with right:
        st.subheader("Investigation-priority explanation")
        detail = value(row, "priority_explanation", "priority_decomposition", "why", "evidence",
                       default="No decomposition was exported; the UI does not reconstruct the priority formula.")
        st.write(detail)
    st.subheader("Suggested next data request")
    requests = []
    if depth == 4:
        requests.append("Extend outgoing transaction history for this gid beyond hop 4, including counterparties and dates.")
    if is_seed:
        requests.append("Retrieve incoming transfers and opening balance context for this seed; seed inflows are incomplete in this sample.")
    if not requests:
        requests.append("Retrieve a longer observation window and KYC / counterparty context for the highest-value adjacent flows.")
    for request in requests:
        st.markdown(f"- {request}")
    render_counterparties(data, gid)


def render_counterparties(data: InvestigationData, gid: int) -> None:
    if data.edges.empty:
        return
    st.subheader("Largest sampled counterparties")
    incoming = data.edges[data.edges["dst"] == gid].sort_values("sum_kzt", ascending=False).head(8)
    outgoing = data.edges[data.edges["src"] == gid].sort_values("sum_kzt", ascending=False).head(8)
    a, b = st.columns(2)
    with a:
        st.caption("Incoming")
        st.dataframe(incoming, hide_index=True, use_container_width=True)
    with b:
        st.caption("Outgoing")
        st.dataframe(outgoing, hide_index=True, use_container_width=True)


def neighborhood(edges: pd.DataFrame, gid: int, hops: int) -> pd.DataFrame:
    """Bidirectional neighborhood for visual investigation; no score calculation."""
    frontier, seen = {gid}, {gid}
    for _ in range(hops):
        related = edges[edges["src"].isin(frontier) | edges["dst"].isin(frontier)]
        frontier = set(related["src"]) | set(related["dst"])
        frontier -= seen
        seen |= frontier
    return edges[edges["src"].isin(seen) & edges["dst"].isin(seen)].copy()


def pyvis_html(edges: pd.DataFrame, nodes: pd.DataFrame, focus_gid: int | None = None) -> str | None:
    if edges.empty:
        return None
    try:
        from pyvis.network import Network
    except ImportError:
        return None
    selected = set(edges["src"]) | set(edges["dst"])
    attributes = nodes[nodes["gid"].isin(selected)].set_index("gid", drop=False)
    graph = Network(height="650px", width="100%", directed=True, bgcolor="#ffffff", font_color="#172033")
    graph.set_options("""{"physics":{"stabilization":{"iterations":150},"barnesHut":{"gravitationalConstant":-5000}},"edges":{"smooth":false,"arrows":{"to":{"enabled":true,"scaleFactor":0.7}}}}""")
    for gid in selected:
        row = attributes.loc[gid] if gid in attributes.index else pd.Series(dtype=object)
        role = str(value(row, "role", default="unassigned")).lower()
        seed = as_bool(value(row, "is_seed", "seed", default=False))
        priority = as_number(value(row, "priority_score", "priority", default=0))
        label = str(gid) + (" ★" if seed else "")
        title = "<br>".join([f"<b>gid {html.escape(str(gid))}</b>", f"role: {html.escape(role)}",
                               f"priority: {priority:.3f}", f"seed: {seed}"])
        graph.add_node(int(gid), label=label, title=title, color={"background": ROLE_COLORS.get(role, "#94a3b8"),
                       "border": "#111827" if seed else "#ffffff"}, borderWidth=4 if seed else 1,
                       size=12 + min(24, priority * 18) + (7 if gid == focus_gid else 0))
    for edge in edges.itertuples(index=False):
        amount, count = as_number(getattr(edge, "sum_kzt", 0)), getattr(edge, "n_tx", "?")
        graph.add_edge(int(edge.src), int(edge.dst), width=max(1, min(10, math.log1p(amount) / 2)),
                       title=f"{amount:,.0f} KZT · {count} transaction(s)")
    return graph.generate_html(notebook=False)


def network_page(data: InvestigationData, nodes: pd.DataFrame) -> None:
    st.title("Network explorer")
    st.caption("Directed arrows are payment flow. Node colour is exported role; ★ marks a seed.")
    if data.edges.empty:
        st.warning("Source `edges.parquet` is required for network exploration.")
        return
    mode = st.radio("View", ["Selected gid · 1 hop", "Selected gid · 2 hops", "Selected cluster", "Full network (optional)"], horizontal=True)
    gid = st.session_state.get("selected_gid", int(nodes["gid"].iloc[0]) if not nodes.empty else None)
    graph_edges = pd.DataFrame()
    if mode.startswith("Selected gid"):
        entered = st.text_input("Selected gid", value=str(gid) if gid is not None else "")
        try:
            gid = int(entered)
        except ValueError:
            st.info("Enter a valid gid.")
            return
        graph_edges = neighborhood(data.edges, gid, 1 if "1 hop" in mode else 2)
        if len(graph_edges) > 500:
            st.info("This neighborhood is large; showing the 500 highest-turnover sampled edges.")
            graph_edges = graph_edges.nlargest(500, "sum_kzt")
    elif mode == "Selected cluster":
        cluster_col = first_column(nodes, "cluster_id", "cluster")
        if not cluster_col:
            st.warning("Cluster labels have not been exported yet.")
            return
        choices = sorted(nodes[cluster_col].dropna().astype(str).unique())
        initial = st.session_state.pop("network_cluster", choices[0] if choices else None)
        selected = st.selectbox("Cluster", choices, index=choices.index(initial) if initial in choices else 0)
        members = set(nodes.loc[nodes[cluster_col].astype(str) == selected, "gid"])
        graph_edges = data.edges[data.edges["src"].isin(members) & data.edges["dst"].isin(members)]
        gid = None
    else:
        confirm = st.checkbox("I understand the full network may be unreadable", value=False)
        if not confirm:
            st.info("Use a neighborhood or cluster for a decision-ready view.")
            return
        graph_edges, gid = data.edges, None
    if graph_edges.empty:
        st.info("No supplied edges match this selection.")
        return
    html_graph = pyvis_html(graph_edges, nodes, gid)
    if html_graph is None:
        st.error("Network view needs the optional `pyvis` package: `pip install pyvis`.")
        return
    components.html(html_graph, height=670, scrolling=True)


def cluster_page(data: InvestigationData, nodes: pd.DataFrame) -> None:
    st.title("Cluster review")
    cluster_col = first_column(nodes, "cluster_id", "cluster")
    if not cluster_col:
        st.warning("No cluster labels are available. Run the clustering stage first.")
        return
    choices = sorted(nodes[cluster_col].dropna().astype(str).unique())
    selected = st.selectbox("Cluster", choices)
    members = nodes[nodes[cluster_col].astype(str) == selected]
    summary = data.clusters[data.clusters["cluster_id"].astype(str) == selected] if "cluster_id" in data.clusters else pd.DataFrame()
    internal = data.edges[data.edges["src"].isin(members["gid"]) & data.edges["dst"].isin(members["gid"])]
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
        st.dataframe(members.sort_values(score, ascending=False).head(15) if score else members.head(15), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Role composition")
        if role_col:
            st.bar_chart(members[role_col].fillna("unassigned").astype(str).value_counts())
        else:
            st.info("Roles are not exported.")
    if st.button("Open this cluster in Network explorer"):
        st.session_state["network_cluster"] = selected
        st.session_state["page"] = "Network explorer"
        st.rerun()


def resilience_page(data: InvestigationData) -> None:
    st.title("Resilience · structural concentration")
    st.caption("This is a structural sensitivity analysis, not proof that blocking accounts would destroy a criminal network.")
    if data.resilience.empty:
        st.info("No resilience dataframe found. Place `resilience.csv` (or `resilience_metrics.csv`) in `out/` to enable this page.")
        st.code("scenario,largest_component_size,n_components,fraction_remaining\nbaseline,...\nremove_top_1,...")
        return
    frame = data.resilience.copy()
    scenario = first_column(frame, "scenario", "removal", "step")
    largest = first_column(frame, "largest_component_size", "largest_component")
    components_col = first_column(frame, "n_components", "number_of_components", "components")
    fraction = first_column(frame, "fraction_remaining", "remaining_fraction")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if scenario and largest:
        st.subheader("Largest component after ranked removals")
        st.bar_chart(frame.set_index(scenario)[largest])
    if scenario and components_col:
        st.subheader("Number of components")
        st.bar_chart(frame.set_index(scenario)[components_col])
    if scenario and fraction:
        st.subheader("Fraction remaining")
        st.bar_chart(frame.set_index(scenario)[fraction])


def ai_page(data: InvestigationData, nodes: pd.DataFrame) -> None:
    st.title("Optional AI analyst")
    st.caption("Available only with `OPENAI_API_KEY`. It can explain deterministic tool results; it cannot create graph facts or make guilt claims.")
    if not os.getenv("OPENAI_API_KEY"):
        st.info("Core investigation features work without an API key. Set `OPENAI_API_KEY` and install `openai` to enable this optional panel.")
        return
    question = st.text_area("Ask about exported evidence", placeholder="Compare gid 101 and gid 202, then explain what to review next.")
    if st.button("Ask grounded assistant", type="primary", disabled=not question.strip()):
        try:
            from src.ai_assistant import GraphInvestigationTools, answer_question
            tools = GraphInvestigationTools(nodes, data.top_nodes, data.clusters, data.edges)
            with st.spinner("Calling deterministic graph tools…"):
                st.markdown(answer_question(question, tools))
        except Exception as exc:
            st.error(f"AI analyst unavailable: {exc}")


def main() -> None:
    st.set_page_config(page_title="Money Graph", page_icon="◌", layout="wide")
    st.sidebar.title("Money Graph")
    output_location = st.sidebar.text_input("Exports directory", "out")
    source_location = st.sidebar.text_input("Source data directory", "data")
    if st.sidebar.button("Reload data"):
        load_data.clear()
    pages = ["Overview", "Investigation queue", "Node card", "Network explorer", "Cluster review", "Resilience", "AI analyst"]
    current_page = st.session_state.get("page", "Overview")
    page = st.sidebar.radio("Workspace", pages, index=pages.index(current_page) if current_page in pages else 0)
    st.session_state["page"] = page
    try:
        data = load_data(output_location, source_location)
    except Exception as exc:
        st.error(f"Could not load the supplied files: {exc}")
        st.stop()
    nodes = merged_nodes(data)
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
