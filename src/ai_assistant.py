"""Grounded, optional AI assistant for the Money Graph analyst interface.

The core UI never imports OpenAI.  This module is loaded only after a user asks a
question and an ``OPENAI_API_KEY`` is present.  The model has no data access
other than the deterministic functions in :class:`GraphInvestigationTools`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx
import pandas as pd


SYSTEM_PROMPT = """You are an AML investigation assistant for a sampled directed
payment graph. You may only make claims supported by the tool results returned in
this conversation. Always call a tool before describing a specific client,
cluster, score, path, or transaction. Cite actual gid values and numeric evidence
from tool results. Describe findings as hypotheses or candidates for review, never
as proof, criminality, guilt, or instructions to block an account.

Important sampling limits: the graph expands outgoing payments from 81 seed
clients for four hops. Incoming activity of seeds can be incomplete. At depth 4,
outgoing transfers beyond the boundary are absent; an out_deg of zero is not proof
of retention. Mention the relevant limitation whenever it affects the conclusion.
If a requested fact is not available, say so plainly rather than estimating it.
"""


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values to ordinary JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@dataclass
class GraphInvestigationTools:
    """Read-only deterministic graph lookups exposed to the language model."""

    nodes: pd.DataFrame
    top_nodes: pd.DataFrame
    clusters: pd.DataFrame
    edges: pd.DataFrame

    def __post_init__(self) -> None:
        self.nodes = self.nodes.copy()
        self.nodes["gid"] = pd.to_numeric(self.nodes["gid"], errors="coerce")
        self.nodes = self.nodes.dropna(subset=["gid"])
        self.nodes["gid"] = self.nodes["gid"].astype("int64")
        self.edges = self.edges.copy()
        if not self.edges.empty:
            self.edges["src"] = pd.to_numeric(self.edges["src"], errors="coerce")
            self.edges["dst"] = pd.to_numeric(self.edges["dst"], errors="coerce")
            self.edges = self.edges.dropna(subset=["src", "dst"])
            self.edges[["src", "dst"]] = self.edges[["src", "dst"]].astype("int64")
        self.graph = nx.from_pandas_edgelist(
            self.edges, "src", "dst", edge_attr=True, create_using=nx.DiGraph
        )

    def _record(self, frame: pd.DataFrame, gid: int) -> dict[str, Any] | None:
        row = frame.loc[frame["gid"] == int(gid)]
        if row.empty:
            return None
        return _json_safe(row.iloc[0].dropna().to_dict())

    def get_node(self, gid: int) -> dict[str, Any]:
        record = self._record(self.nodes, gid)
        if record is None:
            return {"found": False, "gid": gid}
        record["found"] = True
        return record

    def get_top_nodes(self, limit: int = 20, role: str | None = None) -> list[dict[str, Any]]:
        frame = self.top_nodes.copy()
        if frame.empty:
            frame = self.nodes.copy()
        if role and "role" in frame:
            frame = frame[frame["role"].astype(str).str.lower() == role.lower()]
        score = "priority_score" if "priority_score" in frame else "rank"
        if score in frame:
            frame = frame.sort_values(score, ascending=(score == "rank"))
        return _json_safe(frame.head(max(1, min(int(limit), 100))).to_dict("records"))

    def get_cluster(self, cluster_id: str | int) -> dict[str, Any]:
        if "cluster_id" not in self.nodes:
            return {"found": False, "reason": "cluster_id is not present in nodes_roles.csv"}
        members = self.nodes[self.nodes["cluster_id"].astype(str) == str(cluster_id)]
        exported = pd.DataFrame()
        if "cluster_id" in self.clusters:
            exported = self.clusters[self.clusters["cluster_id"].astype(str) == str(cluster_id)]
        result: dict[str, Any] = {
            "found": not members.empty,
            "cluster_id": cluster_id,
            "member_count": int(len(members)),
            "top_gids": _json_safe(members.sort_values("priority_score", ascending=False)
                              .head(10)["gid"].tolist()) if "priority_score" in members else [],
        }
        if not exported.empty:
            result["exported_summary"] = _json_safe(exported.iloc[0].dropna().to_dict())
        return result

    def get_counterparties(self, gid: int, limit: int = 25) -> dict[str, Any]:
        gid = int(gid)
        inbound = self.edges[self.edges["dst"] == gid]
        outbound = self.edges[self.edges["src"] == gid]
        fields = [c for c in ["src", "dst", "sum_kzt", "n_tx", "depth"] if c in self.edges]
        return {
            "gid": gid,
            "inbound": _json_safe(inbound.sort_values("sum_kzt", ascending=False).head(limit)[fields].to_dict("records")),
            "outbound": _json_safe(outbound.sort_values("sum_kzt", ascending=False).head(limit)[fields].to_dict("records")),
        }

    def find_common_descendants(self, gids: list[int], max_hops: int = 4) -> dict[str, Any]:
        gids = [int(g) for g in gids]
        if not gids:
            return {"gids": [], "common_descendants": []}
        descendant_sets = []
        for gid in gids:
            if gid not in self.graph:
                descendant_sets.append(set())
                continue
            lengths = nx.single_source_shortest_path_length(self.graph, gid, cutoff=max(1, min(max_hops, 8)))
            descendant_sets.append(set(lengths) - {gid})
        common = set.intersection(*descendant_sets) if descendant_sets else set()
        return {"gids": gids, "max_hops": max_hops, "common_descendants": sorted(common)[:100]}

    def find_paths(self, src: int, dst: int, max_hops: int = 6, limit: int = 10) -> dict[str, Any]:
        src, dst = int(src), int(dst)
        if src not in self.graph or dst not in self.graph:
            return {"found": False, "reason": "source or destination is absent from the supplied graph"}
        paths = []
        try:
            for path in nx.all_simple_paths(self.graph, src, dst, cutoff=max(1, min(max_hops, 10))):
                paths.append(path)
                if len(paths) >= max(1, min(limit, 25)):
                    break
        except nx.NetworkXNoPath:
            pass
        return {"found": bool(paths), "src": src, "dst": dst, "paths": paths,
                "note": "Paths are only within the supplied four-hop sample."}

    def compare_nodes(self, gid1: int, gid2: int) -> dict[str, Any]:
        return {"node_1": self.get_node(int(gid1)), "node_2": self.get_node(int(gid2))}


TOOL_SPECS = [
    {"type": "function", "function": {"name": "get_node", "description": "Get all exported metrics for one gid.", "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}}, "required": ["gid"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_top_nodes", "description": "Get ranked exported candidates; may filter by role.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}, "role": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_cluster", "description": "Get exported summary and members for one cluster.", "parameters": {"type": "object", "properties": {"cluster_id": {"type": ["string", "integer"]}}, "required": ["cluster_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "find_common_descendants", "description": "Find common directed descendants in the supplied graph.", "parameters": {"type": "object", "properties": {"gids": {"type": "array", "items": {"type": "integer"}}, "max_hops": {"type": "integer"}}, "required": ["gids"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_counterparties", "description": "Get largest inbound and outbound counterparties for a gid.", "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["gid"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "find_paths", "description": "Find sampled directed paths between two gids.", "parameters": {"type": "object", "properties": {"src": {"type": "integer"}, "dst": {"type": "integer"}, "max_hops": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["src", "dst"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "compare_nodes", "description": "Compare two exported node records.", "parameters": {"type": "object", "properties": {"gid1": {"type": "integer"}, "gid2": {"type": "integer"}}, "required": ["gid1", "gid2"], "additionalProperties": False}}},
]


def answer_question(question: str, tools: GraphInvestigationTools, model: str = "gpt-4o-mini") -> str:
    """Use tool calling; model output is permitted only after deterministic calls.

    The OpenAI import is deliberately local so missing optional dependencies never
    affect the Streamlit application.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the optional `openai` package to use AI Analyst.") from exc

    client = OpenAI()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    functions: dict[str, Callable[..., Any]] = {
        name: getattr(tools, name) for name in (
            "get_node", "get_top_nodes", "get_cluster", "find_common_descendants",
            "get_counterparties", "find_paths", "compare_nodes"
        )
    }
    for _ in range(6):
        response = client.chat.completions.create(model=model, messages=messages, tools=TOOL_SPECS, tool_choice="auto")
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content or "No grounded response was returned."
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments)
                result = functions[call.function.name](**args)
            except Exception as exc:  # tool errors are context, not hidden facts
                result = {"error": str(exc)}
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(_json_safe(result), ensure_ascii=False)})
    return "I reached the tool-call limit before completing a grounded response. Please narrow the question."
