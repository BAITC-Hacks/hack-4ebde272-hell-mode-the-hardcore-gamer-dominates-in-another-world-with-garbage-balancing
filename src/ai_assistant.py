"""Grounded, optional AI assistant for the Money Graph analyst interface.

The core UI never imports OpenAI.  This module is loaded only after a user asks a
question and an ``OPENAI_API_KEY`` is present.  The model has no data access
other than the deterministic functions in :class:`GraphInvestigationTools`.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import deque
from dataclasses import dataclass
from numbers import Integral
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

Return your final answer as JSON with only a claims array. Each claim must contain
source_id, path (an RFC 6901 JSON pointer into that tool's result), and value (the
exact scalar at that path). Select the fields relevant to the question, including
exported evidence/explanations. Do not add prose, interpretations, labels, URLs,
or calculations. The application validates each claim and renders the explanation
and sampling limitations itself. Tool outputs include source_id and result.
Identifiers must be copied as exact decimal strings; never convert them to floats.
"""


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values to ordinary JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except ValueError:
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _exact_gid(value: Any) -> int:
    """Reject lossy floating-point identifiers, including apparently integral ones."""
    if isinstance(value, bool) or not (
        isinstance(value, Integral) or isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value)
    ):
        raise ValueError("gid must be an exact integer or decimal string, never a float")
    parsed = int(value)
    if not -(2**63) <= parsed < 2**63:
        raise ValueError("gid is outside signed int64 range")
    return parsed


_ID_KEYS = {"gid", "gid1", "gid2", "src", "dst"}
_ID_LIST_KEYS = {"gids", "top_gids", "common_descendants", "paths"}


def _tool_payload(value: Any, key: str = "") -> Any:
    """Keep identifiers exact across JSON, model output and browser navigation."""
    value = _json_safe(value)
    if isinstance(value, dict):
        return {k: _tool_payload(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_tool_payload(v, key) for v in value]
    if value is not None and key in _ID_KEYS | _ID_LIST_KEYS:
        # Some cluster exports store top_gids as a delimited string, not a list.
        if key == "top_gids" and isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value):
            return value
        return str(_exact_gid(value))
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
        self.nodes["gid"] = pd.Series([_exact_gid(gid) for gid in self.nodes["gid"]], index=self.nodes.index, dtype="int64")
        if self.nodes["gid"].duplicated().any():
            raise ValueError("nodes must contain one record per gid")
        self.top_nodes = self.top_nodes.copy()
        if "gid" in self.top_nodes:
            self.top_nodes["gid"] = pd.Series([_exact_gid(gid) for gid in self.top_nodes["gid"]], index=self.top_nodes.index, dtype="int64")
        self.edges = self.edges.copy()
        if not self.edges.empty:
            for column in ("src", "dst"):
                self.edges[column] = pd.Series([_exact_gid(gid) for gid in self.edges[column]], index=self.edges.index, dtype="int64")
                if not self.edges[column].isin(self.nodes["gid"]).all():
                    raise ValueError("edges reference nodes absent from the exported node table")
        if self.edges.empty:
            self.edges = pd.DataFrame(columns=["src", "dst", "sum_kzt", "n_tx"])
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes["gid"])
        for row in self.edges.to_dict("records"):
            self.graph.add_edge(row["src"], row["dst"], **{k: v for k, v in row.items() if k not in ("src", "dst")})

    def _record(self, frame: pd.DataFrame, gid: int) -> dict[str, Any] | None:
        row = frame.loc[frame["gid"] == _exact_gid(gid)]
        if row.empty:
            return None
        return _json_safe(row.to_dict("records")[0])

    def get_node(self, gid: int) -> dict[str, Any]:
        gid = _exact_gid(gid)
        record = self._record(self.nodes, gid)
        if record is None:
            return {"found": False, "gid": gid}
        record["found"] = True
        return record

    def get_top_nodes(self, limit: int = 20, role: str | None = None) -> list[dict[str, Any]]:
        frame = self.nodes.copy() if "priority_score" in self.nodes else self.top_nodes.copy()
        if role and "role" in frame:
            frame = frame[frame["role"].astype(str).str.lower() == role.lower()]
        score = "priority_score" if "priority_score" in frame else "rank"
        if score in frame:
            frame = frame.sort_values([score, "gid"], ascending=[score == "rank", True], kind="stable")
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
            "top_gids": _json_safe(members.sort_values(
                ["priority_score", "gid"], ascending=[False, True], kind="stable")
                              .head(10)["gid"].tolist()) if "priority_score" in members else [],
        }
        if not exported.empty:
            result["exported_summary"] = _json_safe(exported.iloc[0].dropna().to_dict())
        return result

    def get_counterparties(self, gid: int, limit: int = 25) -> dict[str, Any]:
        gid = _exact_gid(gid)
        if gid not in self.graph:
            return {"found": False, "gid": gid, "reason": "gid is absent from the supplied node table"}
        limit = max(1, min(int(limit), 100))
        inbound = self.edges[self.edges["dst"] == gid]
        outbound = self.edges[self.edges["src"] == gid]
        fields = [c for c in ["src", "dst", "sum_kzt", "n_tx", "depth"] if c in self.edges]
        return {
            "found": True,
            "gid": gid,
            "inbound": _json_safe(inbound.sort_values(
                ["sum_kzt", "src", "dst"], ascending=[False, True, True], kind="stable")
                .head(limit)[fields].to_dict("records")),
            "outbound": _json_safe(outbound.sort_values(
                ["sum_kzt", "src", "dst"], ascending=[False, True, True], kind="stable")
                .head(limit)[fields].to_dict("records")),
        }

    def find_common_descendants(self, gids: list[int], max_hops: int = 4) -> dict[str, Any]:
        gids = [_exact_gid(g) for g in gids]
        max_hops = max(1, min(int(max_hops), 8))
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
        src, dst = _exact_gid(src), _exact_gid(dst)
        if src not in self.graph or dst not in self.graph:
            return {"found": False, "reason": "source or destination is absent from the supplied graph"}
        max_hops, limit = max(1, min(int(max_hops), 10)), max(1, min(int(limit), 25))
        # Bounded BFS prevents an exponential simple-path scan from freezing the UI.
        paths, pending = [], deque([[src]])
        examined, budget = 0, 20_000
        while pending and len(paths) < limit and examined < budget:
            path = pending.popleft()
            if path[-1] == dst:
                paths.append(path)
                continue
            if len(path) - 1 >= max_hops:
                continue
            for neighbor in sorted(self.graph.successors(path[-1])):
                examined += 1
                if neighbor not in path:
                    pending.append([*path, neighbor])
                if examined >= budget:
                    break
        return {"found": bool(paths), "src": src, "dst": dst, "paths": paths,
                "truncated": bool(pending), "max_hops": max_hops,
                "note": "Paths are only within the supplied four-hop sample; search is bounded."}

    def compare_nodes(self, gid1: int, gid2: int) -> dict[str, Any]:
        return {"node_1": self.get_node(gid1), "node_2": self.get_node(gid2)}


TOOL_SPECS = [
    {"type": "function", "function": {"name": "get_node", "description": "Get all exported metrics for one gid.", "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}}, "required": ["gid"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_top_nodes", "description": "Get ranked exported candidates; may filter by role.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}, "role": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_cluster", "description": "Get exported summary and members for one cluster.", "parameters": {"type": "object", "properties": {"cluster_id": {"type": ["string", "integer"]}}, "required": ["cluster_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "find_common_descendants", "description": "Find common directed descendants in the supplied graph.", "parameters": {"type": "object", "properties": {"gids": {"type": "array", "items": {"type": "integer"}}, "max_hops": {"type": "integer"}}, "required": ["gids"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_counterparties", "description": "Get largest inbound and outbound counterparties for a gid.", "parameters": {"type": "object", "properties": {"gid": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["gid"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "find_paths", "description": "Find sampled directed paths between two gids.", "parameters": {"type": "object", "properties": {"src": {"type": "integer"}, "dst": {"type": "integer"}, "max_hops": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["src", "dst"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "compare_nodes", "description": "Compare two exported node records.", "parameters": {"type": "object", "properties": {"gid1": {"type": "integer"}, "gid2": {"type": "integer"}}, "required": ["gid1", "gid2"], "additionalProperties": False}}},
]


for _spec in TOOL_SPECS:
    for _key, _property in _spec["function"]["parameters"]["properties"].items():
        if _key in _ID_KEYS:
            _property["type"] = "string"
            _property["description"] = "Exact decimal gid string; do not round."
        elif _key == "gids":
            _property["items"] = {"type": "string"}


ANSWER_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "cited_graph_evidence", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {"claims": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "source_id": {"type": "string"}, "path": {"type": "string"},
                    "value": {"type": ["string", "number", "boolean", "null"]},
                },
                "required": ["source_id", "path", "value"],
            }}},
            "required": ["claims"],
        },
    },
}


@dataclass(frozen=True)
class GroundedAnswer:
    """Only validated facts and existing decimal gids may reach the UI."""

    text: str
    node_gids: tuple[str, ...] = ()
    sources: tuple[dict[str, Any], ...] = ()
    rejected_claims: int = 0


def _pointer(result: Any, path: str) -> tuple[Any, list[Any], list[str]]:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("Claims require a JSON pointer to an observed field")
    raw = path[1:].split("/")
    if any(re.search(r"~(?![01])", part) for part in raw):
        raise ValueError("Invalid JSON pointer escape")
    parts = [part.replace("~1", "/").replace("~0", "~") for part in raw]
    ancestors = []
    value = result
    for part in parts:
        ancestors.append(value)
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9]\d*", part):
            value = value[int(part)]
        else:
            raise ValueError("Claim does not reference an observed field")
    if isinstance(value, (dict, list)):
        raise ValueError("Claims must reference individual scalar fields")
    return value, ancestors, parts


def _same_scalar(actual: Any, claimed: Any) -> bool:
    if isinstance(actual, (bool, str)) or actual is None:
        return type(actual) is type(claimed) and actual == claimed
    return type(claimed) in (int, float) and math.isfinite(claimed) and actual == claimed


def _markdown_text(value: Any) -> str:
    """Escape evidence as plain text; dataset strings cannot create navigation."""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|])", r"\\\1", text)


def validate_answer(content: str, ledger: dict[str, dict[str, Any]], tools: GraphInvestigationTools) -> GroundedAnswer:
    """Validate each field/value claim, never treat a tool call as prose verification.

    Free-form model text is intentionally not displayed. The model chooses the
    relevant exported facts; rendering, citations, links and caveats are local.
    """
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return GroundedAnswer("The assistant response was rejected: it did not contain verifiable cited claims.", rejected_claims=1)
    if not isinstance(payload, dict) or set(payload) != {"claims"} or not isinstance(payload["claims"], list):
        return GroundedAnswer("The assistant response was rejected: unsupported text or invalid claim structure.", rejected_claims=1)
    sources, lines, node_gids = [], [], set()
    node_paths: dict[str, set[str]] = {}
    rejected = max(0, len(payload["claims"]) - 30)
    seen = set()
    for claim in payload["claims"][:30]:
        try:
            if not isinstance(claim, dict) or set(claim) != {"source_id", "path", "value"}:
                raise ValueError("Unsupported assertion")
            source_id, path = claim["source_id"], claim["path"]
            source = ledger[source_id]
            if isinstance(source["result"], dict) and "error" in source["result"]:
                raise ValueError("Failed tools cannot support claims")
            actual, ancestors, parts = _pointer(source["result"], path)
            if not _same_scalar(actual, claim["value"]):
                raise ValueError("Claim value differs from source")
            # References below found:false cannot invent a valid client identity.
            missing = any(isinstance(item, dict) and item.get("found") is False and "gid" in item for item in ancestors)
            if missing and parts[-1] not in {"found", "gid", "reason"}:
                raise ValueError("Missing nodes have no observed metrics")
            claim_gids = set()
            context = ""
            for ancestor in reversed(ancestors):
                if (source["tool"] == "get_counterparties" and isinstance(ancestor, dict)
                        and {"src", "dst"}.issubset(ancestor)):
                    # The amount belongs to this directed relationship, not
                    # the focus client's total or an unspecified counterparty.
                    context = f"observed edge {ancestor['src']} → {ancestor['dst']}: "
                    break
                if isinstance(ancestor, dict) and "gid" in ancestor:
                    context = f"gid {ancestor['gid']}: "
                    if not missing and _exact_gid(ancestor["gid"]) in tools.graph:
                        claim_gids.add(str(_exact_gid(ancestor["gid"])))
                    break
            if not context:
                args = source["arguments"]
                context = f"{source['tool']}({json.dumps(args, ensure_ascii=False)}): "
            if not missing:
                for ancestor in ancestors:
                    if isinstance(ancestor, dict):
                        for key in ("src", "dst"):
                            if key in ancestor and _exact_gid(ancestor[key]) in tools.graph:
                                claim_gids.add(str(_exact_gid(ancestor[key])))
                if any(part in _ID_KEYS | _ID_LIST_KEYS for part in parts):
                    gid = _exact_gid(actual)
                    if gid in tools.graph:
                        claim_gids.add(str(gid))
            if (source_id, path) in seen:
                continue
            seen.add((source_id, path))
            node_gids.update(claim_gids)
            for gid in claim_gids:
                node_paths.setdefault(gid, set()).add(path)
            display = "unavailable" if actual is None else str(actual)
            label = parts[-1].replace("_", " ")
            if label == "found" and actual is False:
                display = "no result in the supplied sample"
            lines.append(f"- {_markdown_text(context + label + ' = ' + display)} [{source_id}]")
            sources.append({"source_id": source_id, "tool": source["tool"],
                            "arguments": source["arguments"], "path": path, "value": actual})
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            rejected += 1
    if not sources:
        return GroundedAnswer("No supported claims were returned. Ask about a supplied gid or available metric.", rejected_claims=rejected)
    limitations = ["These exported metrics support investigation hypotheses and candidates for review; they do not establish guilt."]
    for gid in sorted(node_gids, key=int):
        record = tools.get_node(gid)
        if record.get("depth") == 4:
            limitations.append(f"gid {gid}: outgoing transfers beyond hop 4 are not present in the supplied sample. Do not interpret out_deg=0 as confirmed retention.")
        if record.get("is_seed") in (True, 1):
            limitations.append(f"gid {gid}: incoming seed activity is incomplete; observed inbound/outbound values are not complete balances.")
        for prefix, label in (("repeated_route", "repeated-route"), ("temporal_return", "reciprocal-date return")):
            if any(prefix in path for path in node_paths.get(gid, ())):
                if record.get(prefix + "_truncated") is True:
                    limitations.append(f"gid {gid}: the {label} search was truncated; exported counts and support are lower bounds within the supplied sample.")
                else:
                    limitations.append(f"gid {gid}: {label} evidence comes from a bounded search of the supplied sample; only limited candidate/date examples are exported. Absent evidence does not rule out other activity.")
    date_fields = ("relay", "burst", "date", "temporal", "same_day", "repeated_route",
                   "amount", "active_days", "max_in_senders", "peak_day_share")
    if any(any(term in source["path"] for term in date_fields) for source in sources):
        limitations.append("Observations are date-only. Date overlap does not prove intraday ordering or movement of the same funds; unavailable/censored windows are not zero activity.")
    if any("amount" in source["path"] for source in sources):
        limitations.append("Amount patterns describe observed transfers only, not intentional splitting. Transfers below the sample's 5,000 KZT threshold are unobserved; exact-repeat and similar-amount groups may overlap, so their shares must not be added.")
    if any(source["tool"] in {"find_paths", "find_common_descendants"} for source in sources):
        limitations.append("Routes and descendants are bounded searches of the supplied directed sample; absent results do not rule out paths outside the sample or search limit.")
    text = "Verified exported evidence:\n\n" + "\n".join(lines) + "\n\n" + "\n\n".join(limitations)
    if rejected:
        text += f"\n\n{rejected} unsupported claim(s) were omitted."
    return GroundedAnswer(text, tuple(sorted(node_gids, key=int)), tuple(sources), rejected)


def answer_question_result(question: str, tools: GraphInvestigationTools, model: str | None = None) -> GroundedAnswer:
    """Select and validate cited tool evidence; never render arbitrary model prose.

    The OpenAI import is deliberately local so missing optional dependencies never
    affect the Streamlit application.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the optional `openai` package to use AI Analyst.") from exc

    client = OpenAI(timeout=30.0, max_retries=1)
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
    ledger: dict[str, dict[str, Any]] = {}
    for _ in range(6):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SPECS,
            tool_choice="auto" if ledger else "required", response_format=ANSWER_FORMAT,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            if not ledger:
                return GroundedAnswer("No verified tool result was returned. Please ask about a specific node or cluster.")
            return validate_answer(message.content, ledger, tools)
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments)
                result = _tool_payload(functions[call.function.name](**args))
                if isinstance(result, dict) and "error" in result:
                    raise ValueError("Tool returned an error")
                source_id = f"S{len(ledger) + 1}"
                ledger[source_id] = {"tool": call.function.name, "arguments": _tool_payload(args), "result": result}
                result = {"source_id": source_id, "result": result}
            except Exception as exc:  # tool errors are context, not hidden facts
                result = {"error": type(exc).__name__, "message": "Tool failed; no evidence source was created."}
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(_json_safe(result), ensure_ascii=False, allow_nan=False)})
    return GroundedAnswer("I reached the tool-call limit before completing a grounded response. Please narrow the question.")


def answer_question(question: str, tools: GraphInvestigationTools, model: str | None = None) -> str:
    """Compatibility wrapper for clients that only render the validated text."""
    return answer_question_result(question, tools, model).text
