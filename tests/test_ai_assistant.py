"""Local graph-tool and mocked assistant tests; no external API calls."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.ai_assistant import (
    GraphInvestigationTools, answer_question, answer_question_result, validate_answer,
    TOOL_SPECS, _json_safe, _tool_payload,
)


@pytest.fixture
def tools():
    # Adjacent integers above JavaScript's safe integer limit must stay distinct.
    gids = [100000000000000001, 100000000000000002, 100000000000000003, 100000000000000004]
    nodes = pd.DataFrame({"gid": gids, "priority_score": [0.9, 0.8, 0.7, 0.1],
                          "cluster_id": [0, 0, 1, 2], "role": ["transit", "transit", "terminal", "peripheral"],
                          "depth": [1, 4, 2, 0], "is_seed": [False, False, False, True],
                          "relay_2d_ratio": [0.5, None, 0.0, None]})
    edges = pd.DataFrame({"src": gids[:2], "dst": gids[1:3], "sum_kzt": [4.0, 3.0], "n_tx": [1, 1]})
    return GraphInvestigationTools(nodes, nodes.head(1), pd.DataFrame(), edges)


def test_graph_tools_preserve_ids_and_support_isolated_nodes(tools):
    gids = tools.nodes.gid.tolist()
    assert tools.get_node(gids[0])["gid"] == gids[0]
    assert not tools.get_node(-1)["found"]
    assert len(tools.get_top_nodes(20)) == 4
    assert tools.get_top_nodes(role="terminal")[0]["gid"] == gids[2]
    assert tools.get_cluster(0)["member_count"] == 2
    assert tools.get_counterparties(gids[1])["inbound"][0]["src"] == gids[0]
    assert tools.find_common_descendants(gids[:2])["common_descendants"] == [gids[2]]
    assert tools.find_paths(gids[0], gids[2])["paths"] == [gids[:3]]
    assert not tools.find_paths(gids[2], gids[0])["found"]
    assert tools.find_paths(gids[3], gids[3])["paths"] == [[gids[3]]]
    assert tools.compare_nodes(gids[0], gids[1])["node_2"]["gid"] == gids[1]
    json.dumps(_json_safe({"a": float("nan"), "b": float("inf")}), allow_nan=False)


def test_empty_edges_are_supported(tools):
    tools = GraphInvestigationTools(tools.nodes, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert tools.get_counterparties(int(tools.nodes.gid.iloc[0]))["outbound"] == []
    assert not tools.find_paths(int(tools.nodes.gid.iloc[0]), -1)["found"]


def test_path_search_has_a_work_budget(tools):
    for a in range(180):
        for b in range(a + 1, min(a + 30, 180)):
            tools.graph.add_edge(a, b)
    tools.graph.add_node(999)
    result = tools.find_paths(0, 999, max_hops=10)
    assert not result["found"]
    assert result["truncated"]


def message(content=None, calls=None):
    return SimpleNamespace(content=content, tool_calls=calls,
                           model_dump=lambda **kwargs: {"role": "assistant", "content": content})


def claim(path, value, source_id="S1"):
    return {"source_id": source_id, "path": path, "value": value}


def install_mock(monkeypatch, *messages):
    client = MagicMock()
    client.chat.completions.create.side_effect = [SimpleNamespace(choices=[SimpleNamespace(message=m)]) for m in messages]
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))
    return client


def tool_message(name, args):
    call = SimpleNamespace(id="call-1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return message(calls=[call])


def final_message(*claims):
    return message(content=json.dumps({"claims": list(claims)}))


def ledger_for(tools, gid):
    return {"S1": {"tool": "get_node", "arguments": {"gid": str(gid)}, "result": _tool_payload(tools.get_node(gid))}}


def test_assistant_validates_each_claim_and_returns_exact_navigation(tools, monkeypatch):
    gid = str(tools.nodes.gid.iloc[0])
    client = install_mock(monkeypatch,
                          tool_message("get_node", {"gid": gid}),
                          final_message(claim("/priority_score", 0.9), claim("/role", "transit")))
    result = answer_question_result("Explain this node", tools)
    assert result.node_gids == (gid,)
    assert result.sources[0] == {"source_id": "S1", "tool": "get_node", "arguments": {"gid": gid},
                                 "path": "/priority_score", "value": 0.9}
    assert "priority score = 0\\.9" in result.text
    assert "investigation hypotheses" in result.text
    calls = client.chat.completions.create.call_args_list
    assert calls[0].kwargs["tool_choice"] == "required"
    assert calls[1].kwargs["tool_choice"] == "auto"
    assert calls[1].kwargs["response_format"]["json_schema"]["strict"]
    tool_messages = [m for m in calls[1].kwargs["messages"] if m["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"])["result"]["gid"] == gid


def test_ungrounded_response_is_not_shown(tools, monkeypatch):
    install_mock(monkeypatch, message(content="Invented evidence"))
    assert "No verified tool result" in answer_question("Explain", tools)


@pytest.mark.parametrize("content", [
    "This client is guilty because of their transfers.",
    json.dumps({"claims": [], "explanation": "This client controls every account."}),
    json.dumps({"claims": "made up"}),
])
def test_successful_tool_call_does_not_validate_free_prose(tools, monkeypatch, content):
    install_mock(monkeypatch, tool_message("get_node", {"gid": str(tools.nodes.gid.iloc[0])}), message(content=content))
    result = answer_question_result("Explain", tools)
    assert "rejected" in result.text
    assert not result.sources
    assert not result.node_gids
    assert "guilty" not in result.text


@pytest.mark.parametrize("bad_claim", [
    claim("/priority_score", 0.99), claim("/role", "coordinator"),
    claim("/undocumented_loss", 1000000), claim("/priority_score", 0.9, "S999"),
    claim("/gid", 100000000000000001), claim("/priority_score", True),
    {**claim("/priority_score", 0.9), "explanation": "This proves guilt"},
    claim("/priority_score", float("nan")),
])
def test_unsupported_claims_are_omitted_without_losing_valid_evidence(tools, bad_claim):
    gid = str(tools.nodes.gid.iloc[0])
    result = validate_answer(json.dumps({"claims": [claim("/role", "transit"), bad_claim]}), ledger_for(tools, gid), tools)
    assert result.rejected_claims == 1
    assert len(result.sources) == 1
    assert result.node_gids == (gid,)
    assert "unsupported claim(s) were omitted" in result.text


def test_failed_tool_cannot_create_an_evidence_source(tools, monkeypatch):
    monkeypatch.setattr(tools, "get_node", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("private failure detail")))
    client = install_mock(monkeypatch, tool_message("get_node", {"gid": "1"}), final_message(claim("/priority_score", 1)))
    result = answer_question_result("Explain", tools)
    assert "No verified tool result" in result.text
    assert not result.sources and not result.node_gids
    messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert "private failure detail" not in json.dumps(messages)
    assert client.chat.completions.create.call_args_list[1].kwargs["tool_choice"] == "required"


def test_missing_node_reports_absence_but_has_no_navigation(tools, monkeypatch):
    install_mock(monkeypatch, tool_message("get_node", {"gid": "-123"}),
                 final_message(claim("/found", False), claim("/priority_score", 0.9)))
    result = answer_question_result("Explain -123", tools)
    assert "gid -123" in result.text and "no result in the supplied sample" in result.text
    assert result.rejected_claims == 1
    assert not result.node_gids


def test_known_isolate_and_unknown_counterparty_lookup_differ(tools):
    isolated = tools.get_counterparties(str(tools.nodes.gid.iloc[3]))
    assert isolated["found"] and isolated["inbound"] == isolated["outbound"] == []
    assert tools.get_counterparties("-1")["found"] is False


def test_boundary_and_seed_caveats_are_added_without_model_claims(tools):
    for index, phrase in [(1, "beyond hop 4"), (3, "incoming seed activity is incomplete")]:
        gid = str(tools.nodes.gid.iloc[index])
        result = validate_answer(json.dumps({"claims": [claim("/role", tools.get_node(gid)["role"])]}), ledger_for(tools, gid), tools)
        assert phrase in result.text


def test_null_temporal_evidence_is_unavailable_and_has_date_caveat(tools):
    gid = str(tools.nodes.gid.iloc[1])
    result = validate_answer(json.dumps({"claims": [claim("/relay_2d_ratio", None)]}), ledger_for(tools, gid), tools)
    assert "unavailable" in result.text and "date-only" in result.text
    assert "beyond hop 4" in result.text


def test_paths_provide_validated_navigation_to_exact_interior_node(tools):
    gids = [str(gid) for gid in tools.nodes.gid.iloc[:3]]
    ledger = {"S1": {"tool": "find_paths", "arguments": {"src": gids[0], "dst": gids[2]},
                     "result": _tool_payload(tools.find_paths(gids[0], gids[2]))}}
    assert ledger["S1"]["result"]["paths"] == [gids]
    result = validate_answer(json.dumps({"claims": [claim("/paths/0/1", gids[1])]}), ledger, tools)
    assert result.node_gids == tuple(gids)
    assert "bounded searches" in result.text


@pytest.mark.parametrize("value", [1.5, float(100000000000000001), True, "1e17", "9223372036854775808"])
def test_lossy_or_invalid_identifiers_rejected(tools, value):
    with pytest.raises(ValueError):
        tools.get_node(value)
    nodes = pd.DataFrame({"gid": pd.Series([value], dtype="object")})
    with pytest.raises(ValueError):
        GraphInvestigationTools(nodes, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def test_string_identifiers_and_sdk_tool_schemas_preserve_adjacent_gids(tools):
    frame = tools.nodes.copy()
    frame["gid"] = frame.gid.astype(str)
    exact = GraphInvestigationTools(frame, pd.DataFrame(), pd.DataFrame(), tools.edges)
    assert len(set(exact.nodes.gid)) == 4
    for spec in TOOL_SPECS:
        for name, schema in spec["function"]["parameters"]["properties"].items():
            if name in {"gid", "gid1", "gid2", "src", "dst"}:
                assert schema["type"] == "string"


def test_tool_limit_is_bounded_without_returning_unverified_claims(tools, monkeypatch):
    client = install_mock(monkeypatch, *[tool_message("get_node", {"gid": str(tools.nodes.gid.iloc[0])}) for _ in range(6)])
    result = answer_question_result("Explain", tools)
    assert "tool-call limit" in result.text
    assert client.chat.completions.create.call_count == 6
    assert not result.node_gids
