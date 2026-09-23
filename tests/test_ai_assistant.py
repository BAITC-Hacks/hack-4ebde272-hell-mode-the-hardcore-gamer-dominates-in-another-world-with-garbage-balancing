"""Local graph-tool and mocked assistant tests; no external API calls."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.ai_assistant import GraphInvestigationTools, answer_question, _json_safe


@pytest.fixture
def tools():
    # Adjacent integers above JavaScript's safe integer limit must stay distinct.
    gids = [100000000000000001, 100000000000000002, 100000000000000003, 100000000000000004]
    nodes = pd.DataFrame({"gid": gids, "priority_score": [0.9, 0.8, 0.7, 0.1],
                          "cluster_id": [0, 0, 1, 2], "role": ["transit", "transit", "terminal", "peripheral"]})
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


def test_assistant_requires_tools_before_returning_an_answer(tools, monkeypatch):
    gid = int(tools.nodes.gid.iloc[0])
    call = SimpleNamespace(id="call-1", function=SimpleNamespace(name="get_node", arguments=json.dumps({"gid": gid})))
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=message(calls=[call]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=message(content="Grounded test answer"))]),
    ]
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))
    assert answer_question("Explain this node", tools) == "Grounded test answer"
    calls = client.chat.completions.create.call_args_list
    assert calls[0].kwargs["tool_choice"] == "required"
    assert calls[1].kwargs["tool_choice"] == "auto"
    tool_messages = [m for m in calls[1].kwargs["messages"] if m["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"])["gid"] == gid


def test_ungrounded_response_is_not_shown(tools, monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=message(content="Invented evidence"))])
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))
    assert "No verified tool result" in answer_question("Explain", tools)
