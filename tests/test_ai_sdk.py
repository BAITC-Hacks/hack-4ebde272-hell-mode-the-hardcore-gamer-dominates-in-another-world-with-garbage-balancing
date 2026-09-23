"""Exercise the installed SDK over an in-memory transport, never the live API."""
import json

import pandas as pd
import pytest

from src.ai_assistant import GraphInvestigationTools, answer_question_result


@pytest.fixture
def configured_tools(monkeypatch, tmp_path):
    monkeypatch.setattr("src.config.DEFAULT_ENV_PATH", tmp_path / ".env")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=fake-sdk-test-key\nOPENAI_MODEL=gpt-4o-mini\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    nodes = pd.DataFrame({"gid": [2**63 - 1], "role": ["consolidator"],
        "priority_score": [.8], "depth": [1], "is_seed": [False]})
    return GraphInvestigationTools(nodes, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def test_real_sdk_tool_round_trip_uses_env_file_and_exact_ids(configured_tools, monkeypatch):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx2")
    requests = []
    gid = str(2**63 - 1)

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.headers["authorization"] == "Bearer fake-sdk-test-key"
        assert payload["model"] == "gpt-4o-mini"
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_node", "type": "function", "function": {
                    "name": "get_node", "arguments": json.dumps({"gid": gid})}}]}
            finish = "tool_calls"
        else:
            tool = json.loads(payload["messages"][-1]["content"])
            assert tool["source_id"] == "S1" and tool["result"]["gid"] == gid
            message = {"role": "assistant", "content": json.dumps({"claims": [{
                "source_id": "S1", "path": "/role", "value": "consolidator"}]})}
            finish = "stop"
        return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion",
            "created": 1, "model": "gpt-4o-mini", "choices": [
                {"index": 0, "finish_reason": finish, "message": message}]})

    with openai.OpenAI(api_key="fake-sdk-test-key", base_url="https://unit.test/v1",
                       http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
        answer = answer_question_result(f"Why review {gid}?", configured_tools)
    assert len(requests) == 2
    assert requests[0]["tool_choice"] == "required"
    assert "role = consolidator" in answer.text
    assert answer.node_gids == (gid,)
    assert answer.rejected_claims == 0


@pytest.mark.parametrize("status", [401, 429, 500])
def test_real_sdk_errors_never_show_provider_response_or_key(configured_tools, monkeypatch, status):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx2")
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={
        "error": {"message": "provider response contains fake-sdk-test-key", "type": "test_error"}}))
    with openai.OpenAI(api_key="fake-sdk-test-key", max_retries=0,
                       http_client=httpx.Client(transport=transport)) as client:
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
        with pytest.raises(RuntimeError, match="AI request failed") as caught:
            answer_question_result("Explain this account", configured_tools)
    assert "fake-sdk-test-key" not in str(caught.value)
    assert "provider response" not in str(caught.value)
