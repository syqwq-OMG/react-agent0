"""§7: HTTPX transport doubles; no API key, model download, or live server.

Both OpenAI SDK (which uses HTTPX) and direct HTTPX/native Ollama clients work.
The tests inspect transport boundaries, never private provider attributes.
"""
import json

import httpx
import pytest

from adapters import make_model


@pytest.fixture
def transport(monkeypatch):
    state = {"requests": [], "content": ' {"type":"final","answer":"ok"} ', "error": None,
             "status": 200, "omit": False}
    def send(client, request, **kwargs):
        state["requests"].append(request)
        assert len(state["requests"]) <= 5, "Retries must be bounded"
        if state["error"]:
            raise state["error"]("CONTROLLED_TRANSPORT_FAILURE", request=request)
        if state["status"] != 200:
            payload = {"error": {"message": "CONTROLLED_AUTH_FAILURE", "type": "authentication_error"}}
        elif state["omit"]:
            payload = {"id": "mock", "object": "chat.completion"}
        else:
            payload = {"id": "mock", "object": "chat.completion", "created": 1, "model": "fixture-model",
                       "choices": [{"index": 0, "finish_reason": "stop", "message": {
                           "role": "assistant", "content": state["content"],
                           "reasoning_content": "PRIVATE_REASONING_SENTINEL"}}],
                       "message": {"role": "assistant", "content": state["content"]}, "done": True}
        return httpx.Response(state["status"], json=payload, request=request)
    monkeypatch.setattr(httpx.Client, "send", send)
    return state


@pytest.mark.parametrize("backend", ["deepseek", "ollama"])
def test_generate_extracts_body_and_uses_explicit_configuration(backend, transport):
    model = make_model(backend, "fixture-model", "http://provider.invalid/custom/v1",
                       api_key="FAKE_TEST_KEY", timeout=1.25)
    messages = [{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "q"}]
    output = model.generate(messages)
    assert isinstance(output, str)
    assert output.strip() == transport["content"].strip()
    assert "PRIVATE_REASONING_SENTINEL" not in output
    assert len(transport["requests"]) == 1
    request = transport["requests"][0]
    assert str(request.url).startswith("http://provider.invalid/custom/")
    body = json.loads(request.content)
    assert body["model"] == "fixture-model"
    assert body["messages"] == messages
    timeout = request.extensions.get("timeout", {})
    # HTTPX exposes configured per-phase limits on the outgoing request.
    assert timeout and all(value is not None and 0 < value <= 1.25 for value in timeout.values())


@pytest.mark.parametrize("backend", ["deepseek", "ollama"])
@pytest.mark.parametrize("content,omit", [(None, False), ("", False), (" \n\t", False), ("ok", True)])
def test_missing_or_empty_response_is_exception(backend, content, omit, transport):
    transport.update(content=content, omit=omit)
    model = make_model(backend, "fixture-model", "http://provider.invalid/v1", "FAKE", 1)
    with pytest.raises(Exception) as caught:
        model.generate([{"role": "user", "content": "q"}])
    assert not isinstance(caught.value, NotImplementedError)


@pytest.mark.parametrize("backend", ["deepseek", "ollama"])
@pytest.mark.parametrize("kind", ["timeout", "authentication"])
def test_transport_failures_are_controlled_and_bounded(backend, kind, transport):
    if kind == "timeout":
        transport["error"] = httpx.ReadTimeout
    else:
        transport["status"] = 401
    model = make_model(backend, "fixture-model", "http://provider.invalid/v1", "FAKE", .01)
    with pytest.raises(Exception) as caught:
        model.generate([{"role": "user", "content": "q"}])
    assert not isinstance(caught.value, NotImplementedError)
    assert 1 <= len(transport["requests"]) <= 5


@pytest.mark.parametrize("backend,model_id,base_url,timeout", [
    ("unknown", "id", "http://provider.invalid/v1", 1),
    ("deepseek", "", "http://provider.invalid/v1", 1),
    ("ollama", "id", "", 1),
    ("ollama", "id", "http://provider.invalid/v1", 0),
    ("ollama", "id", "http://provider.invalid/v1", float("inf")),
])
def test_invalid_model_configuration(backend, model_id, base_url, timeout):
    with pytest.raises(Exception) as caught:
        make_model(backend, model_id, base_url, "FAKE", timeout)
    assert not isinstance(caught.value, NotImplementedError)
