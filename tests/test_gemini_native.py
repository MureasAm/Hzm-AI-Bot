import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from src.plugins.chatbot.config import _ChainCompletions, extract_chat_content
from src.plugins.chatbot.gemini_native import GeminiNativeClient


MODEL = "gemini-3.1-flash-lite"
BASE = "https://relay.invalid/v1beta"
FAKE_KEY = "fake-native-test-key"
REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "在吗？"}]}


@pytest.fixture
async def make_native(monkeypatch):
    clients = []
    client_type = httpx.AsyncClient

    def make(handler, *, base_url=BASE):
        def with_mock_transport(*args, **kwargs):
            return client_type(*args, transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", with_mock_transport)
        client = GeminiNativeClient(base_url, FAKE_KEY, timeout=30)
        clients.append(client)
        return client

    yield make
    for client in clients:
        await client.aclose()


def candidate(*parts, finish="STOP"):
    return {"content": {"parts": list(parts)}, "finishReason": finish}


async def test_posts_native_request_with_system_instructions_and_conversation(make_native):
    def handle(request):
        assert request.method == "POST"
        assert str(request.url) == f"{BASE}/models/{MODEL}:generateContent"
        assert request.headers["x-goog-api-key"] == FAKE_KEY
        assert "authorization" not in request.headers
        assert FAKE_KEY.encode() not in request.content
        assert request.extensions["timeout"]["read"] == 6
        payload = json.loads(request.content)
        assert payload["systemInstruction"]["parts"] == [{"text": "像 QQ 私聊一样说话。\n\n按事实接话。"}]
        assert payload["contents"] == [
            {"role": "user", "parts": [{"text": "你之前提过车票。"}, {"text": "后来呢？"}]},
            {"role": "model", "parts": [{"text": "找到了。"}]},
            {"role": "user", "parts": [{"text": "是在哪儿？"}]},
        ]
        assert payload["generationConfig"] == {"maxOutputTokens": 240, "temperature": 0.9}
        assert set(payload) == {"systemInstruction", "contents", "generationConfig"}
        return httpx.Response(200, json={"candidates": [candidate({"text": "抽屉里。"})]})

    client = make_native(handle, base_url=f"{BASE}/")
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "像 QQ 私聊一样说话。"},
            {"role": "user", "content": "你之前提过车票。"},
            {"role": "user", "content": "后来呢？"},
            {"role": "assistant", "content": "找到了。"},
            {"role": "system", "content": "按事实接话。"},
            {"role": "user", "content": "是在哪儿？"},
        ],
        temperature=0.9,
        max_tokens=240,
        frequency_penalty=0.3,
        reasoning_effort="none",
        extra_body={"thinking": {"type": "disabled"}, "generation_config": {"thinking_level": "low"}},
        timeout=6,
    )
    assert extract_chat_content(response) == "抽屉里。"


async def test_json_response_format_uses_native_mime_type(make_native):
    def handle(request):
        payload = json.loads(request.content)
        assert payload["generationConfig"] == {"responseMimeType": "application/json"}
        assert not payload.get("systemInstruction")
        return httpx.Response(200, json={"candidates": [candidate({"text": '{"reply":"在。"}'})]})

    response = await make_native(handle).chat.completions.create(**REQUEST, response_format={"type": "json_object"})
    assert json.loads(extract_chat_content(response)) == {"reply": "在。"}


async def test_returns_only_visible_parts_with_model_and_usage_metadata(make_native):
    def handle(_request):
        return httpx.Response(200, json={
            "candidates": [candidate({"text": "不可见分析", "thought": True}, {"text": "在。"}, {"text": "说吧。"})],
            "modelVersion": "gemini-3.1-flash-lite-preview",
            "usageMetadata": {
                "promptTokenCount": 48,
                "cachedContentTokenCount": 12,
                "candidatesTokenCount": 7,
                "thoughtsTokenCount": 11,
                "totalTokenCount": 66,
            },
        })

    response = await make_native(handle).chat.completions.create(**REQUEST)
    assert extract_chat_content(response) == "在。说吧。"
    assert response.choices[0].finish_reason == "stop"
    assert response.model == "gemini-3.1-flash-lite-preview"
    assert response.usage.prompt_tokens == 48
    assert response.usage.completion_tokens == 18
    assert response.usage.total_tokens == 66
    assert response.usage.prompt_tokens_details.cached_tokens == 12
    assert response.usage.completion_tokens_details.reasoning_tokens == 11


async def test_max_tokens_keeps_partial_text_with_length_finish(make_native):
    def handle(_request):
        return httpx.Response(200, json={
            "candidates": [candidate({"text": "只生成了一部分"}, finish="MAX_TOKENS")],
        })

    response = await make_native(handle).chat.completions.create(**REQUEST)
    assert response.model == MODEL
    assert response.choices[0].finish_reason == "length"
    assert extract_chat_content(response) == "只生成了一部分"


@pytest.mark.parametrize("payload", [
    {"promptFeedback": {"blockReason": "SAFETY"}},
    {"candidates": [candidate({"text": "已屏蔽文本"}, finish="SAFETY")]},
    {"candidates": [candidate({"text": "单独分析", "thought": True}, finish="MAX_TOKENS")]},
    {"candidates": []},
    {"candidates": "坏形状"},
])
async def test_unusable_native_response_uses_existing_fallback(make_native, payload):
    def handle(_request):
        return httpx.Response(200, json=payload)

    client = make_native(handle)
    fallback_reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="备用回复"))])
    complete = AsyncMock(return_value=fallback_reply)
    fallback = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    chain = _ChainCompletions([(client, None, "native"), (fallback, "fallback-model", "fallback")])

    response = await chain.create(**REQUEST)
    assert extract_chat_content(response) == "备用回复"
    complete.assert_awaited_once_with(**{**REQUEST, "model": "fallback-model"})


@pytest.mark.parametrize(("status", "error_type"), [
    (400, BadRequestError),
    (401, AuthenticationError),
    (403, PermissionDeniedError),
    (404, NotFoundError),
    (429, RateLimitError),
    (503, InternalServerError),
])
async def test_native_http_errors_keep_existing_chain_exception_types(make_native, status, error_type):
    def handle(_request):
        return httpx.Response(status, json={"error": {"message": "模拟上游错误"}})

    with pytest.raises(error_type) as error:
        await make_native(handle).chat.completions.create(**REQUEST)
    assert error.value.status_code == status
    assert error.value.request.method == "POST"


@pytest.mark.parametrize(("httpx_error", "error_type"), [
    (httpx.ReadTimeout, APITimeoutError),
    (httpx.ConnectError, APIConnectionError),
])
async def test_native_network_errors_keep_existing_chain_exception_types(make_native, httpx_error, error_type):
    def handle(request):
        raise httpx_error("模拟网络错误", request=request)

    with pytest.raises(error_type):
        await make_native(handle).chat.completions.create(**REQUEST)
