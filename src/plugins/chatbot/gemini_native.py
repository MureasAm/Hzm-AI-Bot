"""Gemini generateContent adapter for the chatbot's completion chain."""

import re
from types import SimpleNamespace

import httpx
from openai import (
    APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError,
    BadRequestError, ConflictError, InternalServerError, NotFoundError,
    PermissionDeniedError, RateLimitError, UnprocessableEntityError,
)


_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_STATUS_ERRORS = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
}


def _payload(messages, temperature, max_tokens, response_format):
    system_text = []
    contents = []
    for message in messages:
        role = message["role"]
        part = {"text": message["content"]}
        if role == "system":
            system_text.append(part["text"])
            continue
        if role == "assistant":
            role = "model"
        elif role != "user":
            raise ValueError("Unsupported Gemini message role")

        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append(part)
        else:
            contents.append({"role": role, "parts": [part]})

    body = {"contents": contents}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_text)}]}
    generation = {}
    if temperature is not None:
        generation["temperature"] = temperature
    if max_tokens is not None:
        generation["maxOutputTokens"] = max_tokens
    if response_format and response_format.get("type") == "json_object":
        generation["responseMimeType"] = "application/json"
    if generation:
        body["generationConfig"] = generation
    return body


def _token_count(metadata, key):
    count = metadata.get(key)
    return count if type(count) is int and count >= 0 else None


def _usage(metadata):
    if not isinstance(metadata, dict):
        return None
    visible = _token_count(metadata, "candidatesTokenCount")
    reasoning = _token_count(metadata, "thoughtsTokenCount")
    return SimpleNamespace(
        prompt_tokens=_token_count(metadata, "promptTokenCount"),
        completion_tokens=visible + (reasoning or 0) if visible is not None else None,
        total_tokens=_token_count(metadata, "totalTokenCount"),
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=_token_count(metadata, "cachedContentTokenCount"),
        ),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


def _completion(body, requested_model):
    result = SimpleNamespace(model=requested_model, choices=[], usage=None)
    if not isinstance(body, dict):
        return result

    model = body.get("modelVersion")
    if isinstance(model, str) and _MODEL_NAME.fullmatch(model.removeprefix("models/")):
        result.model = model
    result.usage = _usage(body.get("usageMetadata"))

    candidates = body.get("candidates")
    candidate = candidates[0] if isinstance(candidates, list) and candidates else None
    if not isinstance(candidate, dict):
        feedback = body.get("promptFeedback")
        if not isinstance(feedback, dict) or not feedback.get("blockReason"):
            return result
        candidate = {"finishReason": "SAFETY"}

    reason = candidate.get("finishReason")
    finish = {"STOP": "stop", "MAX_TOKENS": "length"}.get(reason) if isinstance(reason, str) else None
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    text = ""
    if reason and finish is None:
        finish = "content_filter"
    elif isinstance(parts, list):
        text = "".join(
            part["text"] for part in parts
            if isinstance(part, dict) and not part.get("thought") and isinstance(part.get("text"), str)
        )

    result.choices.append(SimpleNamespace(
        index=0, finish_reason=finish, message=SimpleNamespace(role="assistant", content=text),
    ))
    return result


class GeminiNativeClient:
    """Expose a Gemini REST endpoint through the chain's chat completion interface."""

    def __init__(self, base_url, api_key, *, timeout):
        url = httpx.URL(base_url)
        if url.scheme != "https" or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("Gemini endpoint must be an HTTPS base URL")
        self._base_url = str(url).rstrip("/")
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout)
        self.chat = SimpleNamespace(completions=self)

    async def aclose(self):
        await self._client.aclose()

    async def create(
        self, *, model, messages, temperature=None, max_tokens=None,
        response_format=None, timeout=httpx.USE_CLIENT_DEFAULT, **_ignored,
    ):
        # The relay's tested native route uses its default thinking settings
        # and does not accept the cross-provider frequency penalty.
        model_id = model.removeprefix("models/")
        if not _MODEL_NAME.fullmatch(model_id):
            raise ValueError("Invalid Gemini model name")
        url = f"{self._base_url}/models/{model_id}:generateContent"
        body = _payload(messages, temperature, max_tokens, response_format)
        try:
            response = await self._client.post(
                url, json=body, headers={"x-goog-api-key": self._api_key}, timeout=timeout,
            )
        except httpx.TimeoutException:
            raise APITimeoutError(request=httpx.Request("POST", url)) from None
        except httpx.RequestError:
            raise APIConnectionError(request=httpx.Request("POST", url)) from None

        if not response.is_success:
            status = response.status_code
            error = _STATUS_ERRORS.get(status, InternalServerError if status >= 500 else APIStatusError)
            # The chain logs errors. Discard the provider body and request headers.
            safe_response = httpx.Response(status, request=httpx.Request("POST", url))
            raise error(f"Gemini upstream HTTP {status}", response=safe_response, body=None)
        try:
            native_response = response.json()
        except ValueError:
            native_response = None
        return _completion(native_response, model)
