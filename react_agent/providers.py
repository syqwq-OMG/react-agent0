"""Model-provider adapters with a shared ``generate`` interface."""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlparse

import httpx


class ProviderError(RuntimeError):
    """Raised when a configured provider cannot return usable text."""


def _validate_configuration(
    model_id: str, base_url: str, api_key: str | None, timeout: float,
) -> tuple[str, str, str | None, float]:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")

    normalized_url = base_url.strip()
    parsed_url = urlparse(normalized_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")

    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive finite number")
    normalized_timeout = float(timeout)
    if not math.isfinite(normalized_timeout) or normalized_timeout <= 0:
        raise ValueError("timeout must be a positive finite number")

    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("api_key must be a non-empty string when provided")

    return model_id.strip(), normalized_url, api_key.strip() if api_key else None, normalized_timeout


class Provider:
    """OpenAI-compatible chat-completions provider."""

    _MAX_ATTEMPTS = 3

    def __init__(
        self, model_id: str, base_url: str, api_key: str, timeout: float = 60.0,
    ) -> None:
        self.model = model_id
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not isinstance(messages, list) or not all(
            isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError("messages must be a list of dictionaries with string role and content")

        response = None
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                response = self.client.post(
                    self.url,
                    json={"model": self.model, "messages": messages},
                )
            except httpx.RequestError as exc:
                if attempt + 1 == self._MAX_ATTEMPTS:
                    raise ProviderError(f"provider request failed: {exc}") from exc
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < self._MAX_ATTEMPTS:
                    continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"provider request failed with HTTP {response.status_code}"
                ) from exc
            break

        try:
            payload: Any = response.json()
        except (TypeError, ValueError) as exc:
            raise ProviderError("provider response was not valid JSON") from exc

        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not choices:
            raise ProviderError("provider response did not contain any choices")

        first_choice = choices[0]
        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("provider response did not contain non-empty assistant content")
        return content.strip()


class DeepseekProvider(Provider):
    def __init__(
        self, model_id: str, base_url: str, api_key: str, timeout: float = 60.0,
    ) -> None:
        super().__init__(model_id, base_url, api_key, timeout)


class OllamaProvider(Provider):
    def __init__(
        self, model_id: str, base_url: str, api_key: str = "ollama", timeout: float = 60.0,
    ) -> None:
        super().__init__(model_id, base_url, api_key, timeout)


def make_model(
    backend: str,
    model_id: str,
    base_url: str,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> Provider:
    """Build a validated DeepSeek or Ollama model adapter."""

    if not isinstance(backend, str) or backend not in {"deepseek", "ollama"}:
        raise ValueError("backend must be either 'deepseek' or 'ollama'")

    model_id, base_url, api_key, timeout = _validate_configuration(
        model_id, base_url, api_key, timeout,
    )
    if backend == "deepseek":
        if api_key is None or api_key == "?":
            raise ValueError("api_key is required for the deepseek backend")
        return DeepseekProvider(model_id, base_url, api_key, timeout)

    return OllamaProvider(model_id, base_url, api_key or "ollama", timeout)
