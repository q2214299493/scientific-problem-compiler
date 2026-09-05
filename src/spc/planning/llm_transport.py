from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any, Protocol
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class LLMTransport(Protocol):
    model_id: str

    def generate_structured(
        self,
        *,
        system_prompt: str,
        input_payload: dict[str, Any],
        response_schema: dict[str, Any],
        temperature: float,
    ) -> str: ...


class FakeLLMTransport:
    """Deterministic scripted transport for network-free provider tests."""

    def __init__(self, responses: Sequence[str | dict[str, Any]], model_id: str = "fake-model") -> None:
        self.model_id = model_id
        self._responses = tuple(responses)
        self.requests: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def generate_structured(
        self,
        *,
        system_prompt: str,
        input_payload: dict[str, Any],
        response_schema: dict[str, Any],
        temperature: float,
    ) -> str:
        self.requests.append(
            {
                "system_prompt": system_prompt,
                "input_payload": input_payload,
                "response_schema": response_schema,
                "temperature": temperature,
            }
        )
        index = len(self.requests) - 1
        if index >= len(self._responses):
            raise RuntimeError("FakeLLMTransport has no scripted response for this attempt")
        response = self._responses[index]
        return response if isinstance(response, str) else json.dumps(response)


class HTTPJSONLLMTransport:
    """Vendor-neutral HTTP transport with a small JSON request/response contract."""

    def __init__(
        self,
        endpoint: str,
        model_id: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not endpoint.strip() or not model_id.strip():
            raise ValueError("LLM endpoint and model_id must not be blank")
        if urlsplit(endpoint).scheme not in {"http", "https"}:
            raise ValueError("LLM endpoint must use http or https")
        self.endpoint = endpoint
        self.model_id = model_id
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate_structured(
        self,
        *,
        system_prompt: str,
        input_payload: dict[str, Any],
        response_schema: dict[str, Any],
        temperature: float,
    ) -> str:
        payload = json.dumps(
            {
                "model": self.model_id,
                "system_prompt": system_prompt,
                "input": input_payload,
                "response_schema": response_schema,
                "temperature": temperature,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, data=payload, headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict) or not isinstance(result.get("output_text"), str):
            raise ValueError("LLM endpoint must return a JSON object with string output_text")
        return result["output_text"]
