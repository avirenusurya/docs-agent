"""OpenRouter chat backend.

OpenRouter speaks the OpenAI chat-completions format, so the same client works
for any model id it routes to (the free ones included). Swap `llm_model` in
settings to change models without touching code.
"""
from __future__ import annotations

import httpx

from app.config import Settings


class OpenRouterLLM:
    def __init__(self, settings: Settings):
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to backend/.env "
                "(get a free key at openrouter.ai)."
            )
        self.model = settings.llm_model
        self.default_temperature = settings.llm_temperature
        self.default_max_tokens = settings.llm_max_tokens
        self._client = httpx.Client(
            base_url=settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                # Optional attribution headers OpenRouter recommends.
                "HTTP-Referer": "https://github.com/avirenusurya/docs-agent",
                "X-Title": "docs-agent",
            },
            timeout=60.0,
        )

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
        }
        resp = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""
