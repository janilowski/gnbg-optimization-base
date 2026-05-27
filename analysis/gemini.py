from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    from .throwaway_generation_core import InvalidApiKeyError
except ImportError:  # pragma: no cover - supports direct script execution
    from throwaway_generation_core import InvalidApiKeyError


DEFAULT_MODEL = "gemini-flash-latest"
ENV_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def response_text(body: dict[str, Any]) -> str:
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"No candidates in Gemini response: {body}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts)
    if not text.strip():
        raise RuntimeError(f"Empty text in Gemini response: {body}")
    return text


def generate_with_gemini(
    *,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    retries: int,
) -> str:
    encoded_model = urllib.parse.quote(model, safe="")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{encoded_model}:generateContent"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": 0.95,
            "maxOutputTokens": max_output_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            return response_text(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if "API_KEY_INVALID" in detail or "API key not valid" in detail:
                raise InvalidApiKeyError(
                    "Gemini rejected the API key. Check that the key in .env is "
                    "from Google AI Studio and has access to the Gemini API."
                ) from exc
            last_error = RuntimeError(f"HTTP {exc.code}: {detail[:1000]}")
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # pragma: no cover - depends on network state
            last_error = exc

        if attempt < retries:
            time.sleep(min(30.0, 2.0**attempt))

    raise RuntimeError(f"Gemini request failed: {last_error}")
