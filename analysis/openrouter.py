from __future__ import annotations

import time

try:
    from .throwaway_generation_core import InvalidApiKeyError
except ImportError:  # pragma: no cover - supports direct script execution
    from throwaway_generation_core import InvalidApiKeyError


DEFAULT_MODEL = "openai/gpt-5.4-nano"
ENV_KEYS = ("OPENROUTER_API_KEY",)


class OpenRouterLLM:
    """Small OpenRouter chat-completions client built on the OpenAI SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.8,
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        import openai  # pyright: ignore[reportMissingImports]

        default_headers = {}
        if site_url is not None:
            default_headers["HTTP-Referer"] = site_url
        if app_name is not None:
            default_headers["X-OpenRouter-Title"] = app_name

        self._openai = openai
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=default_headers,
        )

    def query(
        self,
        session_messages: list[dict[str, str]],
        max_retries: int = 5,
        default_delay: int = 10,
        max_output_tokens: int | None = None,
    ) -> str:
        openai = self._openai
        attempt = 0
        while True:
            try:
                kwargs = {
                    "model": self.model,
                    "messages": session_messages,
                    "temperature": self.temperature,
                }
                if max_output_tokens is not None:
                    kwargs["max_tokens"] = max_output_tokens
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except openai.AuthenticationError as err:
                raise InvalidApiKeyError(
                    "OpenRouter rejected the API key. Check OPENROUTER_API_KEY "
                    "in .env or the environment."
                ) from err
            except openai.RateLimitError as err:
                attempt += 1
                if attempt > max_retries:
                    raise

                retry_after = None
                if getattr(err, "response", None) is not None:
                    retry_after = err.response.headers.get("Retry-After")

                wait = int(retry_after) if retry_after else default_delay * attempt
                time.sleep(wait)
            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.APIError,
            ):
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(default_delay * attempt)


OPENROUTER_LLM = OpenRouterLLM
OpenRouter_LLM = OpenRouterLLM


def generate_with_openrouter(
    *,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    retries: int,
    llm_cls: type[OpenRouterLLM] = OpenRouterLLM,
) -> str:
    llm = llm_cls(api_key=api_key, model=model, temperature=temperature)
    return llm.query(
        [{"role": "user", "content": prompt}],
        max_retries=retries,
        max_output_tokens=max_output_tokens,
    )
