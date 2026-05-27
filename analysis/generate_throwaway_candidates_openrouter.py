from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .openrouter import (
        DEFAULT_MODEL,
        ENV_KEYS,
        generate_with_openrouter,
    )
    from .throwaway_generation_core import (
        api_key_from_env as _api_key_from_env,
        build_prompt,
        file_prefix_for_model,
        load_dotenv,
        run_generation_loop,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from openrouter import (
        DEFAULT_MODEL,
        ENV_KEYS,
        generate_with_openrouter,
    )
    from throwaway_generation_core import (
        api_key_from_env as _api_key_from_env,
        build_prompt,
        file_prefix_for_model,
        load_dotenv,
        run_generation_loop,
    )


def api_key_from_env() -> tuple[str, str]:
    return _api_key_from_env(ENV_KEYS)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(
        description=(
            "Generate OpenRouter candidate algorithms, validate them on the quick "
            "profile, and save passing candidates for BERTopic analysis."
        )
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--profile", default="quick")
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--with-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass through to run_candidate.py during validation.",
    )
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--keep-rejected",
        action="store_true",
        help="Keep rejected candidates under candidates/throwaways/rejected.",
    )
    parser.add_argument(
        "--dry-run-prompt",
        action="store_true",
        help="Print one prompt and exit without calling the API.",
    )
    parser.add_argument("--hint", action="store_true")
    args = parser.parse_args()

    if args.dry_run_prompt:
        print(build_prompt(0, args.hint))
        return

    api_key_name, api_key = api_key_from_env()
    file_prefix = file_prefix_for_model(args.model)
    out_dir = (repo_root / "candidates/throwaways" / file_prefix).resolve()

    def generate_text(prompt: str) -> str:
        return generate_with_openrouter(
            api_key=api_key,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            retries=args.retries,
        )

    run_generation_loop(
        repo_root=repo_root,
        provider_name="OpenRouter",
        api_key_name=api_key_name,
        api_key=api_key,
        args=args,
        out_dir=out_dir,
        output_prefix=file_prefix,
        generate_text=generate_text,
    )


if __name__ == "__main__":
    main()
