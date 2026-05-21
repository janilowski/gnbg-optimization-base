from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gemini-flash-latest"
# DEFAULT_MODEL = "gemma-4-26b-a4b-it"
ENV_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "dataclasses",
    "heapq",
    "itertools",
    "math",
    "numpy",
    "random",
    "statistics",
    "typing",
}
BLOCKED_CALLS = {"compile", "eval", "exec", "input", "open", "__import__"}

VARIATION_HINTS = [
    "multi-start local search with adaptive Gaussian perturbations",
    "small population search with differential-style trial points",
    "coordinate-wise probing around several incumbents",
    "annealed random walk with restarts",
    "elite archive sampling with occasional global refreshes",
    "pattern search using shrinking box-scaled steps",
    "opposition-style initialization followed by local improvement",
    "rank-weighted sampling around the best few points",
    "simple covariance-inspired sampling without matrix-heavy machinery",
    "mixed uniform exploration and incumbent-centered perturbation",
    "success-rate adapted mutation radii",
    "restart hill climbing with coordinate masks",
]


@dataclass
class ValidationResult:
    ok: bool
    summary: dict[str, Any] | None
    stdout_tail: str
    stderr_tail: str
    error: str | None = None


class InvalidApiKeyError(RuntimeError):
    pass


def clean_env_value(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    value = re.split(r"\s+#", value, maxsplit=1)[0]
    return value.strip()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = clean_env_value(value)
        if key and key not in os.environ:
            os.environ[key] = value


def api_key_from_env() -> tuple[str, str]:
    for key in ENV_KEYS:
        value = clean_env_value(os.environ.get(key, ""))
        if value:
            return key, value
    joined = " or ".join(ENV_KEYS)
    raise SystemExit(f"Set {joined} in .env or the environment.")


def masked_key(value: str) -> str:
    if len(value) <= 8:
        return "<set>"
    return f"{value[:4]}...{value[-4:]}"


def build_prompt(candidate_index: int, include_hint: bool = False) -> str:
    hint = ""
    if include_hint:
        hint = f"\nHint: {VARIATION_HINTS[candidate_index % len(VARIATION_HINTS)]}.\n"
    return (
        """Generate one complete Python module for a GNBG black-box
minimization benchmark.

Return only Python code. Do not wrap it in Markdown.

Public interface requirements:
- Define class Algorithm.
- Define Algorithm.__init__(self, budget, dim).
- Define Algorithm.__call__(self, func), returning (best_x, best_y).
- The objective is minimization.
- Never exceed the provided evaluation budget.
- Read bounds from either func.lower / func.upper or func.bounds.lb / func.bounds.ub.
- Use only the Python standard library and numpy.
- Avoid file IO, subprocesses, network calls, multiprocessing, threads, and
  top-level execution.

Candidate direction:
- Keep the implementation compact, readable, and robust across dimensions.
- Include enough comments to make the search behavior understandable.
- Use randomness normally; the harness sets the numpy seed before each run.
"""
        + hint
        + """
Include this structured analysis note near the top of the file, before the
implementation.
Fill it in with plain language that accurately describes the code you wrote.

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: ...
# Search state: ...
# Candidate generation: ...
# Selection and replacement: ...
# Adaptation: ...
# Exploration mechanisms: ...
# Exploitation mechanisms: ...
# Boundary handling: ...
# Budget strategy: ...
# Closest known influences: ...
# Novelty or unusual aspects: ...
# Failure modes: ...
# ALGORITHM_ANALYSIS_NOTE_END
"""
    )


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


def response_text(body: dict[str, Any]) -> str:
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"No candidates in Gemini response: {body}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts)
    if not text.strip():
        raise RuntimeError(f"Empty text in Gemini response: {body}")
    return text


def extract_python(text: str) -> str:
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    return text.strip() + "\n"


def static_errors(source: str) -> list[str]:
    errors: list[str] = []
    if "ALGORITHM_ANALYSIS_NOTE_BEGIN" not in source:
        errors.append("missing ALGORITHM_ANALYSIS_NOTE_BEGIN")
    if "ALGORITHM_ANALYSIS_NOTE_END" not in source:
        errors.append("missing ALGORITHM_ANALYSIS_NOTE_END")

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    algorithm_cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Algorithm":
            algorithm_cls = node
            break
    if algorithm_cls is None:
        errors.append("missing class Algorithm")
    else:
        method_names = {
            node.name
            for node in algorithm_cls.body
            if isinstance(node, ast.FunctionDef)
        }
        if "__init__" not in method_names:
            errors.append("missing Algorithm.__init__")
        if "__call__" not in method_names:
            errors.append("missing Algorithm.__call__")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                errors.append(f"blocked import: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                errors.append(f"blocked call: {node.func.id}")

    return errors


def module_name_for(path: Path, repo_root: Path) -> str:
    rel = path.resolve().relative_to(repo_root.resolve()).with_suffix("")
    return ".".join(rel.parts)


def tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def completed_text(text: str | bytes | None) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return text


def validate_candidate(
    *,
    candidate_path: Path,
    repo_root: Path,
    profile: str,
    seed_base: int,
    workers: int | None,
    with_anchors: bool,
    timeout_s: int,
    validation_dir: Path,
) -> ValidationResult:
    module_name = module_name_for(candidate_path, repo_root)
    validation_dir.mkdir(parents=True, exist_ok=True)
    out_path = validation_dir / f"{candidate_path.stem}.json"
    log_path = validation_dir / "runs.jsonl"

    cmd = [
        sys.executable,
        "run_candidate.py",
        "--profile",
        profile,
        "--module",
        module_name,
        "--seed-base",
        str(seed_base),
        "--out",
        str(out_path),
        "--log",
        str(log_path),
    ]
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    cmd.append("--with-anchors" if with_anchors else "--no-with-anchors")

    try:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ValidationResult(
            ok=False,
            summary=None,
            stdout_tail=tail(completed_text(exc.stdout)),
            stderr_tail=tail(completed_text(exc.stderr)),
            error=f"validation timed out after {timeout_s}s",
        )

    summary = None
    if out_path.exists():
        try:
            summary = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = None

    ok = (
        completed.returncode == 0
        and summary is not None
        and int(summary.get("failures") or 0) == 0
    )
    error = None
    if not ok:
        error = f"returncode={completed.returncode}"
        if summary and summary.get("first_error"):
            error += f"; {summary['first_error']}"

    return ValidationResult(
        ok=ok,
        summary=summary,
        stdout_tail=tail(completed.stdout),
        stderr_tail=tail(completed.stderr),
        error=error,
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def next_output_path(out_dir: Path, prefix: str, index: int) -> Path:
    while True:
        path = out_dir / f"{prefix}_{index:03d}.py"
        if not path.exists():
            return path
        index += 1


def count_existing(out_dir: Path, prefix: str) -> int:
    return len(sorted(out_dir.glob(f"{prefix}_*.py")))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(
        description=(
            "Generate Gemini candidate algorithms, validate them on the quick "
            "profile, and save passing candidates for BERTopic analysis."
        )
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default="candidates/throwaways")
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
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = out_dir / "_validation"
    rejected_dir = out_dir / "rejected"
    manifest_path = out_dir / "manifest.jsonl"

    accepted = count_existing(out_dir, args.model)
    attempts = 0
    max_attempts = args.max_attempts or max(args.count * 3, args.count)
    print(f"Starting with {accepted}/{args.count} existing accepted candidates")
    print(f"Output directory: {out_dir}")
    print(f"Gemini API key: {api_key_name}={masked_key(api_key)}")

    while accepted < args.count and attempts < max_attempts:
        attempts += 1
        candidate_index = accepted
        pending_name = f"_pending_{args.model}_{int(time.time())}_{attempts:04d}.py"
        pending_path = out_dir / pending_name
        final_path = next_output_path(out_dir, args.model, candidate_index)

        print(f"[{attempts}] requesting candidate {candidate_index:03d}...")
        record: dict[str, Any] = {
            "attempt": attempts,
            "target_index": candidate_index,
            "model": args.model,
            "profile": args.profile,
            "path": final_path.relative_to(repo_root).as_posix(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        try:
            raw = generate_with_gemini(
                api_key=api_key,
                model=args.model,
                prompt=build_prompt(candidate_index + attempts, args.hint),
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                retries=args.retries,
            )
            source = extract_python(raw)
            errors = static_errors(source)
            if errors:
                record.update({"status": "static_rejected", "errors": errors})
                append_jsonl(manifest_path, record)
                print(f"    static rejected: {'; '.join(errors[:3])}")
                if args.keep_rejected:
                    rejected_dir.mkdir(parents=True, exist_ok=True)
                    (rejected_dir / f"static_rejected_{attempts:04d}.py").write_text(
                        source, encoding="utf-8"
                    )
                continue

            pending_path.write_text(source, encoding="utf-8")
            result = validate_candidate(
                candidate_path=pending_path,
                repo_root=repo_root,
                profile=args.profile,
                seed_base=args.seed_base,
                workers=args.workers,
                with_anchors=args.with_anchors,
                timeout_s=args.timeout_s,
                validation_dir=validation_dir,
            )

            summary = result.summary or {}
            record.update(
                {
                    "status": "accepted" if result.ok else "runtime_rejected",
                    "score_mean": summary.get("score_mean"),
                    "score_median": summary.get("score_median"),
                    "score_trimmed_mean": summary.get("score_trimmed_mean"),
                    "failures": summary.get("failures"),
                    "error": result.error,
                    "stdout_tail": result.stdout_tail,
                    "stderr_tail": result.stderr_tail,
                }
            )

            if result.ok:
                pending_path.replace(final_path)
                accepted += 1
                print(
                    "    accepted "
                    f"{accepted}/{args.count}: {final_path.name} "
                    f"score_mean={summary.get('score_mean')}"
                )
            else:
                print(f"    runtime rejected: {result.error}")
                if args.keep_rejected:
                    rejected_dir.mkdir(parents=True, exist_ok=True)
                    rejected_path = rejected_dir / (
                        f"runtime_rejected_{attempts:04d}.py"
                    )
                    pending_path.replace(rejected_path)
                else:
                    pending_path.unlink(missing_ok=True)

            append_jsonl(manifest_path, record)
        except KeyboardInterrupt:
            pending_path.unlink(missing_ok=True)
            raise
        except InvalidApiKeyError as exc:
            pending_path.unlink(missing_ok=True)
            record.update({"status": "auth_error", "error": str(exc)})
            append_jsonl(manifest_path, record)
            raise SystemExit(str(exc)) from exc
        except Exception as exc:
            pending_path.unlink(missing_ok=True)
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            append_jsonl(manifest_path, record)
            print(f"    error: {type(exc).__name__}: {exc}")

    print(f"Finished with {accepted}/{args.count} accepted candidates.")
    if accepted < args.count:
        raise SystemExit(
            f"Stopped after {attempts} attempts before reaching --count={args.count}."
        )


if __name__ == "__main__":
    main()
