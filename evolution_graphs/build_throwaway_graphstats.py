"""Build AST graph statistics for generated throwaway candidate code.

This script adapts ``python_ast_analysis.py`` into a reusable batch extractor for
the model-organized corpus under ``candidates/throwaways``. It writes one CSV row
per Python candidate file and keeps per-file errors in the output instead of
stopping the full corpus run.

Throwaway corpus schema:
    - ``model`` is the canonical grouping column, derived from the parent folder.
    - ``alg_id`` is a pseudo-evolution index: stable filename order within model.
    - ``parent_id`` / ``parent_ids`` are compatibility columns only; this corpus
      has no known lineage, so ``lineage_available`` is always false.
    - ``fitness`` is optional and intentionally blank unless a later merge step
      supplies validation metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import warnings
from pathlib import Path
from typing import Any

from tqdm import tqdm

from python_ast_analysis import process_file


DEFAULT_CORPUS_DIR = Path("candidates/throwaways")
DEFAULT_OUTPUT_PATH = Path("evolution_graphs/results/graphstats_throwaways.csv")
SCHEMA_VERSION = "throwaways_v1"
CORPUS_NAME = "throwaways"

METADATA_FIELDS = [
    "schema_version",
    "corpus",
    "path",
    "model",
    "LLM",
    "exp_dir",
    "filename",
    "candidate_id",
    "alg_id",
    "source_index",
    "sequence_kind",
    "parent_id",
    "parent_ids",
    "lineage_available",
    "fitness",
    "fitness_source",
    "has_fitness",
    "parse_ok",
    "graph_ok",
    "complexity_ok",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AST graph and complexity metrics for generated candidates."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help=f"Root generated-code corpus directory (default: {DEFAULT_CORPUS_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of files to process, useful for smoke tests.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress output.",
    )
    return parser.parse_args()


def extract_source_index(path: Path) -> int | None:
    """Return the trailing numeric id from known generated-candidate filenames."""
    match = re.search(r"_(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def file_sort_key(path: Path) -> tuple[float, str]:
    source_index = extract_source_index(path)
    if source_index is None:
        return (math.inf, path.name)
    return (source_index, path.name)


def iter_candidate_files(corpus_dir: Path) -> list[Path]:
    """Return Python candidate files grouped by model directory in stable order."""
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    files: list[Path] = []
    model_dirs = sorted(path for path in corpus_dir.iterdir() if path.is_dir())
    for model_dir in model_dirs:
        if model_dir.name == "__pycache__":
            continue
        model_files = sorted(model_dir.glob("*.py"), key=file_sort_key)
        files.extend(model_files)
    return files


def normalize_value(value: Any) -> Any:
    """Convert non-finite floats to empty cells so downstream CSV reads are stable."""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def build_row(path: Path, alg_id: int) -> dict[str, Any]:
    model = path.parent.name
    candidate_id = path.stem
    source_index = extract_source_index(path)

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "corpus": CORPUS_NAME,
        "path": display_path(path),
        "model": model,
        # Legacy aliases used by the current visualization script. New code
        # should prefer ``model`` and treat ``exp_dir`` as a grouping alias.
        "LLM": model,
        "exp_dir": model,
        "filename": path.name,
        "candidate_id": candidate_id,
        "alg_id": alg_id,
        "source_index": source_index if source_index is not None else "",
        "sequence_kind": "filename_order_within_model",
        "parent_id": "",
        "parent_ids": "[]",
        "lineage_available": False,
        "fitness": "",
        "fitness_source": "",
        "has_fitness": False,
        "parse_ok": True,
        "graph_ok": True,
        "complexity_ok": True,
        "error": "",
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stats = process_file(str(path), visualize=False)
    except Exception as exc:  # Keep corpus extraction robust across bad candidates.
        row["parse_ok"] = False
        row["graph_ok"] = False
        row["complexity_ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
    else:
        row.update({key: normalize_value(value) for key, value in stats.items()})

    return row


def build_rows(corpus_dir: Path, limit: int | None, quiet: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = iter_candidate_files(corpus_dir)
    if limit is not None:
        files = files[:limit]

    alg_ids_by_model: dict[str, int] = {}
    iterator = files if quiet else tqdm(files, desc="Extracting AST stats")
    for path in iterator:
        model = path.parent.name
        alg_id = alg_ids_by_model.get(model, 0)
        alg_ids_by_model[model] = alg_id + 1
        rows.append(build_row(path, alg_id))

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_fields = sorted(
        {field for row in rows for field in row.keys()} - set(METADATA_FIELDS)
    )
    fieldnames = METADATA_FIELDS + dynamic_fields

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args.corpus_dir, args.limit, args.quiet)
    write_csv(rows, args.output)

    failures = sum(1 for row in rows if not row["parse_ok"])
    print(f"Wrote {len(rows)} rows to {args.output}")
    if failures:
        print(f"Files with extraction errors: {failures}")


if __name__ == "__main__":
    main()
