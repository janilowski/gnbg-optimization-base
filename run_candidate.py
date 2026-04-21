from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from gnbg_harness import evaluate_candidate, export_submission, write_json


def sha256_file(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def module_source_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"Could not resolve source path for module {module_name!r}")
    return Path(spec.origin)


def append_jsonl(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _validate_submission(out_dir: Path) -> dict:
    """Check generated .dat files for GNBG-III format compliance.

    Returns a dict with counts and any detected issues.
    """
    dat_files = sorted(out_dir.glob("f*.dat"))
    issues: list[str] = []
    row_counts: dict[str, int] = {}

    for p in dat_files:
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        row_counts[p.name] = len(lines)
        for i, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 2:
                issues.append(f"{p.name} row {i}: expected 2 columns, got {len(parts)}")
                continue
            try:
                float(parts[0])
                int(float(parts[1]))
            except ValueError:
                issues.append(f"{p.name} row {i}: non-numeric value(s): {line!r}")

    return {
        "n_files": len(dat_files),
        "row_counts": row_counts,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate algorithm on GNBG-III and optionally export submission files."
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "search", "hard", "timing", "final"],
        default="quick",
        help="Evaluation profile (default: quick). Use 'final' for a GNBG-III compliant run.",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--budget-scale", type=float, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument(
        "--with-anchors", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--module", default="candidate")
    parser.add_argument("--class-name", default="Algorithm")
    parser.add_argument("--out", default="results/latest.json")
    parser.add_argument("--log", default="results/runs.jsonl")
    parser.add_argument(
        "--export-submission",
        action="store_true",
        default=False,
        help=(
            "After evaluation, write 24 GNBG-III compliant .dat files under "
            "--submission-dir.  Each file has 30 rows and 2 columns: "
            "absolute_error and fes_to_threshold.  Recommended with --profile final."
        ),
    )
    parser.add_argument(
        "--submission-dir",
        default="results/submission",
        help="Output directory for submission .dat files (default: results/submission).",
    )
    args = parser.parse_args()

    summary = evaluate_candidate(
        module_name=args.module,
        class_name=args.class_name,
        profile=args.profile,
        workers=args.workers,
        budget_scale=args.budget_scale,
        reps=args.reps,
        seed_base=args.seed_base,
        with_anchors=args.with_anchors,
    )

    run_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": sha256_file(module_source_path(args.module)),
        **summary,
    }

    write_json(args.out, run_record)
    append_jsonl(args.log, run_record)

    print(
        json.dumps(
            {
                "profile": run_record["profile"],
                "score_mean": run_record["score_mean"],
                "score_std": run_record["score_std"],
                "score_median": run_record["score_median"],
                "score_trimmed_mean": run_record["score_trimmed_mean"],
                "delta_vs_random_mean": run_record["delta_vs_random_mean"],
                "delta_vs_local_mean": run_record["delta_vs_local_mean"],
                "gap_mean": run_record["gap_mean"],
                "gap_std": run_record["gap_std"],
                "failures": run_record["failures"],
                "candidate_sha256": run_record["candidate_sha256"][:12],
                "out": args.out,
                "log": args.log,
            },
            indent=2,
        )
    )

    if run_record.get("first_error"):
        print("\nFIRST ERROR:")
        print(run_record["first_error"])

    if args.export_submission:
        sub_dir = Path(args.submission_dir)
        export_submission(run_record["results"], out_dir=sub_dir)

        validation = _validate_submission(sub_dir)
        print("\n--- Submission export ---")
        print(f"Output directory : {sub_dir.resolve()}")
        print(f"Files written    : {validation['n_files']}")

        # Show a compact per-file row count table.
        all_counts = list(validation["row_counts"].values())
        if all_counts:
            min_rows = min(all_counts)
            max_rows = max(all_counts)
            if min_rows == max_rows:
                print(f"Rows per file    : {min_rows} (uniform)")
            else:
                print(
                    f"Rows per file    : {min_rows}–{max_rows} (WARNING: not uniform!)"
                )
                for fname, cnt in sorted(validation["row_counts"].items()):
                    if cnt != max_rows:
                        print(f"  {fname}: {cnt} rows")

        if validation["issues"]:
            print(f"Format issues    : {len(validation['issues'])}")
            for issue in validation["issues"][:10]:
                print(f"  ! {issue}")
        else:
            print("Format check     : OK (2 numeric columns, no header)")

        # Print a snippet of the first .dat file as a sanity check.
        dat_files = sorted(sub_dir.glob("f*.dat"))
        if dat_files:
            first = dat_files[0]
            lines = first.read_text(encoding="utf-8").splitlines()
            preview = lines[:3]
            print(f"\nPreview of {first.name}:")
            for ln in preview:
                print(f"  {ln}")
            if len(lines) > 3:
                print(f"  ... ({len(lines)} rows total)")


if __name__ == "__main__":
    main()
