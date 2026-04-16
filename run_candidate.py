from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from gnbg_harness import evaluate_candidate, write_json


def sha256_file(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def append_jsonl(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "search", "timing", "final"], default="quick")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--budget-scale", type=float, default=None)
    parser.add_argument("--module", default="candidate")
    parser.add_argument("--class-name", default="Algorithm")
    parser.add_argument("--out", default="results/latest.json")
    parser.add_argument("--log", default="results/runs.jsonl")
    args = parser.parse_args()

    summary = evaluate_candidate(
        module_name=args.module,
        class_name=args.class_name,
        profile=args.profile,
        workers=args.workers,
        budget_scale=args.budget_scale,
    )

    run_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": sha256_file(f"{args.module}.py"),
        **summary,
    }

    write_json(args.out, run_record)
    append_jsonl(args.log, run_record)

    print(json.dumps({
        "profile": run_record["profile"],
        "score_mean": run_record["score_mean"],
        "score_std": run_record["score_std"],
        "failures": run_record["failures"],
        "candidate_sha256": run_record["candidate_sha256"][:12],
        "out": args.out,
        "log": args.log,
    }, indent=2))

    if run_record.get("first_error"):
        print("\nFIRST ERROR:")
        print(run_record["first_error"])


if __name__ == "__main__":
    main()
