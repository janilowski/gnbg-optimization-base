# Minimal GNBG repo for agent-generated optimization algorithms

This repo is the smallest practical setup for evaluating **agent-generated algorithms** on the GNBG benchmark through IOH.

The idea is simple:
1. An AI agent edits `candidate.py`.
2. The harness evaluates that file on GNBG.
3. Each run is logged to `results/runs.jsonl`.

## Files

- `candidate.py` — the single algorithm file Codex should improve
- `gnbg_harness.py` — IOH/GNBG loading, adapter, scoring, evaluation profiles
- `run_candidate.py` — run the benchmark and append a log row

## Setup

Using `uv`:

```bash
uv sync
```

Or with plain pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

You still need the GNBG instance files at:

```text
benchmarks/gnbg/official
```

If your instance directory differs, change `GNBG_INSTANCES_FOLDER` in `gnbg_harness.py`.

## Run locally

Quick smoke test:

```bash
python3 run_candidate.py --profile quick
```

A slightly more informative run:

```bash
python3 run_candidate.py --profile search
```

Selection run (better for ranking candidates than quick/search smoke tests):

```bash
python3 run_candidate.py --profile selection
```

Official-style heavier run:

```bash
python3 run_candidate.py --profile final
```

## Logging

Every run appends one JSON object to:

```text
results/runs.jsonl
```

and also writes the latest run to:

```text
results/latest.json
```

Each record includes:
- timestamp
- candidate file hash
- profile
- mean score
- score std
- score median
- score trimmed mean (10% trimming)
- deltas vs random-search and local-search anchors
- per-problem score aggregates
- per-case results
- first error, if any

Useful knobs:

- `--reps N` to override repetitions per problem.
- `--seed-base S` to change the deterministic seed schedule.
- `--with-anchors/--no-anchors` to enable or disable baseline-anchor comparisons.

## Generating a GNBG-III submission pack

Run the full 30 × 24 × 500 000-FE evaluation and export the required `.dat` files in one command:

```bash
uv run python3 run_candidate.py --profile final --no-with-anchors --export-submission
```

> **Note:** `--no-with-anchors` skips the random/local baseline comparisons to halve wall-clock time.
> Omit it if you want the AOC delta metrics in the JSON log.

Output files written to:

```
results/submission/f1.dat
results/submission/f2.dat
...
results/submission/f24.dat
```

Each `.dat` file contains **30 rows** and **2 whitespace-separated columns** — no header line:

| Column | Meaning |
|--------|---------|
| 1 | `abs(f_best − f*)` — absolute error at end of run |
| 2 | First FE where error ≤ 1e-8; equals 500 000 if never reached |

**Threshold note:** The competition lists evaluation targets 1e-1, 1e-3, 1e-5, 1e-8.
Column 2 uses the tightest target (1e-8) as the single "FEs-to-threshold" value per run.
Runs that never reach 1e-8 report the full budget (500 000), following the ERT convention
that failures count as the maximum budget.

To package for submission, zip the output directory:

```bash
zip -r MyAlgorithm_submission.zip results/submission/
```

Then email to `dsmlossf@gmail.com` with your abstract (see competition page for full requirements).

## Profile reference

| Profile | Problems | Reps | FEs/run | Purpose |
|---------|----------|------|---------|---------|
| `quick` | f1–f2 | 1 | 400 | Smoke test during development |
| `search` | f1–f24 | 3 | 20 000 | Algorithm search / ranking |
| `hard` | 6 hard problems | 5 | 200 000 | Stress test |
| `timing` | f1–f24 | 3 | 60 000 | Wall-clock profiling |
| `final` | f1–f24 | 30 | **500 000** | **GNBG-III compliant submission run** |
