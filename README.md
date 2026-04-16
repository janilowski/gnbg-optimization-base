# Minimal GNBG repo for Codex

This repo is the smallest practical setup for evaluating **Codex-generated algorithms** on the GNBG benchmark through IOH.

The idea is simple:
1. Codex edits `candidate.py`.
2. The harness evaluates that file on GNBG.
3. Each run is logged to `results/runs.jsonl`.

This is intentionally much simpler than an outer evolutionary pipeline like LLaMEA.

## Files

- `candidate.py` — the single algorithm file Codex should improve
- `gnbg_harness.py` — IOH/GNBG loading, adapter, scoring, evaluation profiles
- `run_candidate.py` — run the benchmark and append a log row
- `AGENTS.md` — repo-local guidance for Codex
- `PROMPT.md` — a ready prompt for Codex

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
python run_candidate.py --profile quick
```

A slightly more informative run:

```bash
python run_candidate.py --profile search
```

Official-style heavier run:

```bash
python run_candidate.py --profile final
```

## Using Codex interactively

Inside the repo:

```bash
codex
```

Then paste the contents of `PROMPT.md`.

## Using Codex non-interactively

```bash
codex exec --full-auto --sandbox workspace-write "$(cat PROMPT.md)"
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
- per-case results
- first error, if any

## Important note on scoring

This repo includes a lightweight AOCC/AOC-style scorer so that the loop is self-contained.
For **official competition results**, replace it with your exact `misc.py` helpers from the larger pipeline.
