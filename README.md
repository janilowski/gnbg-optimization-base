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
- per-case results
- first error, if any

