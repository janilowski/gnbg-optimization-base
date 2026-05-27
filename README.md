# GNBG for agent-generated optimization algorithms

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

You need the GNBG instance files at:

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

Competition run:

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

## Generating large amounts of working candidates

Candidate generators ask an LLM for complete `Algorithm` modules,
run the generated file through `run_candidate.py`, and keep only candidates that
pass validation.

For direct Gemini API usage, set a Google AI Studio key in `.env`:

```text
GEMINI_API_KEY=...
```

Then run:

```bash
uv run python3 analysis/generate_throwaway_candidates.py \
  --count 100 \
  --model gemini-flash-latest \
  --profile quick
```

This path talks to Gemini directly and is the intended option for free-quota
generations or to avoid paying the OpenRouter routing fee. Accepted candidates are
written to `candidates/throwaways` by default.

For OpenRouter, set:

```text
OPENROUTER_API_KEY=...
```

Then run:

```bash
uv run python3 analysis/generate_throwaway_candidates_openrouter.py \
  --count 100 \
  --model openai/gpt-5.4-nano \
  --profile quick
```

Accepted candidates are written under `candidates/throwaways/<model-name>`, with the model
name sanitized for filenames.

Common options:

- `--max-attempts N` to cap failed generations.
- `--keep-rejected` to save rejected files for inspection.
- `--hint` to rotate simple search-strategy hints in the prompt.
- `--dry-run-prompt` to print the prompt without calling the API.

## Classifying generated algorithms

LLM-generated candidate files under `candidates/` can be clustered with BERTopic
without adding those dependencies to the default benchmark environment:

```bash
uv sync --group bertopic
uv run --group bertopic python3 analysis/topic_candidates.py
```

The script extracts the structured analysis notes from candidate files and uses
BERTopic for topic labels and per-candidate assignments. By default it uses
`Snowflake/snowflake-arctic-embed-s`; pass `--embedding-model` to use a different
local or cached model.

For embedding models that require custom Hugging Face model code, such as
`nomic-ai/CodeRankEmbed`, pass `--trust-remote-code` explicitly:

```bash
uv run --group bertopic python3 analysis/topic_candidates.py \
  --skip-local-candidates \
  --source-root candidates/throwaways \
  --source-glob '**/*.py' \
  --embedding-model nomic-ai/CodeRankEmbed \
  --trust-remote-code \
  --out-dir results/bertopic_coderank_throwaways
```

Outputs are written to:

```text
results/bertopic_minimal/candidate_topics.csv
results/bertopic_minimal/topic_keywords.csv
```

External experiment folders can be analyzed without copying them into
`candidates/` first:

```bash
uv run --group bertopic python3 analysis/topic_candidates.py \
  --skip-local-candidates \
  --source-root ~/Documents/code/LLaMEA \
  --source-glob 'exp-*/code/*.py' \
  --out-dir results/bertopic_llamea
```

## Generating an LLM-designed EA submission pack

Run the full 31-run evaluation and export the required `.dat` files in one command.
The final profile uses the competition budgets: 500,000 FEs for `f1`-`f15`
and 1,000,000 FEs for `f16`-`f24`.

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

Each `.dat` file contains **31 rows** and **2 whitespace-separated columns** — no header line:

| Column | Meaning |
|--------|---------|
| 1 | `abs(f_best − f*)` — absolute error at end of run |
| 2 | First FE where error ≤ SUBMISSION_THRESHOLD; equals the run budget if never reached |
