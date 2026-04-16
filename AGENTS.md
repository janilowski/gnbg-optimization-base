# GNBG repository instructions for Codex

Goal: improve `candidate.py` for the GNBG benchmark.

Rules:
- Edit `candidate.py` only, unless a benchmark bug makes that impossible.
- Keep the public interface unchanged:
  - class name: `Algorithm`
  - methods: `__init__(self, budget, dim)` and `__call__(self, func)`
- Never exceed the function evaluation budget.
- The objective is minimization.
- Read bounds from either:
  - `func.lower` / `func.upper`
  - or `func.bounds.lb` / `func.bounds.ub`
- Prefer robust heuristics over flashy but brittle ideas.
- Do not add heavy dependencies.

Validation commands:
- `python3 run_candidate.py --profile quick`
- if that works, optionally `python3 run_candidate.py --profile search`

What to optimize for:
- First: correctness and budget safety.
- Second: better `score_mean`.
- Third: simple, readable code.

When finished:
- report the final score from the last run
- summarize the core search idea in 1-2 lines
