Improve `candidate.py` for the GNBG benchmark.

Constraints:
- Keep the `Algorithm` class name.
- Keep the public interface unchanged.
- Do not edit the harness unless a harness bug blocks evaluation.
- Stay within the evaluation budget.
- The objective is minimization.
- Prefer a robust population-based or multi-start local-search style method over pure random search.
- Favor creativity with guardrails: combine at least two ideas (e.g., restart policy + local search, or adaptive mutation + elite archive), but keep the code readable and budget-safe.

Process:
1. Read `AGENTS.md` and `candidate.py`.
2. Implement one meaningful improvement.
3. Run `python3 run_candidate.py --profile quick --seed-base 12345 --with-anchors`.
4. If the quick run passes, run `python3 run_candidate.py --profile search --seed-base 12345 --with-anchors`.
5. For robustness, run at least one multi-seed pass, for example:
   - `python3 run_candidate.py --profile search --reps 5 --seed-base 2026 --with-anchors`
6. If compute budget allows, run:
   - `python3 run_candidate.py --profile selection --seed-base 2026 --with-anchors`
7. Stop after logging results.

In your final message, include:
- the algorithm idea in one sentence
- quick/search (and selection if run) scores: mean, median, trimmed mean
- delta vs random and delta vs local anchors
- one note on creativity tradeoff (what you tried that could fail)
- any assumptions or caveats
