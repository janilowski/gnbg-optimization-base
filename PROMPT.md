Improve `candidate.py` for the GNBG benchmark.

Constraints:
- Keep the `Algorithm` class name.
- Keep the public interface unchanged.
- Do not edit the harness unless a harness bug blocks evaluation.
- Stay within the evaluation budget.
- The objective is minimization.
- Prefer a robust population-based or multi-start local-search style method over pure random search.

Process:
1. Read `AGENTS.md` and `candidate.py`.
2. Implement one meaningful improvement.
3. Run `python run_candidate.py --profile quick`.
4. If the quick run passes, run `python run_candidate.py --profile search`.
5. Stop after logging results.

In your final message, include:
- the algorithm idea in one sentence
- the quick/search scores
- any assumptions or caveats
