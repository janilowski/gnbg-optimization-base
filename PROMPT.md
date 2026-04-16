Improve `candidate.py` for the GNBG benchmark.

Constraints:
- Keep the `Algorithm` class name.
- Keep the public interface unchanged.
- Do not edit the harness unless a harness bug blocks evaluation.
- Stay within the evaluation budget.
- The objective is minimization.
- Prefer a robust population-based or multi-start local-search style method over pure random search.
- Favor creativity: combine at least two ideas (e.g., restart policy + local search, or adaptive mutation + elite archive), but keep the code readable.
- Explain your idea in detail in a comment block at the top of the candidate file. First work on the idea, then try implementing it.
- Include a lot of comments in your code that explain what you are doing.
- You may search the internet for papers on black box function optimization.

Process:
1. Read `AGENTS.md` and `candidate.py`.
2. You have two options: Option A (default when applicable): make the smallest possible changes, preserve comments and variable names, and imitate the surrounding code style. In future minimal-change code responses, mark the modified lines clearly. Option B (revamp): offer or use a deeper rewrite only when the benefits clearly outweigh the cost, and justify why.

3. Run `uv run python3 run_candidate.py --profile quick --seed-base 12345 --with-anchors`.
4. If the quick run passes, run `uv run python3 run_candidate.py --profile hard --seed-base 12345 --with-anchors`.
5. Look at the results, especially the mean difference from random.
6. Try to improve why your program performed as it did and try to improve your result. Aim to reach as close as possible to a benchmark score of 1.0;
7. Run the benchmark again and give me the result.

In your final message, include:
- the algorithm idea in one sentence
- hard (and timing if run) scores: mean, median, trimmed mean
- delta vs random and delta vs local anchors
