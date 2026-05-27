Improve `candidate.py` for the GNBG benchmark.

Constraints:
- Keep the `Algorithm` class name.
- Keep the public interface unchanged.
- Do not edit the harness unless a harness bug blocks evaluation.
- Stay within the evaluation budget.
- The objective is minimization.
- Prefer robust search behavior over pure random search. Do not assume a specific
  optimizer family is required; population-based, single-trajectory, local,
  surrogate-like, restart-based, coordinate-wise, hybrid, or unusual approaches
  are all acceptable if they are budget-safe and well motivated.
- Favor creativity without forcing the design into familiar buckets. If the
  method is closest to a known family, say so, but do not label it as such unless
  that is genuinely how it works.
- Include the structured analysis note below near the top of `candidate.py`,
  before the implementation. Keep the exact begin/end markers. Fill it in with
  plain language based on what the code actually does, using open-ended terms
  rather than picking from a fixed taxonomy.

```text
ALGORITHM_ANALYSIS_NOTE_BEGIN
Summary: One or two sentences describing the search strategy.
Search state: What information the algorithm keeps between evaluations.
Candidate generation: How new points are proposed.
Selection and replacement: How accepted/improving points affect future search.
Adaptation: What changes over time, if anything.
Exploration mechanisms: How the method avoids getting stuck.
Exploitation mechanisms: How the method intensifies around promising regions.
Boundary handling: How out-of-bounds points are repaired or avoided.
Budget strategy: How the evaluation budget is allocated over phases.
Closest known influences: Known optimizer ideas it resembles, or "none/unclear".
Novelty or unusual aspects: What is distinctive about this implementation.
Failure modes: Landscape types or conditions where it may perform poorly.
ALGORITHM_ANALYSIS_NOTE_END
```

- First work on the idea and the analysis note, then implement it.
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
