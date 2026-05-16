from __future__ import annotations

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: The method is plain uniform random search over the full bounded domain.
# Search state: It keeps only the best point and best objective value seen so far.
# Candidate generation: Each evaluation samples one independent point uniformly from the box bounds.
# Selection and replacement: A sampled point replaces the incumbent only if it improves the objective.
# Adaptation: There is no adaptation; sampling distribution and behavior stay fixed.
# Exploration mechanisms: Exploration comes from full-domain independent random sampling.
# Exploitation mechanisms: There is no local intensification beyond retaining the incumbent.
# Boundary handling: Candidates are sampled directly inside the bounds, so no repair is needed.
# Budget strategy: The entire budget is spent on independent global samples.
# Closest known influences: Uniform random search baseline.
# Novelty or unusual aspects: None; this is intentionally simple baseline behavior.
# Failure modes: It scales poorly in high dimensions and does not exploit smoothness or repeated structure.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Baseline algorithm that Codex should replace or improve.
    Interface expected by the harness:
      - __init__(self, budget, dim)
      - __call__(self, func)
    """

    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lower = np.asarray(getattr(func, "lower", func.bounds.lb), dtype=float)
        upper = np.asarray(getattr(func, "upper", func.bounds.ub), dtype=float)

        best_x = None
        best_y = float("inf")

        for _ in range(self.budget):
            x = np.random.uniform(lower, upper, size=self.dim)
            y = float(func(x))
            if y < best_y:
                best_x = x.copy()
                best_y = y

        return best_x, best_y
