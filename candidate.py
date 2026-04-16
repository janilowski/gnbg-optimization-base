from __future__ import annotations

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
