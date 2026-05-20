from __future__ import annotations

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact restart (1+1) evolution strategy seeded from the box center and then random starts.
# Search state: It keeps the current local point, the global incumbent, an evaluation counter, and a mutable Gaussian step size.
# Candidate generation: New candidates are Gaussian perturbations of the current local point, scaled to the search box.
# Selection and replacement: A candidate replaces the local point only when it improves that local objective value; the global best is updated on every evaluation.
# Adaptation: The mutation radius expands after successful moves and contracts after failed moves, with restarts after long stalls.
# Exploration mechanisms: Exploration comes from random restarts, large early mutation radii, and occasional jumps from the global incumbent.
# Exploitation mechanisms: Greedy local selection and shrinking Gaussian steps concentrate evaluations around promising basins.
# Boundary handling: Every candidate is clipped to the supplied lower and upper bounds before evaluation.
# Budget strategy: All objective calls pass through a single guarded evaluator that stops exactly at the provided budget.
# Closest known influences: Rechenberg-style (1+1)-ES, Luus-Jaakola search, restart hill climbing.
# Novelty or unusual aspects: The first local run starts at the domain center, which is a cheap robust anchor before stochastic restarts take over.
# Failure modes: Axis-free Gaussian steps can be inefficient on narrow rotated valleys, and greedy selection may miss deceptive basin exits.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lower, upper = self._bounds(func)
        span = np.maximum(upper - lower, 1e-12)
        center = lower + 0.5 * span

        evals = 0
        best_x = center.copy()
        best_y = float("inf")

        def project(x):
            return np.minimum(np.maximum(x, lower), upper)

        def evaluate(x):
            nonlocal evals, best_x, best_y
            if evals >= self.budget:
                return None
            x = project(np.asarray(x, dtype=float))
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        if self.budget <= 0:
            return best_x, best_y

        evaluate(center)

        starts_done = 0
        while evals < self.budget:
            x = np.random.uniform(lower, upper, self.dim)

            y = evaluate(x)
            if y is None:
                break

            starts_done += 1
            sigma = 0.20
            fail_count = 0
            stall_limit = 100 + 10 * self.dim

            while evals < self.budget:
                trial = x + np.random.normal(0.0, sigma, self.dim) * span
                ty = evaluate(trial)
                if ty is None:
                    break

                if ty < y:
                    x = project(trial)
                    y = ty
                    sigma *= 1.50
                    fail_count = 0
                else:
                    sigma *= 0.84
                    fail_count += 1

                if sigma < 1e-10 or fail_count >= stall_limit:
                    break

        return best_x, best_y

    def _bounds(self, func):
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        else:
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)

        if lower.size == 1 and self.dim > 1:
            lower = np.full(self.dim, float(lower.item()))
        if upper.size == 1 and self.dim > 1:
            upper = np.full(self.dim, float(upper.item()))
        return lower.reshape(-1), upper.reshape(-1)
