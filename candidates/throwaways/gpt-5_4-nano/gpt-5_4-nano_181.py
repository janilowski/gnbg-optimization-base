# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization strategy that
# combines (1) a population-based stochastic search for global exploration with
# (2) a simple gradient-free local refinement around the best-so-far point.
# It is designed to work with any black-box objective by using only function
# evaluations, respecting the provided evaluation budget.
#
# Search state: A current best solution (best_x, best_y) is maintained across
# the entire run. Additionally, a small "population" of candidate points is
# sampled around the best point to propose new evaluations while tracking
# which evaluations have been spent.
#
# Candidate generation: Each iteration generates candidates using Gaussian
# perturbations around the best_x. The perturbation scale shrinks over time,
# and candidate directions include both axis-aligned and isotropic noise to
# help in different geometries.
#
# Selection and replacement: Among the newly evaluated candidates, the best one
# replaces the global best if it yields a lower objective value. The remaining
# candidates do not persist beyond the iteration (replace-and-forget), keeping
# the implementation simple and robust.
#
# Adaptation: The perturbation step size adapts based on whether improvement was
# observed. If there is improvement, the step size is mildly reduced (to focus
# near the promising region); if not, the step size is increased up to a cap
# (to escape flat/local traps).
#
# Exploration mechanisms: Early in the budget, larger steps and a larger
# population size promote exploration. The search distribution is initially
# broad and becomes narrower as evaluations progress.
#
# Exploitation mechanisms: As the run progresses (and especially after finding
# improvements), the algorithm emphasizes local search by shrinking noise scale
# and biasing new samples closer to best_x.
#
# Boundary handling: Candidate points are clipped into the feasible domain using
# bounds read from func.lower/func.upper or func.bounds.lb/func.bounds.ub.
#
# Budget strategy: The algorithm performs at most the provided number of
# evaluations. It uses a fixed evaluation "batch size" computed from the budget
# and dimension, and adjusts the final batch so it never exceeds budget.
#
# Closest known influences: The design is inspired by common derivative-free
# strategies such as Evolution Strategies (ES) / CMA-like intuition (but without
# covariance matrices), plus a success-based step size adaptation.
#
# Novelty or unusual aspects: A lightweight axis-aligned perturbation is mixed
# with isotropic noise, which can help in separable or poorly conditioned
# problems without needing gradient estimation.
#
# Failure modes: For extremely noisy objectives, the algorithm may misinterpret
# noise as improvement and shrink prematurely. For very deceptive landscapes
# with long flat basins, a budget too small to escape them may lead to
# suboptimal results.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._get_bounds(func)
        dim = self.dim

        # Handle degenerate cases
        if self.budget <= 0:
            # No evaluations allowed; return a feasible point with inf value.
            x0 = np.clip(np.zeros(dim, dtype=float), lb, ub)
            return x0, float("inf")

        # Ensure bounds are arrays of shape (dim,)
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.shape[0] != dim or ub.shape[0] != dim:
            raise ValueError("Bounds must match the provided dimension.")

        span = ub - lb
        # If span is zero for some dimensions, keep those fixed.
        span_safe = np.where(span > 0, span, 1.0)

        # Initial best: sample a random feasible point and evaluate it.
        x_best = lb + span_safe * np.random.random(dim)
        y_best = self._safe_eval(func, x_best)

        evals = 1
        if evals >= self.budget:
            return x_best, y_best

        # Initial step size: fraction of the domain width.
        # Shrink over time for exploitation.
        step0 = 0.35 * span_safe
        step = step0.copy()

        # Batch size: more candidates early on for exploration, but never exceed budget.
        # Keep it small for compactness.
        pop_scale = 2
        batch = int(max(1, min(16, pop_scale * (dim + 1) // 4)))
        batch = max(1, min(batch, self.budget - evals))

        # Progress-based scaling: from 1.0 down to ~0.2
        def progress_scale():
            t = evals / max(1, self.budget)
            return max(0.2, 1.0 - t)

        # Success-based adaptation factors
        improve_factor = 0.85
        worsen_factor = 1.15
        step_min = 1e-12 * span_safe
        step_max = 0.9 * span_safe

        # We'll mix isotropic Gaussian noise with occasional axis-aligned kicks.
        # Axis probability increases a bit in exploitation phase.
        axis_prob_base = 0.15

        while evals < self.budget:
            remaining = self.budget - evals
            k = min(batch, remaining)

            # Time-varying scale (exploitation increases as progress increases).
            ps = progress_scale()
            # Adapt step with progress
            curr_step = np.clip(step * ps, step_min, step_max)

            # Generate candidates
            # Candidates are x_best + noise, clipped to bounds.
            # Use vectorized generation for efficiency with numpy only.
            noises = np.random.normal(size=(k, dim))

            # Axis-aligned mixing:
            # For each candidate, with some probability we keep only one randomly selected
            # coordinate from the noise (scaled), making "axis kicks".
            axis_prob = min(0.5, axis_prob_base + (1.0 - ps) * 0.3)
            if dim > 0:
                axis_mask = (np.random.random(size=k) < axis_prob)
                if np.any(axis_mask):
                    # Choose axis indices for those candidates
                    axes = np.random.randint(0, dim, size=k)
                    # Zero out all components except the chosen axis
                    for i in np.where(axis_mask)[0]:
                        idx = axes[i]
                        noises[i, :] = 0.0
                        noises[i, idx] = noises[i, idx]

            # Apply scaling: use elementwise step (per dimension)
            X = x_best + noises * curr_step

            # Clip to feasible region
            X = np.clip(X, lb, ub)

            # Evaluate candidates
            y_vals = np.empty(k, dtype=float)
            for i in range(k):
                y_vals[i] = self._safe_eval(func, X[i])
            evals += k

            # Find best candidate in this batch
            idx = int(np.argmin(y_vals))
            y_cand = float(y_vals[idx])
            x_cand = X[idx].copy()

            improved = y_cand < y_best
            if improved:
                x_best = x_cand
                y_best = y_cand
                # Focus more locally on improvement
                step = np.maximum(step_min, step * improve_factor)
            else:
                # Broaden a bit to escape stagnation
                step = np.minimum(step_max, step * worsen_factor)

            # Update batch size slightly to keep it within budget and compact.
            # Early exploration: larger batches; later: smaller.
            ps2 = progress_scale()
            if ps2 < 0.35:
                batch = max(1, min(batch, 4))
            else:
                batch = max(1, min(batch, 8))

        return x_best, y_best

    @staticmethod
    def _get_bounds(func):
        # Accept multiple common bound schemas.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return np.asarray(func.lower, dtype=float), np.asarray(func.upper, dtype=float)
        if hasattr(func, "bounds"):
            b = func.bounds
            # Try lb/ub first
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return np.asarray(b.lb, dtype=float), np.asarray(b.ub, dtype=float)
            # Try lower/upper on bounds object
            if hasattr(b, "lower") and hasattr(b, "upper"):
                return np.asarray(b.lower, dtype=float), np.asarray(b.upper, dtype=float)
        raise ValueError("Cannot read bounds from func. Expected func.lower/func.upper "
                         "or func.bounds.lb/func.bounds.ub.")

    @staticmethod
    def _safe_eval(func, x):
        # Standardize objective evaluation: ensure float output.
        y = func(x)
        return float(y)
