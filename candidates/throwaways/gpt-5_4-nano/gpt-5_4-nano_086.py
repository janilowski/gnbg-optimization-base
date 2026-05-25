import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm using a blend of
# global exploration (fitness-based candidate sampling) and local exploitation
# (coordinate-wise shrinking around the best point). It is designed to work in
# arbitrary dimensions, while strictly respecting the evaluation budget.
#
# Search state: Tracks the current best solution (x_best, y_best), the number
# of evaluations already used, and a mutable step size (sigma) that controls
# how far candidates are sampled from the current best.
#
# Candidate generation: Each iteration samples a small batch of random candidates
# around x_best using Gaussian perturbations with scale sigma, plus one
# structured coordinate perturbation proposal to improve local coverage.
# Candidates are projected back into bounds before evaluation.
#
# Selection and replacement: After evaluating all candidates in the batch, the
# best candidate is selected. If it improves the current best, x_best and
# y_best are replaced.
#
# Adaptation: sigma is adapted based on success. On improvement, sigma is
# slightly increased to continue progress; otherwise sigma is decreased to focus
# on local refinement.
#
# Exploration mechanisms: Random sampling around the best point with a moderately
# large initial sigma plus occasional larger perturbations provides exploration.
#
# Exploitation mechanisms: When improving (or after stagnation), sigma shrinks
# to concentrate sampling near the best, effectively refining the solution.
#
# Boundary handling: Uses a projection (clipping) of all candidate coordinates to
# lie within the provided lower/upper bounds.
#
# Budget strategy: Computes a conservative number of iterations and candidates
# per iteration so that total objective evaluations never exceed the provided
# budget. Includes the initial evaluation of x_best and then only evaluates
# new candidates up to the remaining budget.
#
# Closest known influences: Inspired by CMA-ES-like sampling ideas and the
# general pattern of success-based step size control, simplified to remain
# compact and dependency-free.
#
# Novelty or unusual aspects: Combines batch random sampling with an additional
# coordinate-perturbation proposal each iteration, giving deterministic local
# structure alongside stochastic exploration.
#
# Failure modes: If the objective is very noisy or has extremely irregular
# landscapes, the simplistic success-based adaptation may stall or oscillate.
# In very low budgets, performance may be limited by the small number of
# evaluations.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim

        # Read bounds from func.lower/upper or func.bounds.lb/ub (as required).
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via func.lower/func.upper or func.bounds.lb/ub")

        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))
        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimension mismatch with Algorithm.dim")

        # Ensure valid bounds ordering.
        if np.any(ub < lb):
            raise ValueError("Upper bounds must be >= lower bounds for all coordinates")

        # Helpers.
        def project(x):
            return np.minimum(np.maximum(x, lb), ub)

        def eval_counted(x, y_cache):
            # Objective minimization. Strictly count evaluations.
            if y_cache["n"] >= self.budget:
                return y_cache["best_y"]
            y = float(func(np.asarray(x, dtype=float)))
            y_cache["n"] += 1
            return y

        # Initialize best point. Use center of bounds to be robust.
        x_center = project(0.5 * (lb + ub))

        y_cache = {"n": 0, "best_y": np.inf}
        y_best = eval_counted(x_center, y_cache)
        x_best = x_center.copy()
        y_cache["best_y"] = y_best

        # If budget is 0 or 1, return immediately.
        if self.budget <= 1:
            return x_best, y_best

        # Initial sigma: fraction of the typical range.
        span = ub - lb
        # Avoid zero spans: replace zeros with 1.0 for scaling only.
        span_safe = np.where(span > 0, span, 1.0)
        sigma = 0.25 * float(np.median(span_safe))
        sigma = max(sigma, 1e-12)

        # Decide batch size and iteration count so total evals won't exceed budget.
        # We'll use a small batch to reduce overhead; adapt batch near budget.
        # Candidate count per iteration includes only new evaluations.
        max_new = self.budget - y_cache["n"]
        batch = min(dim + 3, 12)  # reasonable batch cap
        batch = max(2, batch)
        # Number of iterations based on remaining evaluations.
        iters = max(1, int(np.ceil(max_new / batch)))

        # Success-based adaptation hyperparameters.
        # Increase sigma modestly on success; decrease on failure.
        inc = 1.12
        dec = 0.82
        # Exploration occasionally with larger steps.
        explore_prob = 0.25

        rng = np.random

        for _ in range(iters):
            if y_cache["n"] >= self.budget:
                break

            remaining = self.budget - y_cache["n"]
            k = min(batch, remaining)
            if k <= 0:
                break

            # Random exploration around current best.
            # Generate in a vectorized way for speed.
            # candidates shape: (k, dim)
            candidates = np.empty((k, dim), dtype=float)

            # First candidate: optionally a larger perturbation for exploration.
            # This improves global search when sigma is small early.
            if k >= 1:
                if rng.rand() < explore_prob:
                    z = rng.randn(dim)
                    candidates[0] = x_best + (2.0 * sigma) * z
                else:
                    z = rng.randn(dim)
                    candidates[0] = x_best + sigma * z

            # Remaining candidates: Gaussian around x_best.
            start_idx = 1
            if k > start_idx:
                z = rng.randn(k - start_idx, dim)
                candidates[start_idx:] = x_best + sigma * z

            # Structured coordinate perturbation (if space allows).
            # This adds deterministic local structure.
            if k >= 2:
                j = int(rng.randint(0, dim))
                step_dir = 1.0 if rng.rand() < 0.5 else -1.0
                # Use a step proportional to sigma and span in that coordinate.
                coord_span = span_safe[j]
                candidates[1, j] = x_best[j] + step_dir * (0.75 * sigma + 0.05 * coord_span)
                # Keep other coordinates identical to x_best to focus the proposal.
                if dim > 1:
                    candidates[1, np.arange(dim) != j] = x_best[np.arange(dim) != j]

            # Boundary handling and evaluation.
            improved = False
            best_local_y = y_best
            best_local_x = x_best

            for i in range(k):
                if y_cache["n"] >= self.budget:
                    break
                xi = project(candidates[i])
                yi = eval_counted(xi, y_cache)
                if yi < best_local_y:
                    best_local_y = yi
                    best_local_x = xi
                    improved = True

            # Update global best and adapt sigma.
            if improved:
                x_best = best_local_x
                y_best = best_local_y
                sigma = min(2.0 * sigma * inc, 5.0 * float(np.max(span_safe)))
            else:
                sigma = sigma * dec

            # If sigma becomes extremely small relative to feasible span, prevent stall
            # by resetting to a small fraction of median span_safe.
            if sigma < 1e-15:
                sigma = 0.1 * float(np.median(span_safe))
                sigma = max(sigma, 1e-12)

        return x_best, y_best
