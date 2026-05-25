# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy (derivative-free)
# that alternates between randomized exploration and local coordinate-based
# exploitation. It is designed to work robustly across dimensions while
# respecting the provided evaluation budget.
# Search state: Maintains the current best solution x_best, its objective value
# y_best, an iteration counter, and a step-size vector sigma controlling the
# radius of local moves.
# Candidate generation: Each iteration samples a batch of candidate points around
# the current best using Gaussian perturbations scaled by sigma, plus occasional
# orthogonal coordinate moves to probe local structure.
# Selection and replacement: Evaluates candidates, keeps the best point found, and
# replaces x_best/y_best if improvement occurs.
# Adaptation: Uses a simple 1/5-like adaptation rule: if enough improvements are
# observed in a round, step size increases; otherwise it decreases.
# Exploration mechanisms: Early rounds use larger step sizes and broader random
# perturbations to explore. Occasional coordinate moves help escape poor regions.
# Exploitation mechanisms: After improvements, sigma shrinks, making the method focus
# on local refinements around the best point.
# Boundary handling: Applies clipping to ensure candidates stay within the provided
# bounds (from func.lower/func.upper or func.bounds.lb/ub).
# Budget strategy: Uses exactly the evaluation budget by precomputing how many
# candidates can be evaluated each loop and stopping once the budget is exhausted.
# Closest known influences: Inspired by evolution strategies / CMA-like ideas, but kept
# simpler: (1+λ)-style selection with adaptive step size and bounded clipping.
# Novelty or unusual aspects: Combines Gaussian sampling with sporadic coordinate
# search steps; adaptation is based on the fraction of improved candidates in each
# round rather than on strict success counts.
# Failure modes: If the objective is highly ill-conditioned or extremely noisy, the
# step-size adaptation may converge prematurely or oscillate; clipping at tight
# bounds can reduce effective search diversity.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # --- Read bounds robustly ---
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
            ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)
        else:
            raise AttributeError("func must provide bounds via lower/upper or bounds.lb/bounds.ub")

        if lb.size != dim or ub.size != dim:
            raise ValueError(f"Bounds dimension mismatch: expected dim={dim}, got lb={lb.size}, ub={ub.size}")

        # Ensure numeric sanity and avoid inverted bounds
        lb, ub = np.minimum(lb, ub), np.maximum(lb, ub)
        span = ub - lb
        span = np.where(span > 0, span, 1.0)  # protect against zero-width dimensions

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        # --- Evaluation budget tracking ---
        evals = 0

        def eval_one(x):
            nonlocal evals
            # Never exceed budget: caller ensures we don't call too many times
            y = func(x)
            evals += 1
            return float(y)

        # If budget is 0, return something deterministic-ish (but budget=0 is unusual)
        if budget <= 0:
            x0 = lb.copy()
            return x0, float(func(x0))

        # --- Initialize ---
        # Start at a random point in the box
        rng = np.random
        x_best = lb + rng.rand(dim) * span
        x_best = clip(x_best)
        y_best = eval_one(x_best)

        # Initialize step size: fractions of span
        sigma = 0.35 * span
        # For stability in very small span dimensions
        sigma = np.where(sigma > 0, sigma, 1e-12)

        # --- Main loop ---
        # Use variable batch size depending on remaining budget.
        # (1+λ)-style with adaptive step-size.
        while evals < budget:
            remaining = budget - evals

            # Heuristic batch size: proportional to dim but capped for budget granularity
            lam = min(max(4, 2 * dim), remaining)

            # Exploration/exploitation schedule:
            # early rounds: larger sigma; later: more conservative
            t = evals / max(1, budget)
            exp_scale = 1.0 + 0.6 * (1.0 - t)  # slightly more exploratory early
            sig = sigma * exp_scale

            # Draw candidates around x_best
            # Gaussian perturbations scaled per-dimension by sig
            Z = rng.randn(lam, dim)
            X = x_best + Z * sig[None, :]

            # Occasional coordinate probing to help in axis-aligned structures.
            # Choose k coordinates based on dimension.
            if lam >= 3 and dim >= 2:
                k = min(dim, max(1, lam // 3))
                coords = rng.choice(dim, size=k, replace=False)
                # Choose a subset of candidate indices to modify
                idxs = rng.choice(lam, size=min(len(coords), lam), replace=False)
                for j, coord in enumerate(coords[: len(idxs)]):
                    i = idxs[j]
                    # Move along that coordinate with random sign and a fraction of span
                    step = (0.25 + 0.75 * rng.rand()) * sigma[coord]
                    X[i, coord] = x_best[coord] + (1 if rng.rand() < 0.5 else -1) * step

            # Boundary handling
            X = clip(X)

            # Evaluate candidates (but ensure we do not exceed budget)
            n_eval = min(lam, budget - evals)
            best_y_round = y_best
            best_x_round = x_best
            improved = 0

            for i in range(n_eval):
                y = eval_one(X[i])
                if y < best_y_round:
                    best_y_round = y
                    best_x_round = X[i].copy()
                if y < y_best:
                    improved += 1

            # Update best
            if best_y_round < y_best:
                x_best = best_x_round
                y_best = best_y_round

            # Adapt sigma based on success rate in the evaluated batch.
            # Target success fraction ~ 1/5.
            success_frac = improved / max(1, n_eval)
            # Multiplicative adaptation: increase if many successes; else decrease.
            # Use bounds on adaptation to avoid blow-ups.
            if success_frac > 0.2:
                sigma = np.minimum(ub - lb, sigma * 1.2)
            else:
                sigma = sigma * 0.82

            # If all dimensions are at a fixed bound (span ~ 0), sigma might go to zero.
            # Keep sigma non-zero to allow some stochasticity in feasible space.
            sigma = np.where(sigma > 0, sigma, 1e-12)

        return x_best, y_best
