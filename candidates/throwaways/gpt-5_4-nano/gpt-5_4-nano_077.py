# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, budget-aware black-box minimization algorithm
# inspired by an adaptive trust-region / coordinate-pattern search. It maintains a small
# population of candidate points, evaluates them using the provided objective function,
# and iteratively refines a current best solution by sampling around it with an
# anisotropic step size.
# Search state: The algorithm tracks the best point found so far (best_x) and its
# objective value (best_y), a current step size vector (sigma), a success counter for
# step-size adaptation, and a remaining evaluation budget.
# Candidate generation: Each iteration generates a set of candidates by adding scaled
# random perturbations and coordinate-wise moves to the current best. A few additional
# candidates use orthogonal-ish directions formed from randomized sign masks to improve
# coverage in higher dimensions.
# Selection and replacement: Candidates are evaluated, and the best among them replaces
# the current best if it improves. The population is not maintained persistently; instead
# the algorithm focuses evaluations on areas around the best point.
# Adaptation: The step size sigma shrinks when no improvement is found (failed iteration)
# and expands slightly after improvement (success), with per-dimension scaling to better
# match the domain.
# Exploration mechanisms: Random Gaussian perturbations (plus some sign/axis-based moves)
# provide exploration; occasional larger moves are used when sigma is still large.
# Exploitation mechanisms: Moves are centered at the current best, and coordinate-pattern
# sampling biases search toward structured local improvements.
# Boundary handling: After generating a candidate, it is clipped to the provided bounds
# (from func.lower/upper or func.bounds.lb/ub). This guarantees feasibility.
# Budget strategy: The implementation never exceeds the user-provided evaluation budget.
# It computes how many evaluations are remaining before each batch and uses only that
# many evaluations for candidate evaluation.
# Closest known influences: The design loosely follows derivative-free evolution strategy
# ideas (sampling around best) combined with 1/5-style step-size adaptation logic and
# trust-region-like shrinking.
# Novelty or unusual aspects: The update uses a per-dimension sigma vector initialized
# from the bound widths and maintained with anisotropic scaling, with a small mix of
# random and coordinate/signed directions to remain effective across dimensions.
# Failure modes: If the objective is very noisy or highly multi-modal, the shrinking
# schedule may converge prematurely. If bounds are extremely tight or inconsistent, clipping
# may dominate behavior, reducing search effectiveness.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        # Batch sizes: small enough to keep adaptation responsive, large enough for coverage.
        # Ensure at least 1.
        self._base_batch = max(4, min(16, self.dim + 3))

    def __call__(self, func):
        # ---- Bounds handling ----
        lower, upper = self._get_bounds(func)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        dim = self.dim

        if lower.shape != (dim,) or upper.shape != (dim,):
            # Try to coerce if func provides scalar bounds
            lower = np.broadcast_to(lower, (dim,)).astype(float, copy=False)
            upper = np.broadcast_to(upper, (dim,)).astype(float, copy=False)

        width = upper - lower
        # Guard against zero width dimensions (constant bounds).
        width = np.where(width == 0.0, 1.0, width)

        # ---- Evaluation function with strict budget enforcement ----
        evals_used = 0
        remaining = self.budget

        def eval_one(x):
            nonlocal evals_used, remaining
            if remaining <= 0:
                raise RuntimeError("Evaluation budget exceeded.")
            y = float(func(x))
            evals_used += 1
            remaining -= 1
            return y

        def clip(x):
            return np.minimum(np.maximum(x, lower), upper)

        rng = np.random

        # ---- Initialization ----
        # Random starting point within bounds.
        x0 = lower + rng.rand(dim) * (upper - lower)
        x0 = clip(x0)
        best_x = x0
        best_y = eval_one(best_x)

        # Initialize per-dimension step sizes from bound width.
        # Start moderately small to remain feasible, but not too tiny.
        sigma = 0.25 * width

        # Success counter for adaptation
        success = 0

        # If budget is extremely small, return quickly
        if self.budget <= 1:
            return best_x, best_y

        # ---- Main loop ----
        while remaining > 0:
            # Choose batch size based on remaining budget
            batch = min(self._base_batch, remaining)
            if batch <= 0:
                break

            # Always include the current best as a candidate (helps when budget is tight)
            candidates = []
            candidates.append(best_x.copy())

            # Remaining candidate count to generate
            to_gen = batch - 1

            # If to_gen > 0, create structured/random candidate proposals
            # Mix of Gaussian perturbations and sign/coordinate patterns.
            # Use per-dimension sigma to be effective in varying scales.
            # Gaussian proposals:
            #   x = best_x + N(0, sigma^2)
            # Sign/axis patterns:
            #   x = best_x + step * sigma * s where s in {-1, +1} (and some axis-aligned)
            while len(candidates) < batch:
                k = len(candidates)
                frac = k / max(1, batch - 1)

                if (k % 3) == 1:
                    # Gaussian perturbation
                    step = rng.randn(dim) * sigma
                    x = best_x + step
                elif (k % 3) == 2:
                    # Signed perturbation, occasionally larger
                    s = rng.choice([-1.0, 1.0], size=dim)
                    scale = 1.0 + 0.5 * rng.rand()
                    # Bias slightly larger when sigma is large or early in the run
                    if frac < 0.35:
                        scale *= 1.2
                    x = best_x + (sigma * s) * scale
                else:
                    # Coordinate-pattern move: pick a few axes, move along them
                    x = best_x.copy()
                    n_axes = 1 if dim == 1 else int(np.clip(1 + rng.randint(0, min(dim, 4)), 1, dim))
                    axes = rng.choice(dim, size=n_axes, replace=False)
                    # Draw small coefficients and apply
                    coeffs = (rng.rand(n_axes) * 2.0 - 1.0)  # in [-1, 1]
                    # Local exploitation: smaller step by default
                    local_scale = 0.6 + 0.8 * rng.rand()
                    x[axes] = x[axes] + sigma[axes] * coeffs * local_scale

                candidates.append(clip(x))

            # Evaluate candidates
            improved = False
            for x in candidates:
                if remaining <= 0:
                    break
                y = eval_one(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                    improved = True

            # ---- Adapt sigma based on improvement ----
            # Simple success/failure adaptation with per-dimension scaling.
            if improved:
                success += 1
                # Mild expansion to escape too-small steps after finding improvement.
                # Cap to avoid exploding steps beyond the domain.
                expand = 1.15 + 0.05 * rng.rand()
                sigma = np.minimum(sigma * expand, 0.9 * width + 1e-12)
            else:
                success = 0
                # Shrink when no improvement
                shrink = 0.85 + 0.05 * rng.rand()
                sigma = np.maximum(sigma * shrink, 1e-12 * width)

            # If sigma is effectively tiny in all dimensions, reset partially to keep exploration
            if np.all(sigma <= 1e-12 * width) and remaining > 0:
                sigma = 0.25 * width

        return best_x, best_y

    @staticmethod
    def _get_bounds(func):
        # Prefer func.lower/func.upper
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return func.lower, func.upper
        # Or func.bounds.lb / func.bounds.ub
        if hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            return func.bounds.lb, func.bounds.ub

        # Last resort: if bounds not found, create a broad default.
        # (Robustness for mis-specified harnesses; typical benchmarks provide bounds.)
        # Dimension is inferred from either func.dim or Algorithm.dim elsewhere, but we keep
        # it here simple and raise to make errors explicit.
        raise AttributeError(
            "Objective function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
        )
