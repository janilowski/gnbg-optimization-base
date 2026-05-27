# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# (a derivative-free strategy) using a population of candidate points, repeated
# local perturbations, and a shrinking step-size schedule. It is designed to
# work for any dimension and relies only on the provided evaluation budget.
#
# Search state: The algorithm maintains a current best solution (best_x, best_y),
# a step size sigma that controls the magnitude of random perturbations, and a
# small population used to evaluate multiple candidates per iteration.
#
# Candidate generation: Each iteration samples new candidates by perturbing
# the current best vector with Gaussian noise of scale sigma. Candidates are
# optionally biased toward the current best via the perturbation center.
#
# Selection and replacement: After evaluating all sampled candidates (plus the
# best), the algorithm selects the best among them and updates best_x/best_y.
# The population is not retained; instead the best point becomes the new center
# for the next iteration.
#
# Adaptation: sigma shrinks as the remaining budget decreases to encourage
# convergence, but a small fraction of random “global” exploration is kept to
# reduce stagnation.
#
# Exploration mechanisms: A restart-like exploration is implemented by occasional
# candidates sampled uniformly within bounds (with probability that decreases
# over time). Additionally, the Gaussian sampling always has nonzero variance.
#
# Exploitation mechanisms: Most candidates are generated from the current best
# point, focusing search locally with a decreasing sigma.
#
# Boundary handling: All candidates are clipped to the feasible box defined by
# func.lower/func.upper or func.bounds.lb/ub to ensure all evaluations remain valid.
#
# Budget strategy: The total number of objective evaluations is capped strictly
# by the provided budget. The algorithm uses a per-iteration batch size that
# adapts to the remaining evaluations, including the initial evaluation of the
# starting point.
#
# Closest known influences: This is loosely inspired by evolutionary strategies /
# CMA-like patterns (population sampling around the incumbent) but implemented
# with simpler mechanics and explicit budget control.
#
# Novelty or unusual aspects: The exploration is implemented via a scheduled mix
# of uniform-in-bounds candidates and Gaussian-increment candidates, combined with
# a deterministic budget-aware iteration scheme.
#
# Failure modes: If the objective is extremely noisy or has very narrow feasible
# improvements, random exploration may waste evaluations. In very high dimensions,
# the local Gaussian search might require a larger budget to see progress.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        n = self.dim
        budget = max(0, int(self.budget))
        if budget == 0:
            # No evaluations allowed; return a consistent vector with NaN objective.
            return np.zeros(n, dtype=float), float("nan")

        # ---- Read bounds from func ----
        lb = ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        if lb is None or ub is None:
            # If bounds are not available, fall back to a wide box.
            # (Still robust for unconstrained harnesses.)
            lb = -5.0 * np.ones(n, dtype=float)
            ub = 5.0 * np.ones(n, dtype=float)

        # Ensure correct shapes
        if lb.shape == ():
            lb = np.full(n, float(lb), dtype=float)
        if ub.shape == ():
            ub = np.full(n, float(ub), dtype=float)
        if lb.shape[0] != n or ub.shape[0] != n:
            # Broadcast scalars or attempt to reshape if compatible
            lb = np.resize(lb, n)
            ub = np.resize(ub, n)

        # Handle degenerate bounds safely
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)
        width = ub - lb
        width = np.where(width > 0, width, 1.0)

        def clip_to_bounds(x):
            return np.minimum(np.maximum(x, lb), ub)

        # Determine a starting point: midpoint (clipped)
        x0 = clip_to_bounds((lb + ub) * 0.5)

        # Evaluation accounting
        evals = 0

        # Objective wrapper to enforce budget usage
        def eval_one(x):
            nonlocal evals
            if evals >= budget:
                # Should not happen if logic is correct; keep safe.
                return float("inf")
            y = func(x)
            evals += 1
            # Ensure scalar float
            return float(np.asarray(y).reshape(()))

        # Initialize best
        best_x = x0.copy()
        best_y = eval_one(best_x)

        # Early exit if budget exhausted by initial evaluation
        if evals >= budget:
            return best_x, best_y

        rng = np.random

        # Initial sigma based on the scale of the search space
        # Use a fraction of box width; avoid sigma=0
        sigma = 0.3 * np.mean(width)
        sigma = float(sigma if sigma > 1e-12 else 1.0)

        # Batch size: balance exploration and budget usage
        # Keep it small enough for overhead and robust across dimensions.
        # Typical values: 2-16
        base_pop = int(np.clip(8, 2, max(2, min(16, n + 2))))
        base_pop = max(2, base_pop)

        # Global exploration schedule
        # Starts higher, decays over time.
        def exploration_prob(progress):
            # progress in [0,1]
            # Smooth decay from ~0.25 to ~0.02
            return 0.02 + 0.23 * (1.0 - progress) ** 2

        # Main loop: each iteration evaluates up to batch candidates
        while evals < budget:
            remaining = budget - evals
            pop = min(base_pop, remaining)

            # Adapt sigma with remaining budget (shrink as we use evaluations)
            progress = evals / max(1, budget)
            # Exponential-like shrink for stability
            sigma_eff = sigma * (1.0 - progress) + (sigma * 0.05) * progress
            sigma_eff = max(sigma_eff, 1e-12)

            # Prepare candidates
            # A mix of local Gaussian perturbations and occasional uniform samples.
            cand = np.empty((pop, n), dtype=float)

            p_explore = exploration_prob(progress)

            for i in range(pop):
                if rng.rand() < p_explore:
                    # Uniform exploration within bounds
                    r = rng.rand(n)
                    cand[i] = lb + r * (ub - lb)
                else:
                    # Local exploitation around best_x
                    # Use isotropic Gaussian with per-dim scaling from width.
                    step = rng.normal(0.0, 1.0, size=n)
                    cand[i] = best_x + (sigma_eff * step) * (width / np.mean(width))

                cand[i] = clip_to_bounds(cand[i])

            # Evaluate and select best in this batch
            # Ensure we never exceed the budget.
            batch_best_y = float("inf")
            batch_best_x = None

            for i in range(pop):
                if evals >= budget:
                    break
                x = cand[i]
                y = eval_one(x)
                if y < batch_best_y:
                    batch_best_y = y
                    batch_best_x = x.copy()

            # Update global best
            if batch_best_x is not None and batch_best_y < best_y:
                best_x = batch_best_x
                best_y = batch_best_y
                # Slightly increase sigma when we improve to allow further local exploration
                sigma = sigma_eff * 1.03
            else:
                # No improvement: shrink sigma more aggressively to focus
                sigma = sigma_eff * 0.85

            # Guard against numeric collapse
            if sigma < 1e-14:
                # Ensure continued exploration if stuck
                sigma = 1e-14 + 0.1 * np.mean(width)

        return best_x, best_y
