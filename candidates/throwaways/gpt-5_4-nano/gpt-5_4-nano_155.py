# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm for a
# continuous search space with box constraints. It combines a randomized
# sampling phase with a coordinate-wise local improvement phase, guided by
# the best-so-far point. The algorithm is budget-aware and never evaluates
# the objective more than the provided budget.
# Search state: Maintains the current best solution (best_x, best_y) and a
# step size (sigma) that shrinks over time. Also keeps track of remaining
# evaluation calls.
# Candidate generation: In each iteration, it samples candidate points around
# the current best using Gaussian perturbations and also produces candidates via
# coordinate-wise moves (±sigma along randomly permuted axes). Candidates are
# clipped to remain within the provided bounds.
# Selection and replacement: If a candidate achieves a lower objective value, it
# replaces the current best and triggers a mild step-size adaptation.
# Adaptation: sigma decays gradually as budget is consumed; additionally, after
# a successful improvement, sigma is slightly reduced to focus search, while
# after repeated failures it may be kept larger to escape plateaus.
# Exploration mechanisms: Early in the run, more randomized sampling is used to
# explore broadly; Gaussian perturbations encourage diverse directions.
# Exploitation mechanisms: The coordinate-wise trial moves and progressive
# shrinkage of sigma improve local refinement near the best point.
# Boundary handling: All generated candidates are projected back into the box
# constraints via clipping. Bounds are read from func.lower/func.upper or
# func.bounds.lb/func.bounds.ub.
# Budget strategy: Tracks remaining evaluations and stops exactly when the
# budget is exhausted (or when no further evaluations are possible).
# Closest known influences: Inspired by simple evolution strategies / CMA-like
# step-size decay and coordinate descent, but implemented as a robust,
# lightweight hybrid suitable for black-box benchmarking.
# Novelty or unusual aspects: The method uses a dynamic split between global
# sampling and local coordinate refinements based on remaining budget, plus a
# conservative improvement-driven sigma adjustment without any external state.
# Failure modes: In very noisy or highly non-smooth objectives, improvements may
# be rare and sigma decay can lead to premature local behavior. If the optimum
# lies near a boundary, clipping may reduce effective exploration along
# infeasible directions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0 or dim <= 0:
            # Evaluate nothing; return a safe default.
            # (Budget should normally be >= 1 for benchmarks.)
            x0 = np.zeros(dim, dtype=float)
            return x0, float("inf")

        # ---- Read bounds from func ----
        lower = upper = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective function must provide bounds via lower/upper or bounds.lb/bounds.ub.")

        if lower.shape != (dim,) or upper.shape != (dim,):
            lower = lower.reshape(-1)
            upper = upper.reshape(-1)
            if lower.size != dim or upper.size != dim:
                raise ValueError("Bounds shape does not match dim.")

        # Ensure well-formed bounds
        lb = np.minimum(lower, upper)
        ub = np.maximum(lower, upper)
        span = ub - lb
        span = np.where(span > 0, span, 1.0)  # avoid zero span issues

        # ---- Budget-aware evaluator ----
        evals = 0

        def eval_at(x):
            nonlocal evals
            if evals >= budget:
                return float("inf")
            evals += 1
            y = func(x)
            # Ensure scalar float
            return float(y)

        # ---- Initialization ----
        # Start best at a random point; also optionally include an extreme point early.
        # Use uniform random inside bounds.
        x_best = lb + span * np.random.random(dim)
        y_best = eval_at(x_best)

        # Choose initial sigma proportional to average span
        sigma = 0.35 * float(np.mean(span))
        # Guard against degenerate sigma
        sigma = max(sigma, 1e-12)

        # Determine how many global sampling steps we can afford.
        # Keep it budget-aware and dimension-aware.
        # More dimensions => fewer expensive global samples.
        max_global = min(budget, max(0, int(0.3 * budget)))
        # Leave remaining for local coordinate refinement.
        remaining_global = max_global

        # Precompute random axis permutations each iteration for variety.
        # (No heavy memory; just generated on the fly.)
        # ---- Main loop ----
        while evals < budget:
            # Decay sigma as we consume budget
            t = evals / max(1, budget)
            sigma = max(1e-12, sigma * (1.0 - 0.65 * (t - 0.0) / (1.0 + 0.0 * t)))

            improved = False

            # Decide whether to explore globally or exploit locally.
            do_global = remaining_global > 0 and (np.random.random() < (0.6 if remaining_global > 0 else 0.0))

            if do_global:
                # Global exploration: Gaussian around best with anisotropic scaling
                remaining_global -= 1
                # Number of candidates per global phase (keep small to respect budget)
                k = 3 if dim <= 10 else 2
                for _ in range(k):
                    if evals >= budget:
                        break
                    # Random Gaussian perturbation, scaled by span
                    z = np.random.randn(dim)
                    # Heavier scaling in early stages
                    cand = x_best + (sigma * z) * (0.5 + 0.5 * np.random.random(dim))
                    cand = np.clip(cand, lb, ub)
                    y = eval_at(cand)
                    if y < y_best:
                        x_best, y_best = cand, y
                        improved = True
                # If no improvement, mildly increase exploration a bit (bounded)
                if not improved:
                    sigma *= 1.05
            else:
                # Local exploitation: coordinate-wise trials (random order)
                # Try ±sigma along a subset/ all coordinates depending on budget & dim.
                order = np.random.permutation(dim)
                # Choose number of axes to try this round
                # Balance: for large dim, try fewer axes per loop to save evaluations.
                # Ensure at least a few attempts.
                axes_to_try = min(dim, max(2, int(0.15 * dim)))
                axes = order[:axes_to_try]

                # Alternate direction choices to reduce bias
                for i, ax in enumerate(axes):
                    if evals >= budget:
                        break
                    step = sigma
                    # Randomize sign
                    sign = -1.0 if (np.random.random() < 0.5) else 1.0
                    # Candidate
                    cand = x_best.copy()
                    cand[ax] = np.clip(cand[ax] + sign * step, lb[ax], ub[ax])
                    # If clipped back to same value, also try opposite direction.
                    y = eval_at(cand)
                    if y < y_best:
                        x_best, y_best = cand, y
                        improved = True
                        continue
                    if cand[ax] == x_best[ax] and evals < budget:
                        cand[ax] = np.clip(cand[ax] - 2.0 * sign * step, lb[ax], ub[ax])
                        y2 = eval_at(cand)
                        if y2 < y_best:
                            x_best, y_best = cand, y2
                            improved = True

                    # Slight sigma reduction as we get closer to local optimum
                    if (i + 1) % 4 == 0:
                        sigma *= 0.9

                # If improved, focus search slightly
                if improved:
                    sigma *= 0.8
                else:
                    # If no improvement, reduce shrinkage to keep some exploration.
                    sigma *= 1.03

            # If we're close to evaluation budget, do a final small neighborhood search.
            if evals >= budget:
                break

            # Periodically try one additional random refinement step near best.
            # Keep it extremely budget-conscious.
            if (budget - evals) <= max(1, 0.15 * budget) and evals < budget:
                # Try up to remaining evaluations (but cap to 2 candidates)
                m = min(2, budget - evals)
                for _ in range(m):
                    if evals >= budget:
                        break
                    z = np.random.randn(dim)
                    # Smaller neighborhood near end
                    cand = x_best + (0.3 * sigma) * z
                    cand = np.clip(cand, lb, ub)
                    y = eval_at(cand)
                    if y < y_best:
                        x_best, y_best = cand, y
                break

        # Return best found
        return np.asarray(x_best, dtype=float), float(y_best)
