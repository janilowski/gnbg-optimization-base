# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, budget-aware black-box minimizer using a
# mix of global exploration (random sampling) and local exploitation
# (coordinate-wise Gaussian search around the current best). It is robust
# across dimensions by using simple vectorized operations with adaptive
# step sizes.
# Search state: Maintains the current best point x_best and its objective
# value y_best. Also tracks a current step scale (sigma) and an exploration
# phase probability that decays over time.
# Candidate generation: At each iteration, either samples a population of
# random candidates within bounds (global step) or generates a small batch
# of local candidates by perturbing x_best with isotropic Gaussian noise.
# Additionally, a lightweight coordinate-wise refinement is performed in
# later iterations via axis-aligned probes.
# Selection and replacement: Evaluates a batch of candidates, chooses the best
# among them, and replaces x_best/y_best if an improvement is found.
# Adaptation: If no improvement is observed for several iterations, sigma is
# reduced to focus search; if improvements occur, sigma is modestly increased
# to keep progress.
# Exploration mechanisms: Early-stage random sampling and larger sigma.
# Exploitation mechanisms: Later-stage local Gaussian perturbations and
# coordinate-wise probes around x_best.
# Boundary handling: Candidate points are clipped to the feasible bounds.
# Budget strategy: The algorithm strictly respects the evaluation budget by
# tracking evaluations and stopping when the remaining budget cannot support
# another full batch.
# Closest known influences: Related in spirit to evolution strategies and
# coordinate descent hybrids, but implemented as a simple, compact hybrid
# without external dependencies.
# Novelty or unusual aspects: Uses adaptive batch sizing to match the remaining
# budget and a decaying exploration probability to transition from global to
# local search smoothly.
# Failure modes: Can stagnate on flat landscapes or highly constrained,
# narrow minima if initial sampling misses the basin of attraction. Budget
# too small relative to dimension may limit effectiveness.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds ----
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("func must provide bounds via lower/upper or bounds.lb/bounds.ub")

        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        lb = lb.reshape(-1).copy()
        ub = ub.reshape(-1).copy()
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimensionality does not match dim")

        # Handle degenerate bounds safely
        width = np.maximum(ub - lb, 0.0)
        width_pos = np.where(width > 0, width, 1.0)

        def clip_to_bounds(x):
            return np.minimum(np.maximum(x, lb), ub)

        # ---- Budget-aware evaluation wrapper ----
        evals = 0

        def eval_one(x):
            nonlocal evals
            if evals >= budget:
                # Must never exceed budget; return +inf as "unavailable"
                return np.inf
            y = func(x)
            evals += 1
            return float(y)

        # ---- Initialization ----
        # Start with one random point; also try a few more if budget allows.
        x_best = clip_to_bounds(lb + width_pos * np.random.rand(dim))
        y_best = eval_one(x_best)

        # Choose batch size based on dimension and remaining budget.
        # Keep it small to control evaluations, but allow some parallel-like behavior.
        def suggested_batch_size(remaining):
            # Heuristic: bigger batches early, smaller later
            base = 1 + dim // 2
            # Cap to keep cost reasonable for high dims
            base = min(base, 32)
            # Also respect remaining budget
            return int(max(1, min(base, remaining)))

        # Step size: start at a fraction of the range.
        sigma = 0.25 * width_pos
        # Prevent sigma from becoming all zeros when bounds are tight.
        sigma = np.where(width > 0, sigma, 1e-6)

        # Exploration probability decays with time.
        explore_prob_start = 0.9
        explore_prob_end = 0.15

        # Track stagnation to adapt sigma
        no_improve_streak = 0

        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break

            # Decide phase
            t = evals / max(1, budget)
            explore_prob = explore_prob_start * (1.0 - t) + explore_prob_end * t
            do_explore = (np.random.rand() < explore_prob) or (no_improve_streak >= 6)

            batch_size = suggested_batch_size(remaining)
            # Ensure we can at least evaluate something
            if batch_size <= 0:
                break

            # ---- Candidate generation ----
            if do_explore:
                # Global exploration: sample uniformly in bounds
                # Create a batch in a vectorized manner.
                r = np.random.rand(batch_size, dim)
                X = lb + r * (ub - lb)
                X = np.clip(X, lb, ub)
            else:
                # Local exploitation: perturb around x_best
                # Use isotropic-ish noise scaled by sigma.
                # Use a modest number of candidates, biased towards smaller steps later.
                # Decay factor within the run.
                local_decay = 1.0 - 0.75 * t
                cand_sigma = np.maximum(sigma * local_decay, 1e-12)

                Z = np.random.randn(batch_size, dim)
                X = x_best[None, :] + Z * cand_sigma[None, :]
                X = clip_to_bounds(X)

            # ---- Evaluate batch and select best ----
            best_batch_y = np.inf
            best_batch_x = None

            for i in range(X.shape[0]):
                y = eval_one(X[i])
                if y < best_batch_y:
                    best_batch_y = y
                    best_batch_x = X[i].copy()

                if evals >= budget:
                    break

            if best_batch_x is not None and best_batch_y < y_best:
                x_best, y_best = best_batch_x, best_batch_y
                no_improve_streak = 0
                # If we improved, slightly increase sigma to search broader around the new best
                sigma = np.maximum(sigma * 1.05, 1e-12)
            else:
                no_improve_streak += 1
                # If stagnating, reduce sigma to zoom in
                sigma = np.maximum(sigma * 0.70, 1e-12)

            # ---- Lightweight coordinate refinement (only when near the end or after stagnation) ----
            # To avoid wasting evaluations, do it sparingly.
            if evals < budget and (no_improve_streak >= 3 or (budget - evals) <= max(8, dim // 2)):
                remaining = budget - evals
                # Try at most two probes per coordinate batch, but limited by remaining budget.
                probe_budget = min(2 * dim, remaining)
                # If probe_budget small, sample a subset of coordinates.
                if probe_budget <= 0:
                    continue
                k = min(dim, probe_budget // 2 if probe_budget >= 2 else 1)
                if k <= 0:
                    k = 1
                # Choose coordinates—prefer those with larger width to escape flatness.
                coord_scores = width_pos.copy()
                if np.any(np.isfinite(coord_scores)):
                    idx = np.argsort(-coord_scores)[:k]
                else:
                    idx = np.random.choice(dim, size=k, replace=False)

                # Step along chosen coordinates
                coord_step = np.maximum(0.5 * sigma[idx], 1e-12)
                for j, c in enumerate(idx):
                    if evals >= budget:
                        break
                    for sgn in (-1.0, 1.0):
                        if evals >= budget:
                            break
                        x = x_best.copy()
                        x[c] = x_best[c] + sgn * coord_step[j]
                        x = clip_to_bounds(x)
                        y = eval_one(x)
                        if y < y_best:
                            x_best, y_best = x, y
                            no_improve_streak = 0
                            # Zoom slightly after success
                            sigma = np.maximum(sigma * 0.90, 1e-12)
                # After refinement, continue main loop

        return x_best, y_best
