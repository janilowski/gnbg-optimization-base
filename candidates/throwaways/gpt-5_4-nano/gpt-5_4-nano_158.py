# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy using a
# derivative-free evolutionary loop with adaptive sampling and a local
# coordinate-search refinement.
# Search state: Maintains the current best point (x_best, y_best), a
# population of candidate points, and a global step-size (sigma) that adapts
# based on recent improvements.
# Candidate generation: Each iteration creates offspring by sampling from a
# Gaussian around the current best with two directional components:
# (1) global isotropic perturbations and (2) coordinate-wise moves using a
# per-dimension random sign. Additionally, after a few stagnant iterations,
# it triggers a local refinement using small step sizes along coordinates.
# Selection and replacement: Offspring are evaluated (minimization). Any
# offspring that improves the best replaces it immediately. A simple
# population update keeps the best individuals for diversity.
# Adaptation: If improvements occur, sigma increases slightly (to explore);
# if not, sigma decreases (to exploit). The local refinement step also shrinks
# when it fails to improve.
# Exploration mechanisms: Global Gaussian perturbations around the best and
# occasional wider moves when sigma is relatively large.
# Exploitation mechanisms: Local coordinate searches around the best using
# decreasing step sizes to fine-tune.
# Boundary handling: Candidate points are clamped to the provided bounds.
# If bounds are degenerate (lb==ub), those coordinates are held fixed.
# Budget strategy: Never evaluates more than the provided evaluation budget.
# The number of offspring per iteration is adjusted to ensure the total
# evaluations stays within budget.
# Closest known influences: Combines ideas from CMA-ES-style success adaptation
# (but simplified), evolution strategies with selection, and coordinate-based
# local search.
# Novelty or unusual aspects: Uses a hybrid of population-based global
# sampling and periodic coordinate refinement with budget-aware evaluation
# counting.
# Failure modes: Can stagnate on flat functions or highly discontinuous
# landscapes; if bounds are extremely tight it may effectively reduce to a
# near-deterministic search. If the objective noise is high, sigma adaptation
# may oscillate.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            x0 = np.zeros(dim, dtype=float)
            return x0, float(func(x0))

        # --- Bounds handling (minimization) ---
        # Prefer func.lower/upper; fallback to func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            b = getattr(func, "bounds")
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)

        if lb.shape[0] != dim:
            lb = np.broadcast_to(lb, (dim,)).astype(float, copy=False)
        if ub.shape[0] != dim:
            ub = np.broadcast_to(ub, (dim,)).astype(float, copy=False)

        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        # Handle degenerate bounds by fixing those coordinates.
        span = ub - lb
        fixed = span <= 0
        span_safe = np.where(fixed, 1.0, span)  # avoid division by zero

        rng = np.random

        # --- Evaluation counter with hard budget enforcement ---
        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                # Should never happen with careful planning; keep safe.
                return float("inf")
            # Enforce bounds via clamping
            if np.any(fixed):
                x = np.asarray(x, dtype=float)
                x = np.minimum(np.maximum(x, lb), ub)
                x = np.where(fixed, lb, x)
            else:
                x = np.asarray(x, dtype=float)
                x = np.minimum(np.maximum(x, lb), ub)
            y = float(func(x))
            evals += 1
            return y

        # --- Initialize sigma based on bounds span ---
        # Use a fraction of the average span; if span is tiny, keep sigma tiny.
        avg_span = float(np.mean(np.abs(span_safe)))
        sigma = 0.3 * avg_span / max(1.0, float(dim) ** 0.5)
        sigma = max(sigma, 1e-12)

        # --- Create initial population, including the midpoint ---
        x_mid = np.where(fixed, lb, (lb + ub) / 2.0)
        y_best = evaluate(x_mid)
        x_best = x_mid.copy()

        # Population size: small for speed, but at least 4 when budget allows.
        pop_size = int(min(10, max(4, budget // 20 + 4)))
        pop_size = min(pop_size, max(1, budget - 1))  # leave room for best already evaluated

        population = [x_best.copy()]
        pop_scores = [y_best]

        remaining = budget - evals
        # Fill population with random points (budget-aware).
        while len(population) < pop_size and remaining > 0:
            r = rng.random(dim)
            x = lb + r * (ub - lb)
            if np.any(fixed):
                x = np.where(fixed, lb, x)
            y = evaluate(x)
            population.append(x.copy())
            pop_scores.append(y)
            remaining = budget - evals

        # Keep population sorted by score
        order = np.argsort(np.asarray(pop_scores))
        population = [population[i] for i in order[: len(population)]]
        pop_scores = [pop_scores[i] for i in order[: len(pop_scores)]]

        # --- Main loop ---
        # Iteration budget: adapt number of offspring per iteration.
        no_improve_streak = 0
        stagnation_limit = 8
        # Local refinement parameters
        local_shrink = 0.5
        local_trials = 0

        while evals < budget:
            # Determine how many evaluations to spend this iteration.
            # Use 1..k offspring depending on remaining budget.
            remaining = budget - evals
            if remaining <= 0:
                break

            # Offspring count: depend on dim and remaining budget.
            k = int(min(12, max(1, budget // 30)))
            k = int(min(k, remaining))
            if k <= 0:
                break

            # Choose a parent center: best point with slight chance of using another elite.
            center = x_best
            if len(population) > 1 and rng.rand() < 0.25:
                center = population[int(rng.randint(0, len(population)))]

            # Create offspring candidates.
            offspring = []
            for _ in range(k):
                # 1) Gaussian perturbation around center
                g = rng.normal(size=dim)
                x = center + sigma * g

                # 2) Occasional coordinate-wise step to encourage anisotropy
                if dim >= 2 and rng.rand() < 0.5:
                    idx = int(rng.randint(0, dim))
                    sign = 1.0 if rng.rand() < 0.5 else -1.0
                    coord_step = sigma * (0.5 + rng.rand()) * sign
                    x[idx] = center[idx] + coord_step

                offspring.append(x)

            # Evaluate offspring and update best immediately.
            improved = False
            new_points = []
            new_scores = []

            for x in offspring:
                y = evaluate(x)
                new_points.append(x.copy())
                new_scores.append(y)
                if y < y_best:
                    y_best = y
                    x_best = np.asarray(x, dtype=float).copy()
                    improved = True

            # Update adaptation
            if improved:
                no_improve_streak = 0
                sigma *= 1.08  # slight expansion after success
            else:
                no_improve_streak += 1
                sigma *= 0.82  # contract after failure
            sigma = max(sigma, 1e-12 * avg_span / max(1.0, float(dim) ** 0.5))

            # Population update: merge and keep best elites for diversity.
            # (Budget-safe: uses already-evaluated points only.)
            merged_points = population + new_points
            merged_scores = pop_scores + new_scores
            order = np.argsort(np.asarray(merged_scores))

            # Keep a small elite set
            elite = min(len(merged_points), max(3, pop_size // 2 + 1))
            population = [merged_points[i] for i in order[:elite]]
            pop_scores = [merged_scores[i] for i in order[:elite]]

            # --- Local coordinate refinement when stagnating ---
            if no_improve_streak >= stagnation_limit and evals < budget:
                # Limit local work by remaining budget.
                remaining = budget - evals
                # Use at most a few evaluations for local refinement.
                # Also, stop if we already improved during this iteration (usually won't happen).
                budget_local = min(4 + dim // 2, remaining)
                if budget_local <= 0:
                    continue

                # Start with a smaller step for local search.
                local_sigma = sigma * 0.35
                improved_local = False

                # Choose an order of coordinates based on current best closeness to bounds.
                # Coordinates that have room (not near bounds) get priority.
                room = np.minimum(np.abs(x_best - lb), np.abs(ub - x_best))
                # Larger room => earlier in the list (more room to move).
                coord_order = np.argsort(-room)

                # Each trial tries one coordinate in either +/- direction.
                trials_used = 0
                while trials_used < budget_local:
                    if len(coord_order) == 0:
                        break
                    idx = int(coord_order[trials_used % len(coord_order)])
                    # Skip fixed coords
                    if fixed[idx]:
                        trials_used += 1
                        continue

                    # Try both directions in a "budget aware" way:
                    # If budget_local is larger, do two evaluations; else do one.
                    if budget_local - trials_used >= 2:
                        x1 = x_best.copy()
                        x2 = x_best.copy()
                        x1[idx] = x_best[idx] + local_sigma
                        x2[idx] = x_best[idx] - local_sigma
                        y1 = evaluate(x1)
                        y2 = evaluate(x2)
                        trials_used += 2
                        if y1 < y_best:
                            x_best, y_best = x1, y1
                            improved_local = True
                        if y2 < y_best:
                            x_best, y_best = x2, y2
                            improved_local = True
                    else:
                        # Only one evaluation due to tight remaining budget
                        sign = 1.0 if rng.rand() < 0.5 else -1.0
                        x1 = x_best.copy()
                        x1[idx] = x_best[idx] + sign * local_sigma
                        y1 = evaluate(x1)
                        trials_used += 1
                        if y1 < y_best:
                            x_best, y_best = x1, y1
                            improved_local = True

                    if improved_local:
                        # Optionally reduce local_sigma after an improvement to focus.
                        local_sigma *= local_shrink

                    # If local search consumes budget, exit.
                    if evals >= budget:
                        break

                if improved_local:
                    no_improve_streak = 0
                    sigma *= 1.02
                    local_trials += 1
                else:
                    no_improve_streak = max(0, no_improve_streak - 2)
                    sigma *= local_shrink
                    local_trials += 1

        return x_best, y_best
