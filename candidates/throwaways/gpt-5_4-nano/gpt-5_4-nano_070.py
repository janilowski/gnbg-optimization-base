# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm (derivative-free)
# suitable for noisy or non-smooth objectives. It maintains a small population of candidate points,
# periodically re-centers around the best point, and uses both randomized exploration and local
# refinement via coordinate-wise perturbations.
#
# Search state: The algorithm tracks a population of points, their objective values, the current
# best solution (x*, f*), and an evaluation counter to strictly respect the provided budget.
# It also maintains a "step size" controlling how far candidates are perturbed.
#
# Candidate generation: Each iteration creates new candidates by (1) sampling Gaussian
# perturbations around one or more good points, and (2) performing coordinate-wise probes
# around the current best using scaled step sizes. Proposed points are clipped to the bounds.
#
# Selection and replacement: New candidates are evaluated (counting toward the budget). If a
# candidate improves upon an existing population member, it replaces it; regardless, the global
# best is updated when any evaluated point is better.
#
# Adaptation: The step size shrinks after rounds that fail to find improvements and grows
# slightly when improvements are found, balancing exploration and exploitation over time.
#
# Exploration mechanisms: Population-based randomized perturbations (Gaussian noise with
# decreasing/increasing step size) and occasional re-initialization around the best point
# encourage covering the space.
#
# Exploitation mechanisms: Coordinate-wise local probing around the current best point
# attempts to find better nearby solutions with finer granularity.
#
# Boundary handling: All proposed points are clipped to the feasible box defined by the
# objective bounds (read from func.lower/func.upper or func.bounds.lb/func.bounds.ub).
#
# Budget strategy: Every objective call increments an internal counter. The algorithm stops
# exactly when remaining evaluations cannot fit the next planned evaluation. It never
# exceeds the provided evaluation budget.
#
# Closest known influences: The approach loosely follows ideas from Evolution Strategies and
# pattern search (local coordinate probing), combined with a simple adaptive step size and
# elitist selection.
#
# Novelty or unusual aspects: The algorithm uses a compact hybrid of population perturbations
# plus coordinate-wise probes, with strict budget gating and bound-aware clipping.
#
# Failure modes: If the objective is highly deceptive with many local minima, the shrinking
# step size may prematurely reduce exploration. If the bounds are extremely tight or scaling is
# unfavorable, clipping can make search stagnant. Also, if the budget is very small, the
# algorithm may not have enough evaluations for meaningful exploration.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ----- Read bounds robustly -----
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # If bounds are not available, fall back to something usable but safe:
            # This still clips candidates to a derived box to avoid exploding values.
            # (The prompt requests bounds; this is a robustness fallback.)
            lb = -np.ones(dim, dtype=float)
            ub = np.ones(dim, dtype=float)

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != dim or ub.size != dim:
            # If mismatch, try broadcasting scalar bounds.
            if lb.size == 1:
                lb = np.full(dim, float(lb), dtype=float)
            if ub.size == 1:
                ub = np.full(dim, float(ub), dtype=float)

        lb = np.minimum(lb, ub)
        ub = np.maximum(lb, ub)
        span = ub - lb
        # Avoid zeros: where span==0, the dimension is fixed; step should be 0 there.
        span_safe = np.where(span > 0, span, 1.0)

        rng = np.random  # harness sets numpy seed globally

        evals = 0

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_obj(x):
            nonlocal evals
            if evals >= budget:
                # Should never happen due to gating, but keep safe.
                return None
            y = func(x)
            evals += 1
            return float(y)

        # ----- Budget-aware design -----
        # Use a small population size, but never request more evaluations than budget.
        # For very small budgets, fall back to simple probing.
        min_pop = 2
        max_pop = 12
        pop_size = int(np.clip(budget // 4, min_pop, max_pop))
        pop_size = max(1, min(pop_size, budget))  # just in case

        # Step size initial: fraction of the feasible box.
        # Coordinate scale is span; for fixed dims, step will be 0.
        step = 0.5 * span_safe / np.sqrt(max(1, dim))

        # Initialize population uniformly in box; strictly budget-gated.
        pop = np.empty((pop_size, dim), dtype=float)
        vals = np.full(pop_size, np.inf, dtype=float)

        # If budget is extremely small, we can only evaluate a few points.
        n_init = min(pop_size, budget)
        for i in range(n_init):
            x = lb + rng.rand(dim) * (ub - lb)
            pop[i] = x
            y = eval_obj(x)
            vals[i] = y
        # For remaining population members (if any), set to best so far (no extra eval).
        best_idx = int(np.argmin(vals[:n_init])) if n_init > 0 else 0
        best_x = pop[best_idx].copy() if n_init > 0 else lb.copy()
        best_y = float(vals[best_idx]) if n_init > 0 else float(eval_obj(best_x)) if budget > 0 else np.inf

        if evals < budget and n_init == 0:
            # Ensure at least one evaluation when possible.
            best_x = lb + rng.rand(dim) * (ub - lb)
            best_y = eval_obj(best_x)

        # Elitist reference points: keep a small set from population for candidate centers.
        elite_k = int(min(3, max(1, n_init)))
        # Adaptation counters
        no_improve_rounds = 0

        # Helper to choose elites by current vals among initialized part.
        def get_elites():
            k = min(elite_k, pop_size)
            order = np.argsort(vals)
            return order[:k]

        # ----- Main loop: each iteration performs a few evaluations -----
        # We stop when no further evaluations can be made.
        # Within each round, we attempt:
        #  - A couple of global perturbations around elites.
        #  - A coordinate-wise local probe around current best.
        round_no = 0
        while evals < budget:
            round_no += 1
            remaining = budget - evals
            if remaining <= 0:
                break

            improved_this_round = False

            # Choose dynamic batch size (minimizes overhead and respects budget).
            # Global perturbations batch
            global_batch = min(remaining, 2 + (1 if dim >= 8 else 0))
            if global_batch < 1:
                break

            # Determine candidate centers (elites)
            elite_ids = get_elites()
            centers = pop[elite_ids] if elite_ids.size > 0 else best_x.reshape(1, -1)

            # Evaluate candidates generated from Gaussian perturbations
            for _ in range(global_batch):
                if evals >= budget:
                    break

                # Pick a center among elites
                c = centers[rng.randint(0, centers.shape[0])]
                # Gaussian perturbation scaled by current step; include anisotropic scaling via span.
                noise = rng.randn(dim)
                x_new = c + step * (noise * (span_safe / span_safe))  # keep scale in step
                x_new = clip(x_new)

                y_new = eval_obj(x_new)
                if y_new is None:
                    break

                # Update global best
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new.copy()
                    improved_this_round = True

                # Replacement: if better than some population member, replace the worst
                worst_idx = int(np.argmax(vals))
                if y_new < vals[worst_idx]:
                    pop[worst_idx] = x_new
                    vals[worst_idx] = y_new

            if evals >= budget:
                break

            # Coordinate-wise probes around best: pick a few coordinates to limit evaluations.
            # For larger dims, probe a subset each round.
            remaining = budget - evals
            coord_budget = min(remaining, max(1, dim // 5))
            # Ensure at least one coord probe when possible, but never exceed remaining.
            if coord_budget > 0:
                # Choose coordinate indices: mix systematic and random
                if dim <= 6:
                    coord_idxs = np.arange(dim)
                    coord_idxs = coord_idxs[:coord_budget]
                else:
                    # Slightly bias towards random coordinates; also include a few deterministic ones.
                    r = min(coord_budget, dim)
                    coord_idxs = rng.choice(dim, size=r, replace=False)

                # Probe +/- along each selected coordinate
                for j in np.atleast_1d(coord_idxs):
                    if evals >= budget:
                        break
                    # Local direction along coordinate j
                    # Use smaller step for coordinate probing to refine.
                    dj = 0.5 * step[j] if step[j] != 0 else 0.0
                    if dj == 0.0:
                        continue

                    for sgn in (+1.0, -1.0):
                        if evals >= budget:
                            break
                        x_new = best_x.copy()
                        x_new[j] = x_new[j] + sgn * dj
                        x_new = clip(x_new)

                        y_new = eval_obj(x_new)
                        if y_new is None:
                            break

                        if y_new < best_y:
                            best_y = y_new
                            best_x = x_new.copy()
                            improved_this_round = True

                        worst_idx = int(np.argmax(vals))
                        if y_new < vals[worst_idx]:
                            pop[worst_idx] = x_new
                            vals[worst_idx] = y_new

            # Adapt step size based on improvement
            if improved_this_round:
                no_improve_rounds = 0
                # Slightly increase step to escape plateaus, but cap to prevent explosions.
                step = np.minimum(step * 1.05 + 1e-12, 0.9 * span_safe / np.sqrt(max(1, dim)))
            else:
                no_improve_rounds += 1
                # Shrink step when no improvements happen.
                step = step * (0.85 ** max(1, no_improve_rounds))

            # If step becomes tiny, occasionally re-inject exploration around best.
            if no_improve_rounds >= 5 and evals < budget:
                # Recenter population around best with small random noise (no extra eval yet; will eval next loop).
                # This helps avoid stagnation while staying within budget naturally.
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    # Replace only positions; objective will be re-evaluated when generating candidates.
                    pop[i] = clip(best_x + 0.2 * step * rng.randn(dim))

                # Reset counter slightly to allow progress.
                no_improve_rounds = 0

        return best_x, best_y
