# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# (derivative-free) that works with only function evaluations. It combines a
# population-based evolutionary strategy with local coordinate/gradient-free
# probing. The algorithm tracks the best point found and maintains a mutation
# scale that adapts to progress.
# Search state: The algorithm maintains a current best solution (x_best,
# y_best), a population of candidate solutions, and a step size (sigma) used
# to generate mutations. It also keeps counters to enforce the exact evaluation
# budget.
# Candidate generation: Each iteration samples new candidates by mutating
# points from the current elite set using Gaussian noise scaled by sigma.
# Additionally, it performs lightweight 1D “probing” around the best point by
# trying small positive/negative moves along randomly chosen coordinates to
# discover a better direction.
# Selection and replacement: After evaluating all candidates in an iteration,
# the best solutions are selected (elitism). The rest of the population is
# replaced by new offspring around the elite set.
# Adaptation: The mutation scale sigma is adapted based on whether recent
# improvements were found, using success/failure style logic to increase or
# decrease sigma.
# Exploration mechanisms: Population mutations (diverse sampling) and the
# randomized coordinate probing add exploration, especially early on or when
# progress stalls.
# Exploitation mechanisms: The coordinate probes and sampling centered on the
# current elite/best points focus search locally when improvements are present.
# Boundary handling: All mutated points are clamped to the provided bounds.
# Bounds are read from func.lower/func.upper or from func.bounds.lb/ub.
# Budget strategy: Every function call consumes evaluations; the algorithm
# always stops when it would exceed the provided evaluation budget.
# Closest known influences: The design loosely resembles a (μ+λ)-style evolution
# strategy with adaptive step size, augmented by deterministic coordinate probing.
# Novelty or unusual aspects: The coordinate probing is scheduled adaptively based
# on remaining budget and recent success, which helps robustness across
# dimensions without requiring gradients.
# Failure modes: If the objective is extremely noisy or highly irregular, local
# probing may mislead adaptation. The clamping can also cause premature
# stagnation near bounds, though the population mutations and sigma adaptation
# mitigate this.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from function or its bounds object ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(
            func.bounds, "ub"
        ):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # If bounds are not provided, fall back to a wide box.
            lb = -5.0 * np.ones(dim, dtype=float)
            ub = 5.0 * np.ones(dim, dtype=float)

        lb = np.broadcast_to(lb, (dim,)).copy()
        ub = np.broadcast_to(ub, (dim,)).copy()
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            # Guard against pathological infinities.
            lb = np.where(np.isfinite(lb), lb, -5.0)
            ub = np.where(np.isfinite(ub), ub, 5.0)

        # Ensure valid bounds
        tighter_lb = np.minimum(lb, ub)
        tighter_ub = np.maximum(lb, ub)
        lb, ub = tighter_lb, tighter_ub

        span = ub - lb
        span = np.where(span > 0, span, 1.0)  # avoid zero span issues
        # Provide a starting sigma relative to box size
        sigma = 0.25 * np.min(span) if np.min(span) > 0 else 0.25

        # ---- Budget accounting ----
        evals = 0

        def eval_once(x):
            nonlocal evals
            if evals >= budget:
                # Never exceed budget; return the current best if forced.
                return float("inf")
            y = float(func(x))
            evals += 1
            return y

        def clamp(x):
            return np.minimum(np.maximum(x, lb), ub)

        # ---- Initialize population ----
        # Pop size chosen to balance cost and diversity; cap to budget.
        # Ensure at least 4 evaluations for initial spread when budget allows.
        pop = int(min(max(6, 2 * dim + 2), max(4, budget)))
        # If budget too small, use smaller pop and fewer iterations.
        pop = min(pop, max(2, budget))

        rng = np.random

        # Sample initial points uniformly in bounds
        X = rng.uniform(0.0, 1.0, size=(pop, dim)) * (ub - lb) + lb
        X = np.asarray([clamp(x) for x in X], dtype=float)

        ys = np.empty(pop, dtype=float)
        for i in range(pop):
            if evals >= budget:
                break
            ys[i] = eval_once(X[i])

        # If budget extremely small, return best among evaluated
        if evals >= budget:
            idx = int(np.argmin(ys[: evals]))
            return X[idx].copy(), float(ys[idx])

        # Track global best
        best_idx = int(np.argmin(ys[:pop]))
        x_best = X[best_idx].copy()
        y_best = float(ys[best_idx])

        # ---- Main loop ----
        # Remaining budget controls number of iterations.
        # Each iteration evaluates a batch of candidates; compute how many can fit.
        # Use elite fraction.
        elite_frac = 0.25
        elite_n = max(1, int(np.ceil(pop * elite_frac)))

        # Candidate batch size per iteration
        # Keep it moderate to allow multiple adaptation steps.
        batch = min(pop, max(2, int(np.ceil(pop / 2))))
        batch = min(batch, max(1, budget - evals))

        # Success tracking for sigma adaptation
        consecutive_success = 0
        consecutive_fail = 0

        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break

            # Adjust batch to not exceed budget
            cur_batch = min(batch, remaining)
            # Recompute elite from current population (use evaluated subset only)
            # If fewer than pop were evaluated due to early stop, slice ys/X accordingly.
            # (In normal flow, pop candidates were evaluated.)
            current_pop_n = pop
            elites_idx = np.argsort(ys[:current_pop_n])[:elite_n]
            elites = X[elites_idx]

            # --- Exploration/exploitation: generate offspring around elites ---
            # Offspring = elite + N(0, sigma^2) + occasional differential mutation.
            # Differential mutation uses another elite for direction, improving robustness.
            cand = np.empty((cur_batch, dim), dtype=float)
            for j in range(cur_batch):
                center = elites[rng.randint(0, elite_n)]
                # Random mask for coordinate-wise perturbation (sparsify mutation)
                mask_prob = 0.5 if dim <= 10 else 0.25
                mask = rng.random(dim) < mask_prob
                if not mask.any():
                    mask[rng.randint(0, dim)] = True

                step = rng.normal(0.0, 1.0, size=dim)
                z = np.where(mask, step, 0.0)

                # Differential component with small probability
                if elite_n > 1 and rng.rand() < 0.35:
                    a = elites[rng.randint(0, elite_n)]
                    b = elites[rng.randint(0, elite_n)]
                    diff = a - b
                    # Scale differential by span and sigma to keep units consistent
                    diff_scale = 0.1 * sigma / (0.25 * np.min(span)) if np.min(span) > 0 else 0.1
                    proposal = center + z * sigma + diff_scale * diff
                else:
                    proposal = center + z * sigma

                cand[j] = clamp(proposal)

            # --- Lightweight coordinate probing around best ---
            # Use a few probes when beneficial: on progress, or with some probability.
            # Each probe consumes exactly one evaluation.
            probes = 0
            probe_budget = min(3, budget - evals)  # hard cap per iteration
            # Schedule probes more often when close to bounds or after failures.
            do_probe = (consecutive_success > 0) or (consecutive_fail >= 1) or (rng.rand() < 0.25)
            if do_probe and probe_budget > 0:
                # Choose number of probes based on remaining budget and dimension
                probes = int(min(probe_budget, 1 + (dim // 10)))
                # Ensure we don't exceed remaining evaluations
                probes = min(probes, remaining - cur_batch) if remaining - cur_batch > 0 else min(
                    probes, remaining
                )
                probes = max(0, probes)

            # Evaluate offspring (batch)
            cand_y = np.empty(cur_batch, dtype=float)
            for j in range(cur_batch):
                if evals >= budget:
                    cand_y = cand_y[:j]
                    cand = cand[:j]
                    break
                cand_y[j] = eval_once(cand[j])

            # Evaluate probes (optional)
            probe_x = []
            probe_y = []
            for _ in range(probes):
                if evals >= budget:
                    break
                # pick random coordinate and probe both sides
                k = rng.randint(0, dim)
                # probe step relative to span, reduced with decreasing sigma
                step = (0.5 * sigma / (0.25 * np.min(span)) if np.min(span) > 0 else 0.5) * (
                    0.1 + 0.4 * rng.rand()
                )
                dx = step * (span[k] if span[k] > 0 else 1.0)

                # Try both signs but only consume at most 1 eval at a time:
                # choose sign based on which seems more promising from clamped position.
                x1 = x_best.copy()
                x2 = x_best.copy()
                x1[k] = clamp(x1)[k] + dx
                x2[k] = clamp(x2)[k] - dx
                x1 = clamp(x1)
                x2 = clamp(x2)

                # Heuristic: if x1 is closer to upper than lower, prefer that sign
                # (arbitrary but stable). Still evaluates one point per probe.
                if (x1[k] - ub[k]) ** 2 < (x2[k] - lb[k]) ** 2:
                    y_probe = eval_once(x1)
                    probe_x.append(x1)
                    probe_y.append(y_probe)
                else:
                    y_probe = eval_once(x2)
                    probe_x.append(x2)
                    probe_y.append(y_probe)

            # Combine into new evaluated set for adaptation/selection
            # Update population by replacing worst individuals with new candidates.
            new_points = cand
            new_values = cand_y

            if probe_x:
                new_points = np.vstack([new_points, np.asarray(probe_x, dtype=float)])
                new_values = np.concatenate([new_values, np.asarray(probe_y, dtype=float)])

            # Determine improvements
            local_best_idx = int(np.argmin(new_values))
            local_best_y = float(new_values[local_best_idx])
            local_best_x = new_points[local_best_idx].copy()

            if local_best_y < y_best - 1e-15:
                y_best = local_best_y
                x_best = local_best_x
                consecutive_success += 1
                consecutive_fail = 0
                # Make sigma smaller upon success for exploitation
                sigma *= 0.85
            else:
                consecutive_fail += 1
                consecutive_success = 0
                # If stuck, increase sigma for exploration
                sigma *= 1.08

            # Keep sigma within reasonable limits based on bounds
            min_sigma = 1e-12
            max_sigma = 0.5 * np.max(span) if np.max(span) > 0 else 1.0
            sigma = float(np.clip(sigma, min_sigma, max_sigma))

            # Update population
            # Replace worst in current population with best of new candidates.
            # If population size differs due to tiny budget, handle carefully.
            replace_n = min(len(new_values), current_pop_n)
            # Select the best replace_n among new points
            order_new = np.argsort(new_values)[:replace_n]
            best_new_pts = new_points[order_new]
            best_new_vals = new_values[order_new]

            # Identify worst indices in existing population
            worst_idx = np.argsort(ys[:current_pop_n])[::-1][:replace_n]
            X[worst_idx] = best_new_pts
            ys[worst_idx] = best_new_vals

            # If no progress and failures persist, intensify probing occasionally.
            if consecutive_fail >= 3:
                # Slightly enlarge sigma to escape local traps
                sigma *= 1.15
                consecutive_fail = 0
                consecutive_success = 0

            # Re-evaluate global best from population to be safe
            curr_best_i = int(np.argmin(ys[:current_pop_n]))
            if float(ys[curr_best_i]) < y_best:
                y_best = float(ys[curr_best_i])
                x_best = X[curr_best_i].copy()

            # Stop if budget exactly used
            if evals >= budget:
                break

        return x_best, y_best
