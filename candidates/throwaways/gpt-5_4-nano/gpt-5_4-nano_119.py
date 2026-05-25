# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, black-box, derivative-free minimization algorithm
# that combines random exploration, coordinate-wise local improvement, and periodic
# restarts. It maintains a small population of candidate points and repeatedly
# refines the best solutions using adaptive step sizes while ensuring the total
# number of objective evaluations never exceeds the given budget.
# Search state: Tracks a population of points, their objective values, the current
# best (best_x, best_y), and an adaptive scalar step size sigma. Also tracks
# how many evaluations have been used.
# Candidate generation: Uses a mix of (1) Gaussian perturbations around the
# current best with scale sigma, (2) population-based mutation using other
# individuals as additional anchors, and (3) random coordinate perturbations
# to encourage axis-aligned improvements in any dimension.
# Selection and replacement: After evaluating a batch of candidates, keeps the
# best individuals (elitist selection) and replaces the remaining population with
# newly generated candidates. The global best is updated whenever a better point is found.
# Adaptation: Sigma decays when improvements are found (finer exploitation) and
# inflates slightly when improvements stall (broader exploration).
# Exploration mechanisms: Random perturbations with relatively large sigma early,
# occasional restarts by reinitializing part of the population, and coordinate
# moves with random step signs.
# Exploitation mechanisms: Gaussian sampling centered on the current best and
# repeated coordinate-wise local refinements using one-dimensional step attempts.
# Boundary handling: All points are clipped to the provided box constraints.
# If bounds are degenerate in a dimension, the algorithm keeps that coordinate fixed.
# Budget strategy: The number of evaluations per loop is computed from the remaining
# budget. Total evaluations are capped strictly by the budget; the algorithm stops
# early if it would exceed it.
# Closest known influences: A lightweight blend of Evolution Strategies (elitist
# selection and Gaussian mutations) with coordinate search for local refinement.
# Novelty or unusual aspects: The algorithm uses a hybrid of population sampling and
# deterministic-ish coordinate refinement while keeping the implementation short,
# dimension-agnostic, and budget-safe.
# Failure modes: If the objective is extremely noisy, strict elitism may
# overfit to spurious minima; if bounds are very tight or degenerate, progress
# may stall due to clipping; in deceptive landscapes it may get trapped, though
# sigma inflation and partial restarts mitigate this.
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
            # No evaluations possible: return a deterministic in-bounds point if possible
            lb, ub = _get_bounds(func, dim)
            x0 = _midpoint(lb, ub)
            return x0, float("inf")

        lb, ub = _get_bounds(func, dim)
        widths = ub - lb
        deg_mask = widths <= 0.0

        # Safe initial point: mid of bounds or zeros if unbounded (shouldn't happen for valid tasks)
        x0 = _midpoint(lb, ub)

        # Population size: keep it small and scale with dimension but capped for budget efficiency
        # Aim for at most ~max(4, 10) individuals; ensure at least 2 for diversity.
        pop_size = int(min(max(4, 2 * dim), 12))
        pop_size = min(pop_size, budget)  # cannot evaluate more than budget
        pop_size = max(2, pop_size)

        # Track evaluation budget precisely
        evals = 0

        # Evaluate helper
        def eval_point(x):
            nonlocal evals
            if evals >= budget:
                # Should never happen if caller respects budget.
                return float("inf")
            y = float(func(np.asarray(x, dtype=float)))
            evals += 1
            return y

        # Initialize population: include x0 and random samples in bounds
        pop = np.zeros((pop_size, dim), dtype=float)
        pop[0] = x0
        # Random initialization around x0 + uniform jitter within bounds
        for i in range(1, pop_size):
            r = np.random.rand(dim)
            pop[i] = lb + r * widths
            # Handle degenerate bounds: coordinate fixed
            if np.any(deg_mask):
                pop[i, deg_mask] = lb[deg_mask]

        vals = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            if evals >= budget:
                break
            vals[i] = eval_point(pop[i])

        # If budget allowed fewer than pop_size evaluations (very small budget), truncate
        if evals < pop_size:
            # Find best among evaluated individuals
            idx = int(np.argmin(vals[:evals])) if evals > 0 else 0
            best_x = pop[idx].copy()
            best_y = float(vals[idx]) if evals > 0 else float("inf")
            return best_x, best_y

        best_idx = int(np.argmin(vals))
        best_x = pop[best_idx].copy()
        best_y = float(vals[best_idx])

        # Initialize step size sigma based on typical bound scale; avoid zero
        scale = np.mean(np.maximum(widths, 0.0))
        sigma = 0.3 * scale / np.sqrt(max(1, dim))
        if sigma <= 0:
            sigma = 1.0

        # For stalling detection
        no_improve_count = 0
        last_best_y = best_y

        # Main loop: each iteration samples and refines while respecting budget
        # Use "generations" with variable evaluation count based on remaining budget.
        # We stop when budget is exhausted.
        while evals < budget:
            remaining = budget - evals
            # Reserve some evaluations for coordinate refinement when possible
            # Candidate batch size: at least 1, at most pop_size and remaining.
            batch = min(pop_size, remaining)

            # Generate candidates with a hybrid scheme
            cand = np.empty((batch, dim), dtype=float)

            # Mix weights: exploration vs exploitation
            # More exploitation near the end.
            t = evals / max(1, budget)
            exploit_prob = 0.55 + 0.35 * (1.0 - t)  # higher earlier; decays slightly

            # Mutation anchors
            # Choose random indices from population for diversity
            for j in range(batch):
                if np.random.rand() < exploit_prob:
                    # Exploit: Gaussian around best
                    z = np.random.randn(dim)
                    x = best_x + sigma * z
                else:
                    # Explore: Gaussian around a random population member + slight best pull
                    k = np.random.randint(pop_size)
                    z = np.random.randn(dim)
                    pull = 0.15 * (best_x - pop[k])
                    x = pop[k] + sigma * z + pull

                # Occasionally do a coordinate move (axis-aligned)
                if np.random.rand() < 0.25:
                    c = np.random.randint(dim)
                    step = sigma * (0.5 + np.random.rand())
                    sign = 1.0 if np.random.rand() < 0.5 else -1.0
                    x = x.copy()
                    x[c] = x[c] + sign * step

                # Boundary handling by clipping
                if np.any(deg_mask):
                    # Clip non-degenerate dims, fix degenerate dims
                    x[deg_mask] = lb[deg_mask]
                    # For degenerate dims widths==0, lb==ub and clipping would already fix,
                    # but this makes it explicit.
                    pass
                x = np.clip(x, lb, ub)
                cand[j] = x

            # Evaluate candidates
            cand_vals = np.empty(batch, dtype=float)
            for j in range(batch):
                if evals >= budget:
                    cand_vals[j] = float("inf")
                    continue
                cand_vals[j] = eval_point(cand[j])

            # Combine and elitist selection: keep best pop_size points
            combined = np.vstack([pop, cand]) if cand.shape[0] > 0 else pop
            combined_vals = np.concatenate([vals, cand_vals]) if cand.shape[0] > 0 else vals
            # Keep smallest pop_size
            order = np.argsort(combined_vals, kind="stable")
            keep = order[:pop_size]
            pop = combined[keep].copy()
            vals = combined_vals[keep].copy()

            # Update global best
            current_best_idx = int(np.argmin(vals))
            current_best_y = float(vals[current_best_idx])
            current_best_x = pop[current_best_idx].copy()
            if current_best_y < best_y:
                best_y = current_best_y
                best_x = current_best_x
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Adapt sigma based on improvement
            if best_y < last_best_y:
                sigma *= 0.85
            else:
                sigma *= 1.03

            # Prevent sigma from becoming too small or too large relative to bounds
            sigma_floor = 1e-12 + 1e-6 * scale
            sigma_ceiling = (widths.max() + 1e-6) * 2.0 if np.any(widths > 0) else 1e3
            sigma = float(np.clip(sigma, sigma_floor, sigma_ceiling))
            last_best_y = best_y

            # Occasional coordinate refinement around the best point
            # Use a small number of evaluations if budget allows.
            # This is cheap and can yield fast local improvements.
            if evals < budget and no_improve_count >= 2:
                # Try a few coordinates; keep count small to remain budget-safe.
                remaining = budget - evals
                # Try up to 1 + dim//2 steps but not exceeding remaining.
                k_tries = min(1 + dim // 2, remaining)
                if k_tries > 0:
                    # Shuffle coordinate order for robustness
                    coords = np.random.permutation(dim)[:k_tries]
                    # One-sided and opposite attempts
                    for c in coords:
                        if evals >= budget:
                            break
                        width_c = widths[c] if widths.size > c else 0.0
                        # Use a step relative to sigma but also relative to coordinate range
                        step_mag = sigma * (0.4 + 0.6 * np.random.rand())
                        if width_c > 0:
                            step_mag = min(step_mag, 0.5 * width_c)

                        # Attempt +step and -step (budget-aware: attempt only one if tight)
                        for sign in (1.0, -1.0):
                            if evals >= budget:
                                break
                            if deg_mask[c]:
                                continue
                            x_try = best_x.copy()
                            x_try[c] = np.clip(x_try[c] + sign * step_mag, lb[c], ub[c])
                            # If clipping makes no change, skip
                            if x_try[c] == best_x[c]:
                                continue
                            y_try = eval_point(x_try)
                            if y_try < best_y:
                                best_y = y_try
                                best_x = x_try
                                no_improve_count = 0
                                # After improvement, reduce sigma a bit
                                sigma *= 0.9

            # Partial restart to avoid stagnation
            if no_improve_count >= 5 and evals < budget:
                # Restart a portion of the population, centered around x0 and best_x
                # but keep best individual to preserve progress.
                keep_best = 1
                # Number to restart
                restart_n = min(pop_size - keep_best, max(1, pop_size // 3))
                if restart_n > 0:
                    # Indices to restart (excluding best)
                    # Ensure we don't exceed budget (no evaluations here, just propose points).
                    # We will evaluate restarted points by replacing worst and sampling.
                    # For simplicity under budget constraints, we evaluate up to restart_n candidates.
                    remaining = budget - evals
                    eval_n = min(restart_n, remaining)
                    if eval_n > 0:
                        # Replace worst individuals gradually with new samples
                        worst_order = np.argsort(vals, kind="stable")[::-1]
                        replace_idx = worst_order[:eval_n]
                        for rpos in replace_idx:
                            r = np.random.rand(dim)
                            # Blend between uniform random and best neighborhood
                            if np.random.rand() < 0.5:
                                x_new = lb + r * widths
                            else:
                                x_new = best_x + sigma * np.random.randn(dim)
                            x_new = np.clip(x_new, lb, ub)
                            if np.any(deg_mask):
                                x_new[deg_mask] = lb[deg_mask]
                            pop[rpos] = x_new
                            vals[rpos] = eval_point(x_new)

                        current_best_idx = int(np.argmin(vals))
                        current_best_y = float(vals[current_best_idx])
                        if current_best_y < best_y:
                            best_y = current_best_y
                            best_x = pop[current_best_idx].copy()
                        # After restart, increase sigma to encourage exploration
                        sigma *= 1.2
                        no_improve_count = 0

        return best_x, best_y


def _get_bounds(func, dim):
    # Accept either func.lower/func.upper or func.bounds.lb/func.bounds.ub
    lb = None
    ub = None

    if hasattr(func, "lower") and hasattr(func, "upper"):
        lb = np.asarray(getattr(func, "lower"), dtype=float)
        ub = np.asarray(getattr(func, "upper"), dtype=float)
    elif hasattr(func, "bounds"):
        b = getattr(func, "bounds")
        if hasattr(b, "lb") and hasattr(b, "ub"):
            lb = np.asarray(getattr(b, "lb"), dtype=float)
            ub = np.asarray(getattr(b, "ub"), dtype=float)

    if lb is None or ub is None:
        # Fallback (should not happen in benchmark); use [-5, 5] hypercube
        lb = np.full(dim, -5.0, dtype=float)
        ub = np.full(dim, 5.0, dtype=float)

    if lb.shape[0] != dim or ub.shape[0] != dim:
        # Robustness: attempt to broadcast/reshape
        lb = np.resize(lb, (dim,)).astype(float)
        ub = np.resize(ub, (dim,)).astype(float)

    return lb, ub


def _midpoint(lb, ub):
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    return (lb + ub) * 0.5
