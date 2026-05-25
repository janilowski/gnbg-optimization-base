# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm for
# a bounded continuous search space. It maintains a small population of candidate
# solutions and iteratively improves them by combining local refinement around
# the current best and global diversification using random direction proposals.
#
# Search state: The algorithm tracks a population of points, their objective
# values, the global best solution found so far, and a per-run evaluation
# counter to ensure the budget is never exceeded.
#
# Candidate generation: Each iteration generates new candidates by:
# 1) Local moves: taking the best point and adding a scaled random perturbation
#    (with occasional Gaussian steps) whose scale shrinks over time.
# 2) Global moves: sampling uniformly within bounds to escape stagnation.
# Candidate creation uses clipping to enforce bounds.
#
# Selection and replacement: After evaluating candidates, the algorithm applies a
# greedy replacement strategy: any new candidate that improves on the worst
# population member replaces it. This keeps quality increasing without losing the
# best-so-far solution.
#
# Adaptation: A step-size parameter starts relatively large and is reduced as
# the evaluation budget is consumed. If progress stalls, the algorithm increases
# random exploration (by raising the probability of global uniform sampling).
#
# Exploration mechanisms: Uniform random sampling within bounds and random
# direction steps around the current best.
#
# Exploitation mechanisms: Shrinking local perturbations around the current best
# plus covariance-free "coordinate-like" random direction sampling.
#
# Boundary handling: All candidates are clipped to [lb, ub]. If bounds are
# degenerate (lb==ub), the candidate coordinate is fixed accordingly.
#
# Budget strategy: The algorithm computes how many initial evaluations are needed
# for the population and then runs a loop generating/evaluating as many candidates
# as possible until the budget is exhausted. It strictly never calls the objective
# more than the provided budget.
#
# Closest known influences: Inspired by population-based black-box optimization
# heuristics (e.g., evolutionary strategies / CMA-like intuition) but implemented
# with a lightweight, evaluation-budget-safe mixture of local and global random
# search.
#
# Novelty or unusual aspects: Uses a very small population with a greedy
# replacement into the worst member, and adapts exploration probability based on
# improvement rate while keeping step-size schedule tied to evaluation progress.
#
# Failure modes: For extremely non-smooth or deceptive objectives, progress may
# stall and the algorithm may revert to more global sampling; if the optimum is
# very small in measure relative to the space, any stochastic method can still
# miss it within limited budgets.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        if dim <= 0:
            raise ValueError("dim must be positive")

        # Read bounds from func.lower/func.upper or func.bounds.lb/ub
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
            ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)
        else:
            raise AttributeError("Could not find bounds: expected func.lower/upper or func.bounds.lb/ub")

        if lb.size != dim or ub.size != dim:
            raise ValueError(f"Bounds size mismatch: expected dim={dim}, got lb={lb.size}, ub={ub.size}")

        # Ensure lb <= ub
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        lb = lb2
        ub = ub2

        # Precompute ranges; handle degenerate dimensions robustly
        span = ub - lb
        # If span is zero for some dims, sampling/perturbation should respect fixed coordinate
        fixed_mask = span <= 0.0
        free_span = span.copy()
        free_span[fixed_mask] = 1.0  # avoid divide-by-zero for scaling

        # Clip helper
        def clip_to_bounds(x):
            # For numerical stability, clip only with finite bounds; assume bounds are finite
            return np.minimum(np.maximum(x, lb), ub)

        # Budget-safe evaluation wrapper
        evals = 0
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed: return a deterministic point in bounds
            x0 = lb.copy()
            return x0, float("inf")

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                # Should never happen; guard for safety.
                return float("inf")
            y = func(np.asarray(x, dtype=float))
            evals += 1
            return float(y)

        # Initialize population size: keep small and robust for any dim/budget
        # Ensure at least 1 point evaluated (unless budget=0 handled above).
        pop_size = int(np.clip(4 + dim // 4, 4, 24))
        pop_size = min(pop_size, budget)

        # Start with a mix of uniform samples and a corner-ish point near the lower bound
        # to provide a deterministic anchor in addition to randomness.
        population = []
        # Anchor point: lower bound
        anchor = lb.copy()
        population.append(anchor)

        # Remaining points: uniform random in bounds
        while len(population) < pop_size:
            r = np.random.random(dim)
            x = lb + r * span
            # If some dims fixed, ensure exact fixed coordinates
            if np.any(fixed_mask):
                x[fixed_mask] = lb[fixed_mask]
            population.append(x)

        X = np.vstack(population)
        Y = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            Y[i] = evaluate(X[i])

        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])

        # Step size schedule: start from a fraction of the span; shrink with progress.
        # Use a scale factor that depends on the typical scale of the search space.
        # We use the mean span magnitude as a proxy.
        mean_span = float(np.mean(np.abs(span)))
        if mean_span <= 0.0:
            mean_span = 1.0

        # Initial local step: moderate fraction of search space
        step0 = 0.35 * mean_span
        step_min = 1e-12 * mean_span

        # Exploration probability adapts with progress
        # Start with some exploration, then reduce.
        p_global0 = 0.35
        p_global_min = 0.05

        # Track improvements to detect stagnation
        no_improve_iters = 0
        last_best_y = best_y

        # Main loop: generate/evaluate candidates until budget exhausted.
        # Use iteration counter independent of eval count.
        # Each loop typically evaluates multiple candidates via batches, but never
        # exceeds budget.
        # To keep it simple and robust, evaluate one candidate at a time with occasional
        # small batch when budget allows.
        while evals < budget:
            progress = evals / budget
            # Shrinking local scale: linearly or slightly nonlinearly
            shrink = (1.0 - progress) ** 1.2
            step = max(step_min, step0 * shrink)

            # Adapt exploration probability based on stagnation
            # If no improvement recently, increase exploration.
            if no_improve_iters >= 6:
                p_global = min(0.8, p_global0 + 0.2)
            else:
                p_global = max(p_global_min, p_global0 * (1.0 - 0.85 * progress))

            # Decide candidate type: global uniform or local random around best.
            if np.random.random() < p_global:
                # Global diversification: sample uniformly
                r = np.random.random(dim)
                x_new = lb + r * span
                if np.any(fixed_mask):
                    x_new[fixed_mask] = lb[fixed_mask]
            else:
                # Local exploitation around best_x
                # Combine a couple of random directions:
                # - random Gaussian component scaled by per-dim span
                # - occasional "directional" move using normalized random vector
                # - occasional use of a second-best to diversify within exploitation
                use_second = (np.random.random() < 0.3) and (pop_size >= 2)
                if use_second:
                    # Choose a second-best among population excluding current best index
                    order = np.argsort(Y)
                    # second candidate index as top-k within a window
                    j = int(order[min(1, pop_size - 1)])
                    base = X[j]
                else:
                    base = best_x

                # Per-dimension scaling by span to be scale-aware
                # (fixed dims won't move since span=0 and clipping will fix them)
                noise = np.random.normal(size=dim)
                x_new = base + (step * noise) * (span / (mean_span + 1e-30))

                # Add a directional component sometimes to improve exploration geometry
                if np.random.random() < 0.5 and not np.all(fixed_mask):
                    v = np.random.normal(size=dim)
                    norm = float(np.linalg.norm(v))
                    if norm > 0:
                        v = v / norm
                        # Directional step scaled similarly by span
                        dir_step = step * (0.25 + 0.75 * np.random.random())
                        x_new = x_new + dir_step * v * (span / (mean_span + 1e-30))

                # Occasional coordinate-like move for robustness in high dim
                if dim >= 3 and np.random.random() < 0.25:
                    k = int(np.random.randint(0, dim))
                    if not fixed_mask[k]:
                        coord_noise = np.random.normal()
                        x_new[k] = base[k] + coord_noise * step * (span[k] / (mean_span + 1e-30))

                # Clip to bounds
                x_new = clip_to_bounds(x_new)
                if np.any(fixed_mask):
                    x_new[fixed_mask] = lb[fixed_mask]

            # Evaluate new candidate
            y_new = evaluate(x_new)

            # If we somehow evaluated inf due to guard, stop early.
            if not np.isfinite(y_new):
                break

            # Greedy replacement: replace worst if better
            worst_idx = int(np.argmax(Y))
            if y_new < Y[worst_idx]:
                X[worst_idx] = x_new
                Y[worst_idx] = y_new

            # Update global best
            if y_new < best_y:
                best_y = y_new
                best_x = np.asarray(x_new, dtype=float).copy()
                if best_y < last_best_y - 1e-15:
                    no_improve_iters = 0
                    last_best_y = best_y
                else:
                    no_improve_iters = 0
            else:
                no_improve_iters += 1

            # Optional: small additional evaluations in the same loop when budget is plenty
            # to use remaining budget efficiently.
            # This avoids too many loop iterations while staying budget-safe.
            # Keep it minimal for robustness.
            if evals < budget:
                remaining = budget - evals
                if remaining >= 3 and np.random.random() < 0.25:
                    # Evaluate a tiny batch of extra local refinements
                    # (each candidate is evaluated individually; still budget-safe).
                    batch = min(2, remaining)
                    for _ in range(batch):
                        if evals >= budget:
                            break
                        progress = evals / budget
                        shrink = (1.0 - progress) ** 1.2
                        step = max(step_min, step0 * shrink)

                        # Local exploitation around current best
                        noise = np.random.normal(size=dim)
                        x_extra = best_x + (step * noise) * (span / (mean_span + 1e-30))
                        x_extra = clip_to_bounds(x_extra)
                        if np.any(fixed_mask):
                            x_extra[fixed_mask] = lb[fixed_mask]

                        y_extra = evaluate(x_extra)
                        if not np.isfinite(y_extra):
                            break

                        worst_idx = int(np.argmax(Y))
                        if y_extra < Y[worst_idx]:
                            X[worst_idx] = x_extra
                            Y[worst_idx] = y_extra

                        if y_extra < best_y:
                            best_y = y_extra
                            best_x = np.asarray(x_extra, dtype=float).copy()
                            no_improve_iters = 0
                            last_best_y = best_y
                        else:
                            no_improve_iters += 1

        return best_x, best_y
