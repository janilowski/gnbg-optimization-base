# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization strategy
# using a population-based evolution with a shrinking trust-region.
# It maintains a small set of candidate solutions, evaluates them under a hard
# evaluation budget, and iteratively improves them by mixing global exploration
# (random directions/steps) with local exploitation (crossover around the best).
# Search state: The algorithm keeps a population of points, their objective
# values, the current best solution, and a scalar "step size" (trust radius).
# Candidate generation: Each generation samples offspring by:
#   (1) gaussian perturbations around the current best,
#   (2) directional mutations based on population differences,
#   (3) occasional random restarts for diversity (early in the run).
# Selection and replacement: Offspring are evaluated and then replace the worst
# individuals in the population if they improve; the best of all evaluated
# points is tracked as output.
# Adaptation: The step size shrinks when no improvement is observed for a few
# generations and expands slightly when improvement is frequent.
# Exploration mechanisms: Early random exploration via restarts and larger
# perturbations around the best; directional mutations use scaled differences
# between population members to explore.
# Exploitation mechanisms: Later generations reduce step size, increase the
# probability of smaller gaussian steps around the best, and bias crossover
# toward the best solution.
# Boundary handling: Any candidate proposed outside bounds is clipped back to
# the feasible region. This keeps feasibility without discarding evaluations.
# Budget strategy: Uses the provided evaluation budget exactly; it computes the
# maximum number of objective calls allowed and stops immediately when reaching
# the limit.
# Closest known influences: Lightweight heuristic combining ideas from
# evolution strategies / CMA-like step adaptation, but implemented simply with
# a single global step size and difference-based mutation.
# Novelty or unusual aspects: Uses a dynamic "stagnation" counter that
# triggers step-size shrinkage and occasional diversity refreshes, while
# ensuring the budget is never exceeded.
# Failure modes: For very irregular objectives or extremely tight bounds,
# clipping may reduce effective search; if the budget is too small, behavior
# can be dominated by random sampling. The method is nevertheless robust and
# dimension-agnostic.
# ALGORITHM_ANALYSIS_NOTE_END

from typing import Callable, Tuple, Any
import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func: Any) -> Tuple[np.ndarray, float]:
        lb, ub = self._get_bounds(func)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        dim = self.dim

        # Safety for degenerate bounds
        if lb.shape[0] != dim or ub.shape[0] != dim:
            lb = np.resize(lb, (dim,))
            ub = np.resize(ub, (dim,))
        if np.any(ub < lb):
            # Swap if bounds appear reversed
            tmp = lb.copy()
            lb = np.minimum(lb, ub)
            ub = np.maximum(tmp, ub)

        def clip(x: np.ndarray) -> np.ndarray:
            return np.minimum(ub, np.maximum(lb, x))

        # Objective wrapper that enforces budget exactly.
        evals = 0
        max_evals = max(1, self.budget)

        def eval_obj(x: np.ndarray) -> float:
            nonlocal evals
            if evals >= max_evals:
                # Should never happen due to careful loops, but keep safe.
                return float("inf")
            y = func(x)
            evals += 1
            return float(y)

        # Initialize population size (compact but enough for dimension).
        # Ensure we never exceed budget: initial evals + one generation.
        # Use at least 2 individuals when possible.
        pop_size = int(np.clip(4 + dim // 5, 2, 12))
        # Adjust pop size if budget is too small.
        pop_size = min(pop_size, max_evals)
        if pop_size < 2:
            pop_size = 1

        rng = np.random

        # Step size based on bounds range.
        span = ub - lb
        span = np.where(span > 0, span, 1.0)
        sigma = 0.25 * np.max(span) / (1.0 + 0.1 * dim)
        sigma = max(sigma, 1e-12)

        # Create initial population: uniform random points within bounds.
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        vals = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            vals[i] = eval_obj(pop[i])

        best_idx = int(np.argmin(vals))
        best_x = pop[best_idx].copy()
        best_y = float(vals[best_idx])

        # Evolution control
        # Number of "generations" based on remaining budget.
        # Each generation evaluates offspring (k), then updates population.
        # Keep k small/variable to avoid overshooting budget.
        stagnation = 0
        no_improve_limit = max(3, min(20, dim // 2 + 3))

        # Helper to produce offspring vectors
        def propose_offspring(current_pop: np.ndarray, current_vals: np.ndarray, t_norm: float) -> np.ndarray:
            nonlocal sigma

            # Sort indices to identify best and diversity direction.
            idx_sorted = np.argsort(current_vals)
            best_local = current_pop[idx_sorted[0]]
            # For difference-based mutation, pick two distinct indices.
            if pop_size >= 2:
                a, b = idx_sorted[0], idx_sorted[rng.randint(1, pop_size)]
                diff = current_pop[a] - current_pop[b]
            else:
                diff = rng.normal(size=dim)

            # Mix exploration/exploitation probabilities over time:
            # early: more exploration
            # late: more exploitation around best
            p_exploit = 0.25 + 0.65 * t_norm  # increases with time
            exploit = rng.rand() < p_exploit

            # Directional step scaling:
            scale_diff = 0.8 * (1.0 - 0.7 * t_norm)
            scale_rand = 1.1 * (1.0 - 0.3 * t_norm)

            if exploit:
                # Small gaussian around the best, with slight directional bias.
                z = rng.normal(size=dim)
                step = sigma * z + (0.15 + 0.35 * (1.0 - t_norm)) * scale_diff * diff
                child = best_local + step
            else:
                # More exploratory step: larger perturbation and occasional long jump
                if rng.rand() < (0.08 + 0.12 * (1.0 - t_norm)):
                    # "Long jump" direction
                    dir_vec = rng.normal(size=dim)
                    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
                    step = (0.4 + 0.9 * (1.0 - t_norm)) * sigma * dir_vec * rng.normal()
                else:
                    step = (0.7 + 0.9 * (1.0 - t_norm)) * sigma * rng.normal(size=dim)
                    step += scale_rand * sigma * (rng.rand(dim) - 0.5) * diff / (np.linalg.norm(diff) + 1e-12)
                child = best_local + step

                # Occasional random restart for diversity
                if pop_size > 1 and rng.rand() < (0.10 * (1.0 - t_norm) + 0.02):
                    child = rng.uniform(lb, ub, size=dim)

            return clip(child)

        # Main loop until budget is exhausted
        # Offspring per generation; at least 1, at most pop_size.
        while evals < max_evals:
            remaining = max_evals - evals
            # Decide offspring count; prefer small batches for budget safety.
            k = int(np.clip(pop_size, 1, remaining))
            # If remaining is very small, k becomes small automatically.
            # Compute time-normalized progress:
            t_norm = evals / max_evals

            # If stagnating, shrink sigma; if too long, refresh diversity slightly.
            if stagnation >= no_improve_limit:
                sigma *= 0.7
                # Diversity refresh: replace worst individual(s) with random points.
                idx_sorted = np.argsort(vals)  # ascending
                # Replace up to 2 worst if possible
                num_replace = min(2, pop_size)
                for r in range(1, num_replace + 1):
                    worst_i = idx_sorted[-r]
                    if evals >= max_evals:
                        break
                    pop[worst_i] = rng.uniform(lb, ub, size=dim)
                    vals[worst_i] = eval_obj(pop[worst_i])
                best_idx = int(np.argmin(vals))
                cur_best_y = float(vals[best_idx])
                if cur_best_y < best_y:
                    best_y = cur_best_y
                    best_x = pop[best_idx].copy()
                    stagnation = 0
                else:
                    stagnation = 0  # reset to prevent repeated shrinking only
                continue

            # Possibly adapt sigma based on recent progress.
            # If improving, allow slight expansion; otherwise mild decay.
            # (We only know improvement from previous best update.)
            # We'll adapt after offspring evaluations.
            old_best_y = best_y

            # Generate and evaluate offspring
            idx_sorted = np.argsort(vals)
            for _ in range(k):
                if evals >= max_evals:
                    break
                child = propose_offspring(pop[idx_sorted], vals[idx_sorted], t_norm)
                y_child = eval_obj(child)

                # Replace worst if improved
                worst_idx = int(np.argmax(vals))
                if y_child < vals[worst_idx]:
                    pop[worst_idx] = child
                    vals[worst_idx] = y_child

                    # Update global best
                    if y_child < best_y:
                        best_y = y_child
                        best_x = child.copy()
                        stagnation = 0
                else:
                    # If not inserted, increase stagnation mildly.
                    stagnation += 1

            # Step size adaptation after each generation batch.
            if best_y < old_best_y - 1e-15:
                # Improvement occurred somewhere in the batch: slight increase to keep searching
                sigma *= (1.05 - 0.15 * t_norm)
                sigma = min(sigma, 0.5 * np.max(span) if np.max(span) > 0 else sigma)
            else:
                # No improvement: shrink
                sigma *= (0.90 - 0.10 * t_norm)
                sigma = max(sigma, 1e-12)

            # If budget is exhausted, exit
            if evals >= max_evals:
                break

        return np.asarray(best_x, dtype=float), float(best_y)

    def _get_bounds(self, func: Any):
        # Read bounds from func.lower/func.upper or func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return np.asarray(func.lower, dtype=float), np.asarray(func.upper, dtype=float)
        if hasattr(func, "bounds"):
            b = func.bounds
            # Expected attributes: lb and ub
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return np.asarray(b.lb, dtype=float), np.asarray(b.ub, dtype=float)
        raise AttributeError("Objective function must provide bounds via lower/upper or bounds.lb/bounds.ub.")
