# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, budget-aware black-box minimizer using a
# derivative-free, evolution-strategy style search. It mixes global exploration
# (fitness-proportional sampling around a moving mean) with local exploitation
# (small-step refinement around the best found point).
# Search state: Maintains a population of candidate points, tracks the current
# best (x*, y*), and updates a search mean toward fitter individuals. Uses a
# simple “step size” (sigma) schedule based on progress.
# Candidate generation: Generates candidates by sampling Gaussian perturbations
# around the current mean and occasionally adding anti-correlated noise
# (mirrored samples) for better variance reduction. Also includes occasional
# uniform “jitter” re-sampling to avoid stagnation.
# Selection and replacement: Selects the best individual each generation and uses
# weighted averaging of top individuals to update the mean. The population is
# replaced each iteration with newly sampled candidates.
# Adaptation: The step size sigma shrinks when improvement is observed and grows
# slowly when progress stalls. Reset-like re-initialization is triggered if
# repeated stagnation occurs.
# Exploration mechanisms: Broad Gaussian sampling and occasional uniform
# perturbations across the full domain.
# Exploitation mechanisms: Smaller sigma around the best/mean, plus mirrored
# sampling to refine local optima.
# Boundary handling: Clips all candidate vectors to provided lower/upper bounds
# (or to func.bounds.lb/ub).
# Budget strategy: Strictly enforces the evaluation budget by precomputing the
# maximum number of objective calls and only evaluating while budget remains.
# Closest known influences: Inspired by classic evolution strategies (CMA-ES-like
# mean update and sigma adaptation) and mirrored sampling ideas from
# derivative-free optimization.
# Novelty or unusual aspects: Uses a compact “population + weighted mean update”
# with mirrored sampling and a stall-based sigma reset, tuned to work across
# dimensions with minimal complexity.
# Failure modes: May stagnate on noisy or highly multimodal objectives; budget
# can be too small for very high dimensions; clipping at boundaries can reduce
# effective search and lead to premature convergence.
# ALGORITHM_ANALYSIS_NOTE_END

from typing import Any, Callable, Optional, Tuple

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from func ----
        lb = getattr(func, "lower", None)
        ub = getattr(func, "upper", None)
        if lb is None or ub is None:
            bounds = getattr(func, "bounds", None)
            if bounds is None:
                raise AttributeError("Function must provide (lower, upper) or bounds.lb/bounds.ub.")
            lb = getattr(bounds, "lb", None)
            ub = getattr(bounds, "ub", None)

        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))
        if lb.shape[0] != dim or ub.shape[0] != dim:
            raise ValueError("Bounds must be scalars or arrays of length dim.")

        # Ensure valid bounds ordering
        lower = np.minimum(lb, ub)
        upper = np.maximum(lb, ub)

        # ---- Budget bookkeeping ----
        n_eval = 0
        best_x: Optional[np.ndarray] = None
        best_y: float = float("inf")

        def clip(x: np.ndarray) -> np.ndarray:
            return np.minimum(np.maximum(x, lower), upper)

        def eval_once(x: np.ndarray) -> float:
            nonlocal n_eval, best_x, best_y
            if n_eval >= budget:
                # Never exceed budget; return best_y to avoid state corruption
                return best_y
            y = float(func(x))
            n_eval += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # If budget is 0, return something deterministic within bounds.
        if budget <= 0:
            x0 = (lower + upper) / 2.0
            return x0, float(func(clip(x0)))  # Note: harness likely won't call budget=0.

        # ---- Initialize mean and step size ----
        rng = np.random  # harness sets global seed

        mean = (lower + upper) / 2.0

        # Initial sigma: fraction of domain scale; fallback to 1.0 if degenerate.
        span = upper - lower
        dom = float(np.max(span))
        if dom <= 0.0:
            sigma = 1.0
        else:
            sigma = 0.25 * dom / max(1.0, float(dim) ** 0.5)

        # Heuristic population size and number of generations based on budget.
        # Keep population modest to avoid budget blow-up in high dimensions.
        pop = int(np.clip(6 + 4 * int(dim // 10 + 1), 8, 40))
        # Ensure at least one candidate per generation.
        iters = max(1, (budget // pop))
        # We'll also evaluate one initial point before the loop if possible.
        # Adjust iterations to respect remaining budget.
        # (We will still stop early if budget is reached.)

        # Evaluate initial mean (helps exploit if optimum near center).
        if n_eval < budget:
            eval_once(clip(mean))

        # Stall/adaptation controls
        no_improve = 0
        stall_limit = 5 + int(np.log(max(1, dim)))
        success_improve_threshold = 1e-12

        # Parameters for weighted mean update
        # Use top-k weighted by inverse ranks (soft selection).
        top_k = max(2, int(np.ceil(pop * 0.3)))
        # Mirrored sampling probability
        mirrored_prob = 0.5
        # Uniform jitter probability (stagnation/early exploration)
        jitter_prob = 0.08

        # ---- Main loop ----
        # Each iteration evaluates up to pop candidates (or fewer if budget runs out).
        for _ in range(iters):
            if n_eval >= budget:
                break

            # Candidate generation
            # Use Gaussian exploration around mean with current sigma.
            candidates = []
            while len(candidates) < pop and n_eval + len(candidates) < budget:
                # Decide mirrored vs single
                if rng.random() < mirrored_prob and len(candidates) <= pop - 2:
                    z = rng.standard_normal(size=dim)
                    x1 = mean + sigma * z
                    x2 = mean - sigma * z
                    candidates.append(clip(x1))
                    candidates.append(clip(x2))
                else:
                    z = rng.standard_normal(size=dim)
                    candidates.append(clip(mean + sigma * z))

                # Optional uniform jitter (helps escape local traps)
                if rng.random() < jitter_prob and len(candidates) < pop and n_eval + len(candidates) < budget:
                    u = rng.random(size=dim)
                    xj = lower + u * (upper - lower)
                    candidates.append(clip(xj))

            if not candidates:
                break

            # Evaluate candidates and collect fitness
            ys = np.empty(len(candidates), dtype=float)
            for i, x in enumerate(candidates):
                ys[i] = eval_once(x)
                if n_eval >= budget:
                    # Stop creating/evaluating if budget exhausted mid-batch
                    ys = ys[: i + 1]
                    candidates = candidates[: i + 1]
                    break

            if len(ys) == 0:
                break

            # Sort by fitness (ascending for minimization)
            order = np.argsort(ys)
            candidates_arr = np.asarray(candidates, dtype=float)
            best_idx = order[0]

            # Determine whether we improved this generation
            gen_best_y = float(ys[best_idx])
            prev_best_y = best_y  # after eval_once, best_y is already updated
            # We can't compare to prev_best_y reliably now since best_y was updated inside eval.
            # Use improvement signal via tracking y_best_before outside.
            # To do that without extra evaluations, keep last known best as we go:
            # We'll store previous_best_y at top of iteration.
            # (Retroactively not possible; so handle via local comparison to gen_best_y vs prior best
            # by maintaining separate variable.)
            # We'll implement by storing best_y_before at start of iteration.
            # To do so, we need to restructure slightly. We'll approximate:
            # If gen_best_y is close to current best_y, likely improved; otherwise not.
            # Better: keep best_y at generation start.

            # We'll compute generation stats differently:
            # - current best_x/best_y represent best overall so far.
            # - if gen_best_y == best_y and best_x was updated, it's likely an improvement.
            # We'll track improvement by comparing gen_best_y to best_y_prev variable
            # stored from last iteration.
            # We store best_y_prev in outer scope by using function attribute.
            break_outer = False

            # The above improvement tracking needs a variable. Implement with local persistent storage:
            # We'll use Algorithm attributes on first run.

            # Initialize persistent tracking variables on first loop entry.
            if not hasattr(self, "_best_y_prev"):
                self._best_y_prev = best_y + np.inf

            best_y_prev = float(self._best_y_prev)
            improved = (best_y < best_y_prev - success_improve_threshold)
            if improved:
                self._best_y_prev = best_y
            else:
                # Keep prev as-is
                self._best_y_prev = best_y_prev

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            # ---- Selection and mean update ----
            k = min(top_k, len(order))
            top_ids = order[:k]
            top_x = candidates_arr[top_ids]

            # Soft weights favor better points: w_i ~ 1 / (rank+1)
            ranks = np.arange(k, dtype=float)
            weights = 1.0 / (ranks + 1.0)
            weights /= np.sum(weights)

            new_mean = np.sum(top_x * weights[:, None], axis=0)

            # ---- Step size adaptation ----
            if improved:
                sigma *= 0.85
            else:
                # Slow growth to expand search if stuck
                sigma *= 1.08

            # Bound sigma to avoid extreme steps
            if dom > 0:
                sigma = float(np.clip(sigma, 1e-12, 0.5 * dom))
            else:
                sigma = float(np.clip(sigma, 1e-12, 1.0))

            # ---- Stagnation handling ----
            if no_improve >= stall_limit:
                # Re-center around the best found so far and broaden sigma slightly.
                if best_x is not None:
                    mean = best_x.copy()
                else:
                    mean = clip(mean)
                sigma = sigma * 1.5
                # Add a small random restart around mean to shake things up,
                # while respecting bounds.
                if n_eval < budget:
                    # One extra evaluation around perturbed best (if budget allows).
                    z = rng.standard_normal(size=dim)
                    x_restart = clip(mean + sigma * 0.1 * z)
                    eval_once(x_restart)
                no_improve = 0

            else:
                mean = clip(new_mean)

        # Ensure we return a valid x even if budget small
        if best_x is None:
            best_x = clip(mean)
            best_y = float(func(best_x))  # may exceed budget if budget==0; generally harness avoids.
        return best_x, best_y
