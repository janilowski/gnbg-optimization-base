# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimizer (derivative-free) based on
# an evolutionary strategy with CMA-like diagonal step adaptation plus local
# refinement. It maintains a small population of candidate solutions, evaluates
# them via the provided objective, then updates the search distribution.
# Search state: Maintains current mean (best-so-far-centered), a diagonal
# step size vector (sigma), and a small population buffer for candidate points.
# Candidate generation: Each iteration samples offspring from a diagonal
# Gaussian around the current mean: x = mean + sigma * N(0,1). A
# “best-guided” local perturbation is also used around the current best to
# improve exploitation.
# Selection and replacement: Offspring are sorted by objective value (lower is
# better). The new mean is moved toward a weighted combination of the best
# offspring; sigma is adapted based on success (improvement) using a robust
# 1/5-like heuristic.
# Adaptation: Uses multiplicative updates to sigma (increase on success,
# decrease otherwise) and also shrinks steps when progress stalls.
# Exploration mechanisms: Stochastic offspring sampling with randomized step sizes,
# occasional larger perturbations around the mean, and occasional “restarts”
# of sigma scale when stagnation is detected.
# Exploitation mechanisms: Weighted recombination toward the best offspring and a
# short local search around the current best using reflected boundary handling.
# Boundary handling: Samples are clipped to provided bounds; additionally local
# refinements use reflection to reduce repeated sticking at boundaries.
# Budget strategy: Never exceeds the evaluation budget; each iteration consumes
# a fixed number of evaluations, with a final partial iteration for any remaining
# budget.
# Closest known influences: Combines ideas from evolution strategies (ES/CMA-like
# diagonal adaptation), success-rate step control, and small local refinements.
# Novelty or unusual aspects: Ensures budget adherence with a careful remaining-
# evaluation loop; mixes global ES sampling with a best-centered local update that
# also uses boundary reflection for efficiency.
# Failure modes: In very noisy or highly non-smooth objectives, adaptation may
# oscillate; if the optimum lies on/near narrow feasible regions, clipping can
# reduce diversity and cause premature convergence. The algorithm attempts to
# mitigate with occasional step increases and restart-like sigma scaling.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        n_dim = self.dim
        if n_dim <= 0:
            raise ValueError("dim must be positive")

        # --- Bounds extraction ---
        lower = None
        upper = None
        bounds = getattr(func, "bounds", None)

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif bounds is not None and hasattr(bounds, "lb") and hasattr(bounds, "ub"):
            lower = np.asarray(bounds.lb, dtype=float)
            upper = np.asarray(bounds.ub, dtype=float)
        else:
            raise AttributeError("func must provide (lower, upper) or func.bounds.lb/ub")

        if lower.shape != (n_dim,) or upper.shape != (n_dim,):
            lower = np.reshape(lower, (n_dim,))
            upper = np.reshape(upper, (n_dim,))

        # Ensure valid bounds
        lower = np.minimum(lower, upper)
        upper = np.maximum(lower, upper)

        # Precompute range, avoid division by zero
        span = upper - lower
        span = np.where(span > 0, span, 1.0)  # for scaling only

        # --- Budgeted evaluation wrapper ---
        eval_budget = max(0, self.budget)
        eval_count = 0

        def clamp_reflect(x):
            # Clip is used for global sampling; reflection used in local refinement.
            return np.minimum(np.maximum(x, lower), upper)

        def reflect(x):
            # Reflect values that go out of bounds to reduce "sticking" on edges.
            # Works for per-dimension bounds.
            x = np.asarray(x, dtype=float).copy()
            for i in range(n_dim):
                lo, hi = lower[i], upper[i]
                if hi <= lo:
                    x[i] = lo
                    continue
                width = hi - lo
                # Map to [0, 2*width) then reflect
                t = (x[i] - lo) % (2.0 * width)
                if t > width:
                    t = 2.0 * width - t
                x[i] = lo + t
            return x

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= eval_budget:
                # Should never happen if budget accounting is correct.
                return None
            eval_count += 1
            y = func(np.asarray(x, dtype=float))
            return float(y)

        # --- Initialization ---
        # Start mean near the middle to be robust; if bounds are wide, use them.
        mean = lower + 0.5 * span

        # Diagonal step sizes: about 25% of the span, capped.
        sigma = 0.25 * span
        # For near-degenerate bounds, keep sigma small.
        sigma = np.where(span > 0, sigma, 1e-3)

        # Track best
        best_x = mean.copy()
        best_y = float("inf")

        # Determine population size based on budget and dimension (compact but adaptive)
        # Aim for ~5-10 iterations with enough offspring to select.
        if eval_budget <= 0:
            return best_x, best_y

        # If dim is large relative to budget, choose smaller population.
        # Always evaluate at least 1 point.
        pop = int(np.clip(4 + n_dim // 2, 4, 24))
        pop = min(pop, max(1, eval_budget))  # can't evaluate more than budget in one step

        # Weighted recombination: use top-k weights
        top_k = max(2, pop // 2)

        # Strategy parameters
        # Success threshold roughly corresponds to 1/5 rule
        success_improve_factor = 0.999  # strictness; lower means easier improvement
        alpha_success = 1.2  # sigma increase multiplier
        alpha_fail = 0.82    # sigma decrease multiplier
        min_sigma = 1e-12
        max_sigma = 0.75 * span + 1e-12

        stagnation_counter = 0
        stagnation_limit = max(10, 2 * n_dim)

        # Initial evaluation of mean (counts towards budget)
        y0 = evaluate(mean)
        if y0 is not None:
            best_y = y0
            best_x = mean.copy()

        # Number of remaining evaluations
        remaining = eval_budget - eval_count

        # --- Main loop ---
        while remaining > 0:
            # Ensure we don't exceed budget: choose m offspring evaluations.
            m = min(pop, remaining)
            # Sample offspring with diagonal Gaussian
            # x = mean + sigma * N(0,1)
            z = np.random.randn(m, n_dim)
            candidates = mean + z * sigma

            # Global boundary handling: clip
            candidates = np.minimum(np.maximum(candidates, lower), upper)

            # Evaluate candidates
            ys = np.empty(m, dtype=float)
            for i in range(m):
                y = evaluate(candidates[i])
                if y is None:
                    # Budget exhausted mid-loop; keep partial results
                    ys = ys[:i]
                    candidates = candidates[:i]
                    m = i
                    break
                ys[i] = y

            # If nothing was evaluated, stop
            if m <= 0:
                break

            # Update best
            idx_best = int(np.argmin(ys))
            if ys[idx_best] < best_y:
                best_y = float(ys[idx_best])
                best_x = candidates[idx_best].copy()

            # Selection: keep top_k among current offspring (and best_x if better already)
            order = np.argsort(ys)
            k = min(top_k, m)

            # Weighted recombination toward the selected points
            # Use decreasing weights from best to kth-best
            # weights sum to 1
            ranks = np.arange(k, dtype=float)
            weights = np.log(k + 0.5) - np.log(ranks + 1.0)
            weights /= np.sum(weights)

            selected = candidates[order[:k]]
            new_mean = np.sum(selected * weights[:, None], axis=0)

            # Step adaptation via success rate:
            # Consider improvement if any selected beats current mean's fitness estimate.
            # Since we only have best_y, use best_y proxy.
            # Success if at least one of selected improves beyond a tiny factor.
            improved = np.min(ys[order[:k]]) < best_y * success_improve_factor

            # Update sigma: multiplicative rule + mild shrinkage based on rank spread
            # Rank spread helps adapt in flatter regions.
            spread = float(np.std(ys[order[:k]]) + 1e-12)
            relative_spread = spread / (abs(best_y) + 1.0)

            if improved:
                sigma = np.minimum(max_sigma, sigma * alpha_success)
                stagnation_counter = max(0, stagnation_counter - 1)
            else:
                sigma = np.maximum(min_sigma, sigma * alpha_fail)
                stagnation_counter += 1

            # Mild additional shrink when relative spread is tiny (near-flat)
            if relative_spread < 0.05:
                sigma = np.maximum(min_sigma, sigma * 0.98)

            # Local refinement around best_x for exploitation, but budget-aware.
            # Do a very small number of evaluations if budget allows.
            remaining = eval_budget - eval_count
            local_evals = 0
            # One local evaluation when we have at least 1 remaining
            if remaining > 0:
                local_evals = 1
                # Generate 1 reflected perturbation biased by the current best.
                # Use smaller steps to focus.
                step_local = 0.25 * sigma
                direction = np.random.randn(n_dim)
                direction /= (np.linalg.norm(direction) + 1e-12)
                candidate_local = best_x + (np.random.randn(n_dim) * step_local) + 0.1 * direction * step_local
                candidate_local = reflect(candidate_local)
                y_local = evaluate(candidate_local)
                if y_local is not None and y_local < best_y:
                    best_y = float(y_local)
                    best_x = np.asarray(candidate_local, dtype=float).copy()
                    # Pull mean toward new best to accelerate convergence
                    new_mean = 0.7 * new_mean + 0.3 * best_x
                    # Slightly reduce sigma after success to exploit
                    sigma = np.maximum(min_sigma, sigma * 0.92)

            # Adaptation of mean
            mean = new_mean

            # Stagnation handling: expand sigma slightly and re-center around best_x
            remaining = eval_budget - eval_count
            if stagnation_counter >= stagnation_limit and remaining > 0:
                mean = 0.5 * mean + 0.5 * best_x
                sigma = np.minimum(max_sigma, sigma * (1.0 / alpha_fail) ** 0.5)
                stagnation_counter = 0

            # Update remaining loop condition
            remaining = eval_budget - eval_count
            if remaining <= 0:
                break

        return best_x, best_y
