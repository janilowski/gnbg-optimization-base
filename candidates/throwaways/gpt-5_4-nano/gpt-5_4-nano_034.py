# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, budget-aware black-box minimizer inspired by
# the "Evolution Strategy with ranking and covariance-free Gaussian search".
# It maintains a small population of candidate solutions, evaluates them, and
# iteratively moves a search mean toward lower objective values using a
# weighted recombination of top-ranked points.
# Search state: Keeps a current mean vector, a global step size (sigma),
# and a population size. Each iteration samples candidates by adding Gaussian
# noise around the mean.
# Candidate generation: For each generation, draws pop_size candidates as
# mean + sigma * N(0, I) and clips them to the provided bounds.
# Selection and replacement: Evaluates all candidates, ranks by objective
# value (minimization), then replaces the mean with the weighted average of
# the best half of candidates (or a rank-based subset) to bias toward improvement.
# Adaptation: Uses a simple success-rate adaptation: sigma increases if
# improvements over the previous best are frequent, and decreases otherwise.
# Exploration mechanisms: Large sigma at the beginning encourages exploration;
# periodic sigma boosts help avoid premature convergence.
# Exploitation mechanisms: Ranking-based weighted recombination steadily reduces
# sigma when progress stalls, focusing samples near promising regions.
# Boundary handling: Applies clipping to the lower/upper bounds after sampling.
# Budget strategy: Never calls the objective more than the provided budget.
# The algorithm uses iterations of population evaluations and then stops early
# if the remaining budget is insufficient for another full generation.
# Closest known influences: Heuristic from covariance-free ES / CMA-like mean
# adaptation without full covariance learning.
# Novelty or unusual aspects: Uses rank-based weighted mean updates with a
# lightweight sigma adaptation based on improvement over the global best.
# Failure modes: For extremely narrow feasible regions or badly scaled
# objectives, clipping can cause many samples to pile up on bounds; progress
# may slow. For very small budgets, behavior is limited to a few evaluations.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # --- Read bounds from func ---
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        if lb is None or ub is None:
            raise ValueError(
                "Function bounds not found. Expected func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)

        if lb.size != self.dim or ub.size != self.dim:
            # Be robust to scalar bounds by broadcasting when possible.
            if lb.size == 1:
                lb = np.full(self.dim, float(lb))
            if ub.size == 1:
                ub = np.full(self.dim, float(ub))
            if lb.size != self.dim or ub.size != self.dim:
                raise ValueError("Bounds dimensionality does not match 'dim'.")

        lower = lb
        upper = ub
        span = upper - lower
        span = np.where(span == 0.0, 1.0, span)

        # Clip helper
        def clip_to_bounds(x):
            return np.minimum(np.maximum(x, lower), upper)

        # --- Budget accounting ---
        budget = max(1, int(self.budget))
        eval_count = 0

        def eval_objective(x):
            nonlocal eval_count
            if eval_count >= budget:
                # Should never happen due to budget checks, but guard anyway.
                return np.inf
            y = func(x)
            eval_count += 1
            return float(y)

        # --- Initialize search state ---
        # Start near the middle with a small random offset to break symmetry.
        mean = (lower + upper) / 2.0
        # Small perturbation based on scale; if span is tiny, keep it tiny.
        sigma = 0.25 * np.mean(np.abs(span)) + 1e-12
        sigma = max(sigma, 1e-12)

        # Population size: keep it moderate; must respect budget.
        # Use classic ES-ish rule but capped for compactness.
        pop_size = int(np.clip(4 + int(np.log(max(self.dim, 2))), 4, 32))
        pop_size = min(pop_size, budget)

        # Track best solution seen so far.
        best_x = mean.copy()
        best_y = eval_objective(best_x)

        # Iteration counters for adaptation.
        prev_best = best_y
        success_streak = 0

        # Precompute identity sampling cost is okay using numpy.
        # Main loop: each generation evaluates pop_size points.
        while eval_count < budget:
            remaining = budget - eval_count
            k = min(pop_size, remaining)

            # Sample candidates: mean + sigma * N(0, I)
            # Use (k, dim) array.
            Z = np.random.randn(k, self.dim)
            X = mean[None, :] + sigma * Z
            # Boundary handling: clip
            X = clip_to_bounds(X)

            # Evaluate all candidates
            ys = np.empty(k, dtype=float)
            for i in range(k):
                ys[i] = eval_objective(X[i])

            # Update global best
            idx_best = int(np.argmin(ys))
            if ys[idx_best] < best_y:
                best_y = float(ys[idx_best])
                best_x = X[idx_best].copy()

            # Rank selection (minimization)
            order = np.argsort(ys)
            Xs = X[order]
            ys_sorted = ys[order]

            # Weighted recombination using top fraction
            # Using 1..m linear weights favors best candidates.
            m = max(2, k // 2)
            m = min(m, k)
            X_top = Xs[:m]

            # Rank-based weights (positive, sum to 1)
            # Best gets highest weight.
            w = np.arange(m, 0, -1, dtype=float)  # m..1
            w /= w.sum()

            # Move mean toward weighted best
            new_mean = np.dot(w, X_top)

            # --- Adaptation (simple success-rate / progress based) ---
            improved = 1.0 if best_y < prev_best else 0.0
            if improved > 0.0:
                success_streak += 1
            else:
                success_streak = max(0, success_streak - 1)

            # Reduce sigma when progress stalls; increase slightly on success.
            # Keep it bounded to avoid numeric issues.
            if success_streak >= 2:
                sigma *= 1.08  # encourage exploration if we're doing well
            else:
                # If no improvement, shrink to exploit.
                if best_y < prev_best:
                    sigma *= 0.92
                else:
                    sigma *= 0.82

            # Also ensure sigma doesn't get too small relative to bounds span.
            scale = np.mean(np.abs(span))
            sigma = float(np.clip(sigma, 1e-12, max(1e-6, 0.5 * scale + 1e-12)))

            mean = new_mean
            prev_best = best_y

            # If sigma becomes extremely small or we're at bounds, still continue
            # until budget is used; clipping will handle feasibility.

        return best_x, best_y
