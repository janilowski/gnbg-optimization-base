import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# for continuous functions with unknown smoothness. It uses a simple
# evolution strategy with diagonal step-size adaptation (inspired by
# CMA-ES style ideas, but kept lightweight and robust).
# Search state: Maintains a current mean vector (center), a single
# global-ish step size (sigma), and a decreasing population of candidate
# solutions each generation. It tracks the best (lowest) objective value
# found so far.
# Candidate generation: Each iteration samples a small population of
# perturbations around the current mean using an isotropic Gaussian
# distribution scaled by sigma. It evaluates each candidate.
# Selection and replacement: It selects the best candidates (lowest function
# values) and updates the mean toward their average to shift the search
# towards promising regions.
# Adaptation: It adapts sigma based on the relative success rate
# (how many sampled points improved upon the current best). Higher success
# increases sigma cautiously; lower success decreases sigma.
# Exploration mechanisms: Gaussian sampling with a nonzero sigma provides
# global-ish exploration; occasional "diversification" resets/boosts sigma
# if progress stalls.
# Exploitation mechanisms: Re-centering on the best-performing candidates and
# shrinking sigma with low success gradually focuses search locally.
# Boundary handling: Candidates are clipped to the provided bounds to
# guarantee feasibility.
# Budget strategy: The algorithm computes the number of generations based
# on the evaluation budget and ensures it never exceeds the allowed number
# of objective evaluations (each candidate is evaluated exactly once).
# Closest known influences: A lightweight (μ,λ)-ES with 1/5 success-rule-like
# step-size adaptation and mean update from top performers.
# Novelty or unusual aspects: Uses a stall-based sigma boost to reduce the
# chance of premature convergence in noisy/flat landscapes, while remaining
# budget-safe.
# Failure modes: If the objective is extremely noisy or bounds are very
# tight, selection pressure may be weak or clipping may dominate, leading
# to slow progress. If sigma becomes too small early, the stall mechanism
# helps recover but may still be limited by the budget.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._read_bounds(func, self.dim)

        # Helper: safe evaluation with budget tracking.
        evals_used = 0
        best_x = None
        best_y = np.inf

        def eval_one(x):
            nonlocal evals_used, best_x, best_y
            y = float(func(x))
            evals_used += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # If budget is extremely small, still make progress safely.
        if self.budget <= 0:
            # No evaluations allowed; return a feasible point (center).
            x0 = self._initial_point(lb, ub)
            return x0, float("inf")

        # Choose an initial mean: center of the box.
        mu = self._initial_point(lb, ub)
        mu = self._clip(mu, lb, ub)

        # Evaluate initial mean (counts toward budget).
        eval_one(mu)

        # Population size: small, robust across dimensions, but budget-aware.
        # Ensure at least 2 candidates per generation if possible.
        # Rough heuristic: 4..10 scaled mildly with dim, clamped.
        base_lam = int(np.clip(4 + self.dim // 5, 4, 10))
        lam = min(base_lam, max(2, self.budget - 1))  # candidates per gen
        # Each generation evaluates lam points; total budget is evals_used + lam * gens <= budget.
        remaining = self.budget - evals_used
        if remaining <= 0:
            return best_x, best_y

        # Determine number of generations.
        gens = max(1, remaining // lam)
        # If budget allows partial generation at the end, we'll handle it in loop.

        # Initial sigma: based on box size.
        box = ub - lb
        # Avoid zero ranges; fallback to 1.0 scale.
        scale = float(np.max(box)) if np.max(box) > 0 else 1.0
        sigma = 0.3 * scale / (self.dim**0.5 if self.dim > 0 else 1.0)
        sigma = max(sigma, 1e-12)

        # Selection size: use top ~1/3 of candidates, at least 1.
        k = max(1, lam // 3)

        # Adaptation parameters.
        # Target success ratio around 1/5 (classic), adjusted slightly for minimization.
        target_success = 0.2
        # Learning rates
        c_sigma_up = 1.25
        c_sigma_down = 0.85
        # Mean update mixing
        c_mean = 0.7
        # Diversification/stall mechanism
        stall_patience = 5
        stall_counter = 0
        prev_best = best_y

        for _ in range(gens):
            # Stop if we've exhausted budget.
            if evals_used >= self.budget:
                break

            # Possibly shrink last generation if budget doesn't allow full lam.
            cur_lam = min(lam, self.budget - evals_used)
            if cur_lam <= 0:
                break

            # Sample population.
            # Use isotropic Gaussian perturbations.
            # Shape: (cur_lam, dim)
            noise = np.random.randn(cur_lam, self.dim)
            X = mu + sigma * noise
            X = self._clip(X, lb, ub)

            # Evaluate candidates.
            ys = np.empty(cur_lam, dtype=float)
            for i in range(cur_lam):
                ys[i] = eval_one(X[i])

            # Identify top candidates for selection.
            # Lower objective is better.
            order = np.argsort(ys)
            top_idx = order[:k]
            top_X = X[top_idx]

            # Update mean toward selected candidates (exploitation).
            new_mean = np.mean(top_X, axis=0)
            mu = (1.0 - c_mean) * mu + c_mean * new_mean
            mu = self._clip(mu, lb, ub)

            # Success-based sigma adaptation (exploration/exploitation balance).
            # Success: candidate improves current best_y.
            # Note: best_y has already been updated by eval_one calls.
            # So compare to the best found at start of generation for a stable signal.
            # We use prev_best to approximate that.
            improved = np.sum(ys < prev_best)
            success_ratio = improved / float(cur_lam)

            if success_ratio > target_success:
                sigma *= c_sigma_up
            else:
                sigma *= c_sigma_down
            sigma = float(np.clip(sigma, 1e-14, 10.0 * max(scale, 1.0)))

            # Stall detection: if best hasn't improved, occasionally boost sigma.
            if best_y < prev_best - 1e-12:
                prev_best = best_y
                stall_counter = 0
            else:
                stall_counter += 1
                if stall_counter >= stall_patience:
                    # Diversify by increasing sigma and re-centering on current best_x if available.
                    if best_x is not None:
                        mu = self._clip(best_x, lb, ub)
                    sigma *= 1.8
                    sigma = float(np.clip(sigma, 1e-14, 10.0 * max(scale, 1.0)))
                    stall_counter = 0

        # Ensure best_x exists (it will if budget>=1).
        if best_x is None:
            best_x = self._initial_point(lb, ub)
            best_x = self._clip(best_x, lb, ub)
            best_y = float("inf")
        return best_x, best_y

    @staticmethod
    def _read_bounds(func, dim):
        # Bounds may be provided as func.lower/func.upper (arrays)
        # or func.bounds.lb/func.bounds.ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
            ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)
        else:
            raise AttributeError("Objective function must provide bounds via lower/upper or bounds.lb/bounds.ub.")

        if lb.size != dim or ub.size != dim:
            # Try to broadcast if possible (robust fallback for scalar bounds).
            if lb.size == 1:
                lb = np.full(dim, float(lb[0]))
            if ub.size == 1:
                ub = np.full(dim, float(ub[0]))

        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        if lb.shape[0] != dim or ub.shape[0] != dim:
            raise ValueError(f"Bounds size mismatch: expected dim={dim}, got lb={lb.shape}, ub={ub.shape}.")

        # Ensure valid ordering.
        # If bounds are accidentally swapped, fix them.
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        return lo, hi

    @staticmethod
    def _initial_point(lb, ub):
        return 0.5 * (lb + ub)

    @staticmethod
    def _clip(x, lb, ub):
        # Works for both 1D and 2D arrays.
        return np.minimum(np.maximum(x, lb), ub)
