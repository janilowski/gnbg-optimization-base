import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm using a
# simple evolutionary strategy with dynamic step-size and optional
# occasional "restart" behavior when progress stalls. Works for any
# dimension given bounds from the objective.
# Search state: Maintains current best solution x_best, current step size
# sigma, and a candidate population each iteration.
# Candidate generation: Samples offspring around the current best using
# Gaussian perturbations scaled by sigma (isotropic). Uses a small fraction
# of additional candidates sampled from the broader search region early
# to encourage exploration.
# Selection and replacement: Evaluates all candidates created in an iteration
# and keeps the best one as the new x_best. If the best improves, sigma
# shrinks; otherwise sigma expands slightly to escape local minima.
# Adaptation: Uses a success-based adaptation rule on sigma (log-scale).
# Exploration mechanisms: Early broad sampling and sigma expansion on
# stagnation. Additionally performs a light restart when there is repeated
# stagnation.
# Exploitation mechanisms: Majority of candidates are drawn around the best
# point found so far, with step size controlled by sigma.
# Boundary handling: Clips candidate points to the provided bounds after
# sampling, ensuring feasible evaluations.
# Budget strategy: Strictly caps the number of objective evaluations to the
# provided budget. Uses iteration blocks whose total candidate evaluations
# never exceed remaining budget.
# Closest known influences: Inspired by classical (1+λ)-ES / (μ+λ)-ES and
# the success-based 1/5-like step-size adaptation, implemented in a
# straightforward, robust way.
# Novelty or unusual aspects: Combines success-based sigma adaptation with a
# minimal restart heuristic triggered by stagnation, while staying fully
# budget-aware.
# Failure modes: If bounds are extremely tight or the budget is very small,
# progress may be limited. If the objective has extreme ill-conditioning,
# isotropic steps may be inefficient; however adaptation mitigates this.
# ALGORITHM_ANALYSIS_NOTE_END


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
        elif hasattr(func, "bounds") and func.bounds is not None:
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        if lb is None or ub is None:
            raise AttributeError(
                "Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )

        # Normalize shapes to (dim,)
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != self.dim or ub.size != self.dim:
            # Try a best-effort broadcast-like behavior: if bounds are scalars
            if lb.size == 1 and ub.size == 1:
                lb = np.full(self.dim, float(lb), dtype=float)
                ub = np.full(self.dim, float(ub), dtype=float)
            else:
                raise ValueError(f"Bounds do not match dim={self.dim}.")

        if not np.all(ub >= lb):
            raise ValueError("Upper bounds must be >= lower bounds component-wise.")

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        # --- Budget-aware evaluation counter ---
        max_evals = self.budget
        evals = 0

        def eval_once(x):
            nonlocal evals
            if evals >= max_evals:
                # Should never happen if we manage budget properly.
                return float("inf")
            y = func(x)
            evals += 1
            return float(y)

        # --- Initialize: random point in bounds ---
        # Use harness seed: randomness should be via numpy.
        x_best = lb + np.random.rand(self.dim) * (ub - lb)
        x_best = clip(x_best)
        y_best = eval_once(x_best)

        # Initial sigma from bound scale (avoid zero)
        span = ub - lb
        # If span is all zeros, the optimum is at the single feasible point.
        if np.all(span == 0):
            return x_best, y_best

        # A robust initial scale: fraction of median nonzero span
        nonzero = span[span > 0]
        base = np.median(nonzero) if nonzero.size else float(np.max(span))
        sigma = 0.3 * base / np.sqrt(self.dim)
        sigma = float(max(sigma, 1e-12))

        # Step-size adaptation parameters
        success_shrink = 0.82
        failure_grow = 1.18

        # Candidate population sizing: proportional to dimension but budget-aware
        # Typical ES: lambda ~ 4..10 * dim; keep modest to respect budgets.
        lam = int(max(4, min(12 * self.dim, 50 * self.dim)))
        lam = max(4, lam)

        # Exploration quota: early stage uses extra wide candidates
        explore_frac = 0.25

        # Stagnation / restart
        best_improv = 0.0
        stall_iters = 0
        stall_limit = 8

        # --- Main loop: each iteration evaluates a block of candidates ---
        # We stop when we cannot evaluate at least one more candidate.
        while evals < max_evals:
            remaining = max_evals - evals
            # Determine how many candidates we can afford this iteration.
            cur_lam = min(lam, remaining)

            # Decide how many are exploratory (wide) vs exploitative (around best).
            n_explore = int(cur_lam * explore_frac)
            n_explore = min(n_explore, cur_lam)
            n_exploit = cur_lam - n_explore

            # Offspring matrix
            # Exploit: x_best + sigma * N(0,1)
            # Explore: sample uniformly from bounds or from a wider Gaussian around best
            # (we use uniform for robustness).
            X = np.empty((cur_lam, self.dim), dtype=float)

            # Exploit block
            if n_exploit > 0:
                Z = np.random.randn(n_exploit, self.dim)
                X[:n_exploit] = x_best[None, :] + sigma * Z

            # Explore block
            if n_explore > 0:
                # Uniform in bounds (broad exploration)
                U = np.random.rand(n_explore, self.dim)
                X[n_exploit:] = lb[None, :] + U * (ub - lb)[None, :]

            # Boundary handling
            X = clip(X)

            # Evaluate and select best
            y_iter_best = y_best
            x_iter_best = x_best
            for i in range(cur_lam):
                xi = X[i]
                yi = eval_once(xi)
                if yi < y_iter_best:
                    y_iter_best = yi
                    x_iter_best = xi

            # Adaptation and replacement
            if y_iter_best < y_best:
                # Improvement: accept best
                prev_best = y_best
                x_best = x_iter_best
                y_best = y_iter_best

                # Success-based sigma shrink (stronger if relative improvement is large)
                rel = abs(prev_best - y_iter_best) / (abs(prev_best) + 1e-12)
                shrink = success_shrink ** (1.0 + 2.0 * min(1.0, rel * 10.0))
                sigma *= shrink
                stall_iters = 0
            else:
                # No improvement: grow sigma a bit and count stall
                sigma *= failure_grow
                stall_iters += 1

            sigma = float(min(max(sigma, 1e-12), 2.0 * base))  # keep bounded

            # Restart heuristic on repeated stall: re-center near best but with larger sigma,
            # plus occasional uniform injection already present via explore sampling.
            if stall_iters >= stall_limit and evals < max_evals:
                stall_iters = 0
                # Recenter: sample a new point near the best with a large step
                # (still within bounds via clipping).
                big_sigma = min(3.0 * base, max(2.0 * sigma, 0.8 * base / np.sqrt(self.dim)))
                Z = np.random.randn(self.dim)
                x_restart = clip(x_best + big_sigma * Z)
                y_restart = eval_once(x_restart)
                if y_restart < y_best:
                    x_best = x_restart
                    y_best = y_restart
                    sigma = max(0.5 * sigma, 1e-12)
                else:
                    sigma = max(0.9 * sigma, 1e-12)

        return x_best, y_best
