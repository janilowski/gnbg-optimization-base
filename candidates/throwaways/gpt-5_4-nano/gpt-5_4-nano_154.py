import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer inspired by
# a repeated "restarts with shrinking Gaussian steps" strategy. It maintains
# a population of candidate points, iteratively samples around the best-so-far,
# and occasionally performs exploratory restarts when progress stalls.
# Search state: Keeps a current evaluation budget, tracks the best point/value
# found so far, and maintains step-size (sigma) controlling how far new samples
# are perturbed. Uses a small population to reduce variance per iteration.
# Candidate generation: Generates new points by adding Gaussian noise to the
# current best (and, in early phases, also samples around the current population
# best). Samples are clipped to bounds.
# Selection and replacement: Evaluates candidates and keeps the best observed point.
# Between iterations it replaces the population with newly sampled points and
# updates sigma based on improvement.
# Adaptation: Sigma shrinks when improvements are observed and slowly grows
# (bounded) when progress stalls, enabling coarse-to-fine search.
# Exploration mechanisms: Periodic "restart" (re-sampling around a random
# point within bounds) when no improvement occurs for several iterations.
# Exploitation mechanisms: Most sampling focuses on perturbations of the current
# best solution with progressively reduced sigma.
# Boundary handling: Uses reflection-like clipping by clamping to [lb, ub] after
# perturbation to ensure validity.
# Budget strategy: Uses strict remaining-evaluation accounting; stops immediately
# once budget is exhausted and returns the best point/value found so far.
# Closest known influences: A simplified combination of (1) CMA-ES-like step-size
# adaptation feel, (2) (μ,λ)-style selection by best, and (3) restart-based
# global diversification, but implemented compactly without covariance estimation.
# Novelty or unusual aspects: Uses a best-centered Gaussian sampler with adaptive
# sigma and a stall-based restart trigger, tuned to work for both small and large
# dimensions with minimal hyperparameters.
# Failure modes: If the objective is extremely noisy or discontinuous, sigma
# adaptation and selection may mislead; clipping can also reduce effective
# search near boundaries; very tight budgets may yield limited exploration.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget_total = self.budget
        if budget_total <= 0 or dim <= 0:
            # No evaluations possible; return a feasible point if possible.
            bounds = self._get_bounds(func, dim)
            lb, ub = bounds
            x0 = (lb + ub) * 0.5
            return x0, float("inf") if hasattr(func, "__call__") else x0

        lb, ub = self._get_bounds(func, dim)
        rng = np.random

        # Track evaluations.
        evals = 0
        best_x = None
        best_y = float("inf")

        def clamp(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_one(x):
            nonlocal evals, best_x, best_y
            y = float(func(np.asarray(x, dtype=float)))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.array(x, dtype=float, copy=True)
            return y

        # Helper to sample a random point within bounds.
        def sample_uniform():
            return lb + (ub - lb) * rng.random(dim)

        # Helper to sample around a center with Gaussian noise, then clamp.
        def sample_around(center, sigma):
            # Use per-dimension isotropic noise scaled by (ub-lb) to be robust.
            span = np.maximum(ub - lb, 1e-12)
            x = center + (sigma * span) * rng.standard_normal(dim)
            return clamp(x)

        # If bounds are degenerate (lb==ub), optimum is that point.
        if np.allclose(lb, ub, rtol=0, atol=1e-15):
            x = np.array(lb, dtype=float)
            if evals < budget_total:
                eval_one(x)
            return best_x, best_y

        # Initialize with a random point and evaluate.
        center = sample_uniform()
        eval_one(center)

        # Population size per iteration (small, robust across dim).
        # Keep it at least 2 to allow selection pressure but not too big for budget.
        # Maximize λ slightly when budget allows.
        lam_max = 12
        lam = int(max(2, min(lam_max, max(2, (budget_total - 1) // 10))))
        lam = min(lam, max(2, budget_total - 1)) if budget_total > 1 else 1

        # Sigma adaptation bounds.
        span = np.maximum(ub - lb, 1e-12)
        # Start with fairly global moves, shrink down later.
        sigma = 0.5
        sigma_min = 1e-6
        sigma_max = 2.0

        # Stall-based restart.
        stall_iters = max(3, int(np.ceil(0.05 * budget_total / lam))) if budget_total > 10 else 3
        since_improve = 0
        it = 0

        # Evaluate strategy: repeated batches around best_x.
        while evals < budget_total:
            it += 1
            improved_in_batch = False

            # Decide exploration vs exploitation based on progress and sigma.
            # When stalled, increase exploration (larger noise and random centers).
            explore = since_improve >= stall_iters

            if explore:
                # Restart: pick a new random center within bounds.
                center = sample_uniform()
                # Evaluate center to seed exploitation around it (cost 1 eval).
                if evals < budget_total:
                    eval_one(center)
                since_improve = 0

            # Generate a small batch of candidates.
            # Always include the current best to allow "keeping" when noise is too small.
            candidates = []
            candidates.append(np.array(best_x, copy=False))
            # Remaining λ-1 points are sampled around best or current center.
            # During exploration, sample around random center; otherwise around best.
            base = center if explore else best_x

            remaining = budget_total - evals
            # In each batch, we can evaluate up to lam candidates or whatever remains.
            batch_n = min(lam, remaining)
            # If budget is very low, just evaluate as many as possible.
            # Ensure at least one candidate besides best if possible.
            while len(candidates) < batch_n:
                # Mix exploitation and slight diversification within each batch.
                # This reduces risk of local traps.
                if rng.random() < 0.75:
                    s = sigma
                else:
                    # A slightly larger step sometimes helps escape tiny basins.
                    s = min(sigma_max, sigma * 1.8)
                candidates.append(sample_around(base, s))

            # Evaluate candidates and select improvements.
            pre_best = best_y
            for x in candidates:
                if evals >= budget_total:
                    break
                eval_one(x)

            if best_y < pre_best - 1e-15:
                improved_in_batch = True
                since_improve = 0
                # Shrink sigma to exploit the basin.
                sigma = max(sigma_min, sigma * 0.85)
            else:
                since_improve += 1
                # Slowly grow sigma when not improving.
                sigma = min(sigma_max, sigma * (1.03 + 0.02 * rng.random()))

            # Optional: if sigma is extremely small and we're not improving, trigger restart.
            if sigma <= sigma_min * 1.5 and since_improve >= max(2, stall_iters // 2):
                since_improve = stall_iters  # force explore next loop

        # Ensure we have a best_x (should be set by at least one evaluation).
        if best_x is None:
            best_x = sample_uniform()
            best_y = float("inf")
        return best_x, best_y

    def _get_bounds(self, func, dim):
        # Support:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
            ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)
        else:
            # If bounds are not provided, default to [-5, 5]^dim.
            lb = -5.0 * np.ones(dim, dtype=float)
            ub = 5.0 * np.ones(dim, dtype=float)

        if lb.size != dim or ub.size != dim:
            # Try broadcasting scalars.
            if lb.size == 1:
                lb = np.full(dim, float(lb.item()), dtype=float)
            if ub.size == 1:
                ub = np.full(dim, float(ub.item()), dtype=float)
        if lb.size != dim or ub.size != dim:
            # Fallback: pad/truncate deterministically.
            lb = (np.resize(lb, dim)).astype(float, copy=False)
            ub = (np.resize(ub, dim)).astype(float, copy=False)

        # Ensure lb <= ub
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        return lb2, ub2
