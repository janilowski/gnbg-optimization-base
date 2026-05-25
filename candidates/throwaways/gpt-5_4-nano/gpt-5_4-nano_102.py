import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm (budgeted, derivative-free)
# using a simple evolutionary strategy (ES) with coordinate-wise mutation strengths and
# occasional local refinement around the best found point.
# Search state: Maintains a current mean (best_x), its function value (best_y), a per-dimension
# mutation scale (sigma), and the remaining evaluation budget. Uses an evolving population
# centered around the current mean each iteration.
# Candidate generation: Each iteration samples a small population from a multivariate normal
# centered at the current best point, using an adaptive sigma per coordinate. Includes
# occasional uniform random samples to escape stagnation.
# Selection and replacement: Evaluates all candidates (within budget), selects the best candidate,
# and replaces the mean/best if an improvement is found. Mutation scales adapt based on whether
# improvements occur.
# Adaptation: Uses 1/5-success style logic per iteration (based on whether any candidate improves),
# shrinking sigma on low success and expanding slightly when success is observed.
# Exploration mechanisms: Adds sporadic global exploration by sampling random points uniformly
# within bounds; also uses larger sigma when sigma shrinks too much.
# Exploitation mechanisms: Includes a lightweight local refinement step by generating additional
# candidates with reduced step size around the best point when progress is detected.
# Boundary handling: Candidates are clipped to the provided bounds after mutation.
# Budget strategy: Tracks evaluations explicitly and never calls the objective after the budget
# is exhausted. Iteration sizes are chosen to fit the remaining budget.
# Closest known influences: Inspired by simple evolutionary strategies / (μ+λ) selection patterns
# and 1/5-success rule adaptation, with added bounded random restarts.
# Novelty or unusual aspects: Uses per-coordinate sigma adaptation with a fallback when sigma becomes
# overly small; incorporates a deterministic evaluation ordering for robustness within budget.
# Failure modes: Can stagnate on flat/noisy objectives or if bounds are extremely tight; if the
# objective has misleading noise, success-based adaptation may overreact and reduce step sizes too late.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        lb, ub = self._read_bounds(func)

        # Ensure finite bounds for clipping; if not finite, fall back to a wide interval.
        # (Black-box benchmarks are usually bounded; this is a robustness safeguard.)
        if not np.all(np.isfinite(lb)):
            lb = np.where(np.isfinite(lb), lb, -1.0)
        if not np.all(np.isfinite(ub)):
            ub = np.where(np.isfinite(ub), ub, 1.0)

        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        # Total evaluation counter
        evals = 0

        def eval_obj(x):
            nonlocal evals
            if evals >= self.budget:
                # Safety: Should never happen if bookkeeping is correct.
                return np.inf
            y = func(x)
            evals += 1
            return float(y)

        # If budget is extremely small, just sample random points within bounds.
        if self.budget <= 0:
            x0 = lb.copy()
            return x0, float(eval_obj(x0))  # will be inf if budget=0, but shouldn't occur.

        # Initialization: pick a reasonable starting mean as the midpoint,
        # then perturb to encourage initial exploration.
        mean = 0.5 * (lb + ub)
        # Per-coordinate sigma based on bounds width
        width = np.maximum(ub - lb, 1e-12)
        sigma = 0.3 * width
        # If dim is large, reduce initial sigma slightly to avoid excessive clipping.
        if dim > 50:
            sigma *= 0.7

        # Evaluate initial mean (counted)
        best_y = eval_obj(mean.copy())
        best_x = mean.copy()

        # Evolution parameters: compact, budget-aware
        # Population size per iteration: small to fit within budgets tightly.
        base_pop = 8
        # Local refinement: extra candidates when improvements happen.
        refine_k = 3

        # Success/adaptation parameters
        shrink = 0.85
        expand = 1.12
        min_sigma = 1e-12 * width
        max_sigma = 0.5 * width

        # Stagnation handling
        no_improve_iters = 0
        stagnation_limit = 10  # iterations

        # Main loop
        # Each iteration evaluates a set of candidates centered around the best.
        # We keep iteration count implicit by budget.
        it = 0
        while evals < self.budget:
            it += 1

            remaining = self.budget - evals
            if remaining <= 0:
                break

            # Determine how many candidates we can evaluate this iteration.
            pop = min(base_pop, remaining)
            if pop <= 0:
                break

            # Candidate generation around current best_x
            # (Normal mutations, clipped to bounds)
            candidates = np.empty((pop, dim), dtype=float)
            # Use a mix of local exploration and global exploration.
            # Probability of random exploration increases when stuck.
            p_global = 0.15 + 0.25 * min(no_improve_iters / float(stagnation_limit), 1.0)

            for i in range(pop):
                if np.random.rand() < p_global:
                    # Uniform random exploration within bounds
                    candidates[i] = lb + np.random.rand(dim) * (ub - lb)
                else:
                    # Gaussian mutation with per-coordinate sigma
                    step = np.random.randn(dim) * sigma
                    x = best_x + step
                    # Boundary handling: clip to bounds
                    candidates[i] = np.minimum(np.maximum(x, lb), ub)

            # Evaluate candidates
            # Track best in this batch
            batch_best_y = best_y
            batch_best_x = best_x
            improved = False

            for i in range(pop):
                y = eval_obj(candidates[i])
                if y < batch_best_y:
                    batch_best_y = y
                    batch_best_x = candidates[i].copy()
                    improved = True

            # Selection and replacement
            if improved:
                best_x = batch_best_x
                best_y = batch_best_y
                no_improve_iters = 0
            else:
                no_improve_iters += 1

            # Adaptation of sigma:
            # If improved, slightly expand; if not, shrink.
            # This keeps exploration alive while focusing when progress happens.
            if improved:
                sigma = np.minimum(sigma * expand, max_sigma)
            else:
                sigma = np.maximum(sigma * shrink, min_sigma)

            # Optional local refinement when improvement occurs recently
            # (only if we have budget left).
            if improved and (evals < self.budget) and refine_k > 0:
                remaining = self.budget - evals
                k = min(refine_k, remaining)
                for _ in range(k):
                    # Smaller step sizes around the best
                    local_sigma = 0.35 * sigma
                    x = best_x + np.random.randn(dim) * local_sigma
                    x = np.minimum(np.maximum(x, lb), ub)
                    y = eval_obj(x)
                    if y < best_y:
                        best_y = y
                        best_x = x.copy()

            # If stuck for long, do a mild "restart" towards a new random mean,
            # but still centered around the best with increased sigma to escape.
            if no_improve_iters >= stagnation_limit and evals < self.budget:
                # Increase sigma to broaden search
                sigma = np.minimum(sigma * 1.5, max_sigma)
                # Move best_x towards a random point to shake the landscape.
                # Keep best_y/best_x assignment after evaluating the new mean once.
                x_rand = lb + np.random.rand(dim) * (ub - lb)
                # Blend towards random point to keep continuity
                mean = 0.7 * best_x + 0.3 * x_rand
                mean = np.minimum(np.maximum(mean, lb), ub)
                # Evaluate new mean if budget allows
                y_mean = eval_obj(mean)
                if y_mean < best_y:
                    best_y = y_mean
                    best_x = mean.copy()
                no_improve_iters = 0

        return best_x, best_y

    def _read_bounds(self, func):
        # Prefer func.lower/func.upper
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Robust fallback: use [-1, 1] bounds if not provided.
            lb = -np.ones(self.dim, dtype=float)
            ub = np.ones(self.dim, dtype=float)

        # Ensure correct shape
        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != self.dim or ub.size != self.dim:
            # Best-effort coercion to dim if bounds are scalars
            if lb.size == 1:
                lb = np.full(self.dim, float(lb[0]), dtype=float)
            if ub.size == 1:
                ub = np.full(self.dim, float(ub[0]), dtype=float)

        # Final check
        if lb.size != self.dim or ub.size != self.dim:
            # Fallback if still inconsistent
            lb = -np.ones(self.dim, dtype=float)
            ub = np.ones(self.dim, dtype=float)

        # Ensure lb <= ub
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        return lo, hi
