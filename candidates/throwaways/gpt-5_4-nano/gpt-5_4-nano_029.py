import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, robust black-box minimization algorithm inspired by
# evolution strategies with coordinate-wise adaptation. It maintains a population of
# candidate solutions, samples around the current best using anisotropic step sizes,
# and refines the search by selecting the best individuals and adapting step sizes.
# Search state: Keeps track of the current best solution (best_x, best_y), an
# evaluation counter, a per-dimension step size sigma, and a population of candidates.
# Candidate generation: At each iteration, it generates offspring by sampling
# best_x + sigma * N(0,1) for a small population. A mixture of random restarts around
# the best and occasional uniform exploration is used to reduce stagnation.
# Selection and replacement: Uses fitness ranking (minimization) to select the top
# individuals. The new best is the best among the evaluated points; the search center
# remains near the best. Step sizes are adapted from the success of offspring.
# Adaptation: Uses a simple 1/5-style success heuristic based on whether the best
# offspring improves upon best_y. Step sizes are increased or decreased accordingly.
# Exploration mechanisms: Adds occasional uniform samples within bounds and uses
# larger sigma when improvements are frequent or stagnation is detected.
# Exploitation mechanisms: Otherwise concentrates sampling around the current best with
# shrinking step sizes as improvements occur.
# Boundary handling: All candidates are clipped to the provided bounds. The algorithm
# also keeps sigma within reasonable limits relative to the bounds to avoid numerical
# issues.
# Budget strategy: Strictly enforces the provided evaluation budget by counting every
# objective call and stopping when the next batch would exceed the remaining evaluations.
# Closest known influences: Combines ideas from evolution strategies (population sampling
# and recombination-like update via best-centered sampling) with success-based step-size
# control and occasional random restarts.
# Novelty or unusual aspects: Uses per-dimension sigma scaling based on improvement and
# a deterministic (seeded externally) mixture strategy for exploration, while staying
# compact and fully budget-safe.
# Failure modes: Can stagnate on flat landscapes or highly constrained problems if sigma
# shrinks too quickly; uniform exploration mitigates but does not fully eliminate this.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._get_bounds(func)
        n = self.dim
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.shape[0] != n or ub.shape[0] != n:
            raise ValueError("Bounds must match the provided dimension.")

        # Handle degenerate bounds safely
        span = ub - lb
        span = np.where(span > 0, span, 1.0)

        evals = 0

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        def safe_eval(x):
            nonlocal evals
            # Enforce budget strictly
            if evals >= self.budget:
                return None
            y = float(func(x))
            evals += 1
            return y

        # Initialize sigma relative to bounds; use global scaling to be dimension-agnostic.
        sigma_min = 1e-12 * float(np.mean(span))
        sigma_max = float(0.5 * np.max(span)) if np.max(span) > 0 else 1.0
        sigma = 0.25 * np.maximum(sigma_min, span)
        sigma = np.minimum(sigma, sigma_max)

        # Initial candidates: evaluate a small set (center + corners-like randoms)
        center = 0.5 * (lb + ub)
        best_x = clip(center)
        best_y = safe_eval(best_x)
        if best_y is None:
            # Budget is zero: return any point
            return best_x, float("inf")

        remaining = self.budget - evals
        # Population size scales mildly with dimension but remains small/compact
        pop_size = int(np.clip(4 + 2 * (n // 5), 6, 18))
        pop_size = min(pop_size, max(1, remaining)) if remaining > 0 else 0

        # Helper to decide if we can evaluate a full batch
        def can_evaluate_batch(k):
            return evals + k <= self.budget

        # Main loop: each iteration evaluates up to pop_size offspring (budget-safe)
        # Use a success counter to modulate exploration.
        success = 0
        stagnation = 0

        # If remaining budget is very small, we just do a single batch and stop.
        max_iters = 10_000  # budget-based break should occur first
        it = 0
        while it < max_iters and evals < self.budget:
            it += 1
            remaining = self.budget - evals
            if remaining <= 0:
                break

            # Determine actual batch size for this iteration
            batch = min(pop_size, remaining)
            if batch <= 0:
                break

            # Exploration probability increases with stagnation
            # (lower stagnation => more exploitation around best)
            if stagnation >= 3:
                p_uniform = 0.25
            elif stagnation == 2:
                p_uniform = 0.15
            elif stagnation == 1:
                p_uniform = 0.08
            else:
                p_uniform = 0.05

            # Offspring generation: mostly around best_x with anisotropic step sigma
            # Occasionally uniform within bounds to escape stagnation.
            # Use vectorized candidate generation.
            Z = np.random.randn(batch, n)
            X = best_x[None, :] + Z * sigma[None, :]

            # Mix in uniform exploration for some individuals
            if p_uniform > 0:
                m = int(np.floor(p_uniform * batch))
                if m > 0:
                    idx = np.random.choice(batch, size=m, replace=False)
                    U = lb[None, :] + np.random.rand(m, n) * (ub - lb)[None, :]
                    X[idx, :] = U

            # Boundary handling: clip
            X = clip(X)

            # Evaluate and select best offspring
            best_off_y = None
            best_off_x = None

            # Budget-safe evaluation (batch already chosen to fit budget)
            for i in range(batch):
                y = safe_eval(X[i, :])
                if y is None:
                    break
                if best_off_y is None or y < best_off_y:
                    best_off_y = y
                    best_off_x = X[i, :]

            # If we somehow did not evaluate anything, stop
            if best_off_x is None:
                break

            improved = best_off_y < best_y
            if improved:
                best_x, best_y = best_off_x, best_off_y
                success += 1
                stagnation = 0

                # Exploitation: shrink sigma when we're improving
                # Slightly stronger shrink after more consecutive successes.
                shrink = 0.82 ** (1 + min(3, success // 2))
                sigma = np.maximum(sigma_min, sigma * shrink)
            else:
                success = 0
                stagnation += 1

                # Exploration: increase sigma slowly when no improvement
                # to escape local traps.
                grow_factor = 1.10 if stagnation == 1 else (1.18 if stagnation == 2 else 1.28)
                sigma = np.minimum(sigma_max, sigma * grow_factor)

            # If stagnation persists, do a lightweight "restart" centered near a random point
            # (still best-biased to keep behavior stable).
            if stagnation >= 4 and evals < self.budget:
                # One more candidate evaluation (fits within remaining because we check budget)
                remaining = self.budget - evals
                if remaining > 0:
                    x_rand = lb + np.random.rand(n) * (ub - lb)
                    # Blend towards best to preserve exploitation
                    blend = 0.5 + 0.2 * np.random.rand()
                    x_restart = clip(blend * best_x + (1 - blend) * x_rand)
                    y_restart = safe_eval(x_restart)
                    if y_restart is not None and y_restart < best_y:
                        best_x, best_y = x_restart, y_restart
                        sigma = np.maximum(sigma_min, sigma * 0.9)
                        stagnation = 0
                        success = 1
                    else:
                        # Reset stagnation but expand sigma a bit more
                        sigma = np.minimum(sigma_max, sigma * 1.15)
                        stagnation = min(stagnation, 3)

        return best_x, best_y

    @staticmethod
    def _get_bounds(func):
        # Priority: func.lower/func.upper then func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = func.bounds.lb
            ub = func.bounds.ub
        else:
            raise AttributeError("Objective function must provide bounds via lower/upper or bounds.lb/bounds.ub.")
        return lb, ub
