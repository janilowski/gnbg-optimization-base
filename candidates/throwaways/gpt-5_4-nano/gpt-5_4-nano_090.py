import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm for a box-bounded continuous search space. It maintains a small population of candidate solutions and iteratively improves them using a mix of exploitation (local sampling around the best point) and exploration (global moves via differential-style direction sampling). It is robust across dimensions by using dimension-agnostic random sampling and clipping to bounds.
# Search state: The algorithm tracks a population X of size pop_size, their objective values y, the best-so-far point best_x and best_y, and a remaining evaluation counter to strictly respect the provided budget.
# Candidate generation: Each iteration creates trial points by combining (1) the current best and a random population member with a shrinking Gaussian step (local search), and (2) a differential-style direction formed from two random population members (global-ish exploration). Additionally, a small probability performs a purely random re-sampling within bounds to avoid getting stuck.
# Selection and replacement: For each trial point, the objective is evaluated once (if budget allows). If the trial improves (lower objective), it replaces the corresponding parent in the population. The best-so-far is updated after each evaluation.
# Adaptation: The step scale shrinks over time based on the fraction of budget already used. The algorithm also increases exploration when improvements stall, by temporarily mixing more differential-style moves.
# Exploration mechanisms: Differential-style vectors between population members and occasional uniform random re-sampling within bounds.
# Exploitation mechanisms: Gaussian perturbations centered at the best-so-far with a shrinking step size, plus mixing with the current population members to encourage directed moves.
# Boundary handling: All generated points are clipped to the provided bounds; after clipping, candidates are re-used as-is for evaluation (no extra evaluations).
# Budget strategy: The number of objective evaluations is capped exactly at the provided budget. Initialization uses pop_size evaluations (or fewer if budget is tiny). Each subsequent iteration schedules at most pop_size new evaluations, stopping as soon as the budget is exhausted.
# Closest known influences: A blend of (1) (μ+λ)-style selection, (2) differential evolution-inspired mutation (using population differences), and (3) CMA-ES-like step-size decay without covariance adaptation, implemented in a minimal form.
# Novelty or unusual aspects: The algorithm uses a deterministic evaluation schedule that adapts exploration intensity based on observed improvement frequency, while keeping the implementation short and dimension-robust.
# Failure modes: If the objective is extremely noisy or highly deceptive, the population may converge prematurely; boundary clipping can also reduce effective movement in narrow feasible regions. The algorithm mitigates these via occasional global re-sampling and exploration mixing.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0 or dim <= 0:
            # No evaluations possible; return a zero vector by convention.
            return np.zeros(dim, dtype=float), np.inf

        # Read bounds from func attributes as specified.
        lb = ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and func.bounds is not None:
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        if lb is None or ub is None:
            # If bounds are not provided, fall back to [-1, 1] hypercube.
            # (Still safe and dimension-robust.)
            lb = -np.ones(dim, dtype=float)
            ub = np.ones(dim, dtype=float)

        # Ensure correct shapes and ordering
        lb = np.broadcast_to(lb, (dim,)).astype(float, copy=False)
        ub = np.broadcast_to(ub, (dim,)).astype(float, copy=False)
        # If misordered, swap per-dimension.
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        lb, ub = lo, hi
        span = ub - lb
        span = np.where(span == 0.0, 1.0, span)  # avoid division issues

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        # Population size: keep small to allow budget usage in later refinements.
        # Ensure pop_size >= 2 for differential-like operations.
        pop_size = int(np.clip(4 + dim, 4, 40))
        pop_size = min(pop_size, budget) if budget > 0 else 0
        if pop_size < 2:
            pop_size = 1

        # Initialize population uniformly in bounds.
        X = lb + span * np.random.rand(pop_size, dim)
        best_x = None
        best_y = np.inf
        evals = 0

        def eval_one(x):
            nonlocal evals, best_x, best_y
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.array(x, dtype=float, copy=True)
            return y

        # Evaluate initial population (up to budget)
        y = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            if evals >= budget:
                break
            y[i] = eval_one(X[i])

        # If budget was smaller than pop_size, shrink population to evaluated part.
        if evals < pop_size:
            X = X[:evals]
            y = y[:evals]
            pop_size = len(X)

        # If nothing evaluated, return default.
        if evals == 0:
            return np.zeros(dim, dtype=float), np.inf

        # Best index for exploitation sampling
        def argmin_idx(arr):
            return int(np.argmin(arr))

        no_improve_steps = 0
        prev_best = best_y

        # Main loop: each round can evaluate up to pop_size trials
        while evals < budget:
            # Fraction used in [0,1)
            t = evals / max(1, budget)
            # Shrinking step size: start ~ 10% of range, end tiny
            sigma = 0.1 * (1.0 - t) + 1e-12
            # Exploration mixing increases when no improvement
            stall_factor = 1.0 + (no_improve_steps / 10.0)
            explore_prob = min(0.7, 0.15 * stall_factor + 0.1 * t)  # bounded

            best_idx = argmin_idx(y)
            x_best = X[best_idx]

            # Build trial points
            # We'll attempt up to pop_size evaluations but stop when budget is exhausted.
            trials = []
            parents = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                xi = X[i]

                r = np.random.rand()
                if r < explore_prob and pop_size >= 3:
                    # Differential-style exploration: xi + F*(x_a - x_b)
                    a, b = np.random.choice(pop_size, size=2, replace=False)
                    xa, xb = X[a], X[b]
                    F = 0.5 + 0.5 * np.random.rand()  # in [0.5,1.0)
                    # Also bias slightly towards best to reduce wandering
                    mix = 0.3 + 0.5 * np.random.rand()
                    direction = F * (xa - xb)
                    cand = xi + mix * direction + (1.0 - mix) * (x_best - xi)
                elif r < 0.9:
                    # Exploitation: Gaussian around best and optionally towards xi
                    # Local step magnitude scales with remaining budget fraction.
                    # Use isotropic Gaussian in all dimensions.
                    g = np.random.randn(dim)
                    local = x_best + sigma * span * g
                    # Blend with current parent to keep diversity near promising areas
                    blend = 0.2 + 0.6 * np.random.rand()
                    cand = blend * local + (1.0 - blend) * xi
                else:
                    # Pure random restart within bounds
                    cand = lb + span * np.random.rand(dim)

                cand = clip(cand)
                trials.append(cand)
                parents.append(i)

            # Evaluate and select/replacement
            for cand, i in zip(trials, parents):
                if evals >= budget:
                    break
                yc = eval_one(cand)
                # Replace if better
                if yc < y[i]:
                    X[i] = cand
                    y[i] = yc

                # Track stagnation for adaptation
                if best_y < prev_best - 1e-12:
                    prev_best = best_y
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1

            # If we somehow exhausted all evaluations, exit.
            if evals >= budget:
                break

        # Ensure best_x is set
        if best_x is None:
            best_x = X[argmin_idx(y)]
            best_y = float(np.min(y)) if len(y) else np.inf

        return best_x, best_y
