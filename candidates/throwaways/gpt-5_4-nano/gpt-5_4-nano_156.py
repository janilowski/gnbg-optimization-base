import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box optimizer for minimization based on
# a simple evolutionary strategy with optional restart. It maintains a small
# population, samples offspring using Gaussian mutations around the current
# best, and replaces the population each generation while strictly tracking
# the evaluation budget.
# Search state: Tracks evaluation count, current best point/value, and a
# population of candidate solutions with associated fitness values.
# Candidate generation: Generates offspring by adding normally distributed
# noise to parent vectors. The mutation scale adapts to success by comparing
# offspring fitness to the current best.
# Selection and replacement: Uses (1+λ) style replacement within a generation:
# if any offspring improves the best, the best is updated and a new population
# is formed around the updated best; otherwise the population is kept but
# shrinks its mutation scale over time.
# Adaptation: Mutation step size (sigma) is adjusted adaptively. It increases
# slightly when improvements happen and decreases when progress stalls.
# Exploration mechanisms: Larger sigma early on and occasional random restart
# helps explore new regions.
# Exploitation mechanisms: After improvements, sampling centers on the best
# discovered solution and sigma is reduced to refine.
# Boundary handling: Clamps all sampled candidates to the provided box bounds.
# Budget strategy: The algorithm never evaluates more than `budget` points
# total, including initial population evaluations.
# Closest known influences: Inspired by classic evolution strategies / CMA-like
# adaptation but kept lightweight and dimension-agnostic (uses diagonal
# Gaussian sampling).
# Novelty or unusual aspects: Uses a strict budget-aware evaluation loop with
# generation-sized offspring and step-size success heuristics, aiming to be
# robust under small budgets.
# Failure modes: If the objective is extremely noisy or adversarial to
# mutation directions, progress may stall; the restart mechanism mitigates but
# cannot guarantee improvement. For very small budgets, the method may behave
# like random search.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # --- Read bounds from func ---
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/ub.")

        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Ensure finite bounds where possible
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)
        span = ub - lb
        # If span is degenerate, avoid zero step size
        span_norm = np.max(np.abs(span))
        if not np.isfinite(span_norm) or span_norm <= 0:
            span_norm = 1.0

        # Helper: clamp to bounds
        def clamp(x):
            return np.minimum(np.maximum(x, lb), ub)

        # Evaluation wrapper with strict budget accounting
        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                # Should not happen if logic is correct
                return float("inf")
            x = clamp(np.asarray(x, dtype=float))
            y = func(x)
            evals += 1
            # Objective should be minimization; ensure scalar float
            return float(y)

        # If budget is extremely small, do minimal work
        if budget <= 0:
            x0 = np.clip(np.zeros(dim), lb, ub)
            return x0, float("inf")

        # --- Initialization ---
        # Choose population size based on budget and dimension (keep compact).
        # We want enough offspring to select improvements but not exceed budget.
        lam = int(np.clip(4 + dim, 4, 32))
        # Ensure we can evaluate at least 1 point
        lam = min(lam, max(1, budget - 1)) if budget > 1 else 1

        # Initial sigma based on bounds
        # Start relatively broad to explore, but not too large for tiny spans.
        sigma = 0.25 * span_norm / np.sqrt(dim + 1.0)
        if sigma <= 0 or not np.isfinite(sigma):
            sigma = 0.25

        # Initialize best with a random point inside bounds
        x_best = lb + np.random.rand(dim) * (ub - lb)
        x_best = clamp(x_best)
        y_best = evaluate(x_best)

        # Optional initial population around best (only if budget allows)
        # This helps quickly gather selection signal.
        remaining = budget - evals
        if remaining > 0 and lam > 1:
            pop_size = min(lam, remaining)
            # Build a population of candidates
            pop = np.empty((pop_size, dim), dtype=float)
            fit = np.empty(pop_size, dtype=float)
            for i in range(pop_size):
                # sample around best with broader sigma
                step = np.random.randn(dim) * sigma
                xi = clamp(x_best + step)
                pop[i] = xi
                fit[i] = evaluate(xi)
                # Track best
                if fit[i] < y_best:
                    y_best = float(fit[i])
                    x_best = xi.copy()

            # Use pop as current population center for subsequent sampling
            # (we'll sample offspring around x_best anyway)
        else:
            pop = None
            fit = None

        # --- Main loop ---
        # We run as many generations as budget allows. Each generation evaluates `k` offspring.
        # Use dynamic k to not overshoot budget.
        while evals < budget:
            # Decide offspring count for this generation
            remaining = budget - evals
            # Evaluate at most lam offspring per generation
            k = min(lam, remaining)
            if k <= 0:
                break

            # Parent center: current best for exploitation
            center = x_best

            # Generate offspring
            # Use slightly different scales per offspring to maintain diversity
            # and use diagonal Gaussian perturbations.
            # Candidate points are clipped to bounds.
            improved = False
            best_off_y = y_best
            best_off_x = x_best

            # Mutation scale may shrink after stalls; also scale with span
            # to stay dimension-robust.
            local_sigma = sigma
            if not np.isfinite(local_sigma) or local_sigma <= 0:
                local_sigma = 0.25

            # Evaluate offspring
            for _ in range(k):
                # Sample direction with standard normal
                # and optionally rescale magnitude by random chi-like factor
                z = np.random.randn(dim)
                # Use occasional larger jump to encourage exploration
                if np.random.rand() < 0.1:
                    jump = 2.5
                else:
                    jump = 1.0
                xi = center + (z * local_sigma) * jump
                yi = evaluate(xi)
                if yi < best_off_y:
                    best_off_y = yi
                    best_off_x = clamp(xi)
                    improved = True

            # Update best
            if improved:
                x_best = best_off_x
                y_best = float(best_off_y)
                # Slightly increase sigma to keep exploring around improvements
                sigma *= 1.05
            else:
                # If no improvement, shrink sigma to exploit locally
                sigma *= 0.85

            # Occasional restart if sigma gets too small relative to bounds,
            # to avoid being stuck in a poor basin (only if budget remains).
            if evals < budget:
                # Use relative threshold
                if sigma < 1e-12 or sigma < 1e-3 * span_norm / np.sqrt(dim + 1.0):
                    if np.random.rand() < 0.3:
                        # Restart: sample new center uniformly, reset sigma
                        # but only update x_best if evaluated point improves.
                        x_new = lb + np.random.rand(dim) * (ub - lb)
                        y_new = evaluate(x_new)
                        if y_new < y_best:
                            x_best = x_new
                            y_best = float(y_new)
                        sigma = max(0.25 * span_norm / np.sqrt(dim + 1.0), 1e-6)

        return x_best, y_best
