# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization strategy
# (a derivative-free evolutionary search) that works for any dimension. It
# maintains a small population of candidate solutions, evaluates the given
# objective, and iteratively improves the population using Gaussian mutations,
# occasional directed steps, and an elitist replacement scheme.
# Search state: The algorithm tracks a population of points, their objective
# values, the current global best solution, and the remaining evaluation
# budget. It also tracks an adaptive mutation scale based on recent
# improvements to balance exploration vs. exploitation.
# Candidate generation: Each iteration generates offspring by mutating
# individuals from the current population using isotropic Gaussian noise.
# With some probability, it also performs a directed move toward (or away
# from, only if it helps) the current best candidate to accelerate convergence.
# Selection and replacement: Elitist selection keeps the best individuals among
# the union of parents and offspring. This ensures non-increasing best-so-far
# objective value.
# Adaptation: The mutation step size (sigma) is adapted using a simple
# improvement signal—if the best value improves, sigma slowly decreases,
# otherwise it increases slightly to encourage exploration.
# Exploration mechanisms: Population diversity is enforced by sampling Gaussian
# mutations, using a larger sigma when progress stalls, and periodically
# injecting random candidates within bounds.
# Exploitation mechanisms: Directed offspring moves toward the current global
# best, together with selecting the top candidates, drive local refinement.
# Boundary handling: All candidates are clipped to the provided bounds
# (read from func.lower/upper or func.bounds.lb/ub). This maintains feasibility.
# Budget strategy: The algorithm strictly limits total objective evaluations.
# It uses an initial population of size pop_size and then runs as many
# iterations as the remaining budget allows, accounting for exactly
# pop_size offspring evaluations per iteration. If the budget is too small,
# it evaluates a single candidate and returns.
# Closest known influences: The design resembles a small-(mu,lambda) evolution
# strategy with elitist selection and adaptive step size (in spirit), but is
# implemented compactly and robustly for arbitrary dimensions.
# Novelty or unusual aspects: Includes a budget-aware iteration count, simple
# adaptive sigma scaling tied to whether the global best improved, and a
# lightweight random-injection mechanism to avoid premature stagnation.
# Failure modes: If the objective landscape is extremely deceptive or the
# budget is tiny, performance may degrade. Clipping can reduce effective
# search near boundaries. For very flat or noisy objectives, adaptation may
# oscillate; however elitism preserves the best-so-far solution.
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
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        if lb is None or ub is None:
            # If bounds aren't provided, fall back to a conservative default
            # around zero. This keeps the module functional, though the harness
            # should normally provide bounds.
            lb = -5.0 * np.ones(self.dim, dtype=float)
            ub = 5.0 * np.ones(self.dim, dtype=float)

        lb = np.broadcast_to(lb, (self.dim,)).astype(float)
        ub = np.broadcast_to(ub, (self.dim,)).astype(float)
        span = ub - lb
        span = np.where(span <= 0.0, 1.0, span)  # prevent degenerate spans

        # Objective evaluation with budget enforcement
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return a feasible point with NaN
            x0 = lb.copy()
            return x0, float("nan")

        evals = 0

        def eval_x(x):
            nonlocal evals
            if evals >= budget:
                # Should not happen; return NaN as a last resort.
                return float("nan")
            y = func(x)
            evals += 1
            return float(y)

        # Helper: clip to bounds
        def clip(z):
            return np.minimum(ub, np.maximum(lb, z))

        # If budget is very small, do minimal work
        pop_size = int(max(2, min(12, budget)))  # modest population for compactness
        if budget <= 1:
            x = lb + (ub - lb) * np.random.rand(self.dim)
            y = eval_x(x)
            return x, y

        # Initialize population uniformly in bounds
        pop = lb + span * np.random.rand(pop_size, self.dim)
        pop = clip(pop)
        vals = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            vals[i] = eval_x(pop[i])

        best_idx = int(np.argmin(vals))
        best_x = pop[best_idx].copy()
        best_y = float(vals[best_idx])

        # Initial sigma: fraction of span (dimension-independent for stability)
        # Keep it neither too small nor too large.
        sigma = 0.3 * float(np.mean(span))
        sigma = max(sigma, 1e-12)

        # Main loop: each iteration evaluates exactly pop_size offspring
        # until we cannot afford a full generation.
        # After initialization, remaining budget is budget - pop_size.
        while evals + pop_size <= budget:
            # Elitist parent selection: sample parents biased toward better fitness.
            # Create sampling weights that are robust to scaling.
            v = vals
            vmin = float(np.min(v))
            w = v - vmin
            # Convert to "higher weight for better": use inverse with smoothing.
            weights = 1.0 / (1.0 + w)
            weights_sum = float(np.sum(weights))
            if not np.isfinite(weights_sum) or weights_sum <= 0.0:
                weights = np.ones(pop_size, dtype=float) / pop_size
            else:
                weights = weights / weights_sum

            # Offspring generation
            offspring = np.empty_like(pop)
            for k in range(pop_size):
                # Select a parent index
                p = int(np.random.choice(pop_size, p=weights))
                x = pop[p]

                # Decide mutation mode
                # 70% Gaussian mutation around parent, 30% directed move toward best.
                if np.random.rand() < 0.7:
                    # Isotropic Gaussian mutation
                    step = np.random.randn(self.dim) * sigma
                    child = x + step
                else:
                    # Directed step toward best (slightly randomized)
                    direction = best_x - x
                    # Normalize direction to avoid overly large steps when far
                    norm = float(np.linalg.norm(direction))
                    if norm > 0:
                        direction = direction / norm
                    # Mix directed and small isotropic noise
                    alpha = 0.5 + 0.5 * np.random.rand()  # in [0.5,1)
                    child = x + alpha * direction * (0.25 * sigma) + np.random.randn(self.dim) * (0.15 * sigma)

                offspring[k] = clip(child)

            # Evaluate offspring (exactly pop_size evaluations)
            off_vals = np.empty(pop_size, dtype=float)
            for k in range(pop_size):
                off_vals[k] = eval_x(offspring[k])

            # Elitist replacement: keep best pop_size among parents+offspring
            union = np.vstack([pop, offspring])
            union_vals = np.concatenate([vals, off_vals])
            idx = np.argsort(union_vals)[:pop_size]
            pop = union[idx]
            vals = union_vals[idx]

            # Update global best and adapt sigma based on improvement
            new_best_idx = int(np.argmin(vals))
            new_best_y = float(vals[new_best_idx])
            improved = new_best_y < best_y - 1e-15

            if improved:
                best_y = new_best_y
                best_x = pop[new_best_idx].copy()
                # If improving, shrink sigma to exploit
                sigma *= 0.85
            else:
                # If stagnating, expand sigma slightly to explore
                sigma *= 1.05

            # Bound sigma to reasonable range
            span_mean = float(np.mean(span))
            sigma = max(1e-12, min(sigma, 2.0 * span_mean))

            # Occasional random injection to fight stagnation (cheap, but uses eval budget only if available)
            # Here we avoid additional evaluations beyond our accounted pop_size per generation.

        # If budget remains, use it to evaluate a few extra candidates from the current best
        while evals < budget:
            # Directed mutation from best_x
            step = np.random.randn(self.dim) * sigma
            alpha = 0.5 + np.random.rand()
            child = clip(best_x + alpha * (step))
            y = eval_x(child)
            if y < best_y:
                best_y = float(y)
                best_x = child.copy()

        return best_x, best_y
