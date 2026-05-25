import numpy as np


# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm using a mixture of
# (1) population-based random search and (2) local coordinate descent–style refinement
# around the current best. It is designed to work for any dimension and only relies
# on NumPy plus evaluation queries to the provided black-box objective.
# Search state: Maintains a small population of candidate solutions, tracks the best
# seen point (best_x) and its objective value (best_y), and keeps a global step size
# controlling how far candidates are sampled from the current best.
# Candidate generation: First phase samples candidates uniformly across the bounds.
# Later phases sample from a Gaussian around the current best, plus occasionally mutate
# additional diversity points. Local refinement uses coordinate-wise probes from the
# current best with diminishing step sizes.
# Selection and replacement: Each generation evaluates a batch of candidates, selects the
# best among them, and replaces the population around the best using the current step
# size. The global best is updated whenever a new lower objective value is found.
# Adaptation: Step size decays over time based on remaining budget, and the algorithm
# also adapts by increasing/decreasing how aggressively local refinement is performed
# depending on whether improvements were observed.
# Exploration mechanisms: Uniform sampling at the start, plus occasional wide Gaussian
# perturbations and diversity mutations to escape local minima.
# Exploitation mechanisms: Coordinate probes around the best point with a shrinking
# radius, attempting to improve each coordinate in turn.
# Boundary handling: All candidate points are clipped to the provided bounds after every
# perturbation to ensure feasibility.
# Budget strategy: The algorithm strictly never exceeds the provided evaluation budget.
# It estimates how many evaluations to spend per phase and truncates any final batch
# to fit the remaining budget.
# Closest known influences: The design loosely follows simple derivative-free strategies
# combining global random search, Gaussian sampling around the incumbent, and coordinate
# descent–like local refinement (similar in spirit to CMA-ES-like sampling but much
# simpler and budget-aware).
# Novelty or unusual aspects: Uses a deterministic budget accounting mechanism to
# interleave population sampling and coordinate refinement without needing hyperparameter
# tuning beyond a few robust constants; works for arbitrary bounds provided by the
# function object.
# Failure modes: If the objective has extremely narrow feasible improvements or is highly
# deceptive, random exploration may miss good regions; boundary clipping can reduce
# effective search near tight bounds.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        # Robust defaults that scale with dimension and budget.
        # Keep small to reduce per-iteration overhead; batch evaluation is still sequential.
        self.pop_scale = 8  # population size multiplier before falling back to 1..k
        self.min_step = 1e-12  # avoid numerical stagnation

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        if budget <= 0:
            x0 = np.zeros(dim, dtype=float)
            y0 = float(func(x0))
            return x0, y0

        lower, upper = self._get_bounds(func, dim)
        span = upper - lower
        # If bounds are degenerate in a dimension, span might be 0: avoid zero-width issues.
        span_safe = np.where(span > 0, span, 1.0)

        rng = np.random.default_rng()

        # Helper to clip to bounds.
        def clip(x):
            return np.minimum(upper, np.maximum(lower, x))

        # Budget accounting
        evals = 0

        def eval_one(x):
            nonlocal evals
            # Ensure feasibility
            xx = clip(np.asarray(x, dtype=float))
            y = float(func(xx))
            evals += 1
            return xx, y

        # Choose evaluation budget split between global sampling and local refinement.
        # Global phase: a bit more when budget is larger.
        # Ensure at least a few evaluations for local improvement.
        global_budget = int(min(budget, max(3 * dim, budget * 0.55)))
        local_budget = budget - global_budget
        # If budget is tiny, global_budget may consume all.
        if global_budget < 1:
            global_budget = min(budget, 1)
        if local_budget < 0:
            local_budget = 0

        # Initialize incumbents
        best_x = None
        best_y = np.inf

        # ---------- Global exploration phase ----------
        # Population size chosen to fit remaining budget.
        # Prefer pop in [1, 2*dim] depending on budget.
        base_pop = int(max(1, min(2 * dim, self.pop_scale * max(1, dim // 2))))
        pop = max(1, min(base_pop, global_budget))

        # Step size for Gaussian around best
        # Start with something like 0.5*span (but robust).
        step0 = 0.5 * span_safe

        # Evaluate at least one random point to bootstrap.
        x = lower + rng.random(dim) * span_safe
        x, y = eval_one(x)
        best_x, best_y = x, y

        # Remaining global evaluations
        while evals < global_budget:
            remaining = global_budget - evals
            k = min(pop, remaining)
            candidates = lower + rng.random((k, dim)) * span_safe

            # Add one candidate with slight bias toward current best to speed up
            # when we're already in a promising region.
            if k >= 2:
                # Standard deviation shrinks with global progress.
                prog = evals / max(1, global_budget)
                step = step0 * (1.0 - 0.5 * prog)
                # Use Gaussian perturbation around best.
                candidates[0] = best_x + rng.normal(0.0, step, size=dim)

            for i in range(k):
                xx, yy = eval_one(candidates[i])
                if yy < best_y:
                    best_y, best_x = yy, xx

            # Optionally reduce population slightly as global phase progresses.
            if pop > 1 and evals > global_budget * 0.7:
                pop = max(1, pop // 2)

        # ---------- Local refinement phase ----------
        # Coordinate-wise probes with diminishing step size.
        # Also occasionally do multivariate Gaussian sampling for escape.
        remaining = budget - evals
        if remaining <= 0:
            return best_x, best_y

        # Local step schedule
        t = 0
        step = 0.2 * span_safe
        # Ensure at least some progress.
        step = np.maximum(step, self.min_step)

        improved_in_cycle = True

        # Determine how many "cycles" we can afford: each cycle probes up to dim coords,
        # but we truncate using remaining budget.
        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break

            # Occasionally do a Gaussian burst (exploration) instead of full coord loop.
            # The probability decreases as budget shrinks.
            p_explore = 0.25 * (remaining / max(1, budget))
            if rng.random() < p_explore:
                # Evaluate a small batch around best.
                k = min(3, remaining)
                # Step size based on current schedule
                local_step = step * (1.0 + 0.5 * rng.random())
                for _ in range(k):
                    cand = best_x + rng.normal(0.0, local_step, size=dim)
                    xx, yy = eval_one(cand)
                    if yy < best_y:
                        best_y, best_x = yy, xx
                # Decay step after exploration
                step *= 0.9
                t += 1
                continue

            # Coordinate refinement: try moving each coordinate in + and - direction.
            # Diminish step as cycles proceed; adapt based on success.
            # Order coordinates randomly to avoid systematic bias.
            order = rng.permutation(dim)

            # Number of coordinate-probes we can do in this iteration.
            # Each coordinate can use up to 2 evaluations (+ and -), but we stop by budget.
            for j in order:
                if evals >= budget:
                    break

                # If step is too small, break out early.
                if np.all(step <= self.min_step):
                    break

                step_j = step[j] if np.ndim(step) > 0 else step
                step_j = max(float(step_j), self.min_step)

                # Try + direction
                x_try = best_x.copy()
                x_try[j] = x_try[j] + step_j
                xx, yy = eval_one(x_try)
                if yy < best_y:
                    best_y, best_x = yy, xx
                    improved_in_cycle = True
                    # Slightly increase exploitation after success
                    step *= 0.95
                else:
                    # Try - direction
                    x_try[j] = best_x[j] - step_j
                    xx, yy = eval_one(x_try)
                    if yy < best_y:
                        best_y, best_x = yy, xx
                        improved_in_cycle = True
                        step *= 0.95
                    else:
                        # No improvement on this coordinate: shrink step for that coordinate
                        improved_in_cycle = False
                        # Shrink this coordinate's step more aggressively
                        if np.ndim(step) > 0:
                            step[j] = max(step[j] * 0.5, self.min_step)
                        else:
                            step = max(step * 0.5, self.min_step)

            # After a coordinate pass, global step decay
            if improved_in_cycle:
                step *= 0.90
            else:
                step *= 0.75

            # Ensure step doesn't vanish too quickly to allow escape with remaining budget.
            if evals < budget * 0.2:
                step = np.maximum(step, 1e-6 * span_safe)

            t += 1

            # Occasionally re-randomize step if it becomes too small early.
            if t % 3 == 0 and evals < budget * 0.5:
                step = np.maximum(step, 0.05 * span_safe)

        return best_x, best_y

    @staticmethod
    def _get_bounds(func, dim):
        # Bounds can be provided in multiple common formats:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        # Each may be list/tuple/np array/scalar. We convert to arrays.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = getattr(func, "lower")
            upper = getattr(func, "upper")
        elif hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            lower = getattr(b, "lb", None)
            upper = getattr(b, "ub", None)
        else:
            # If bounds are not provided, default to [-5, 5] per dimension.
            # However, task requires bounds to be read from the objective;
            # this fallback aims robustness.
            lower = -5.0
            upper = 5.0

        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        if lower.shape == () and dim > 1:
            lower = np.full(dim, float(lower), dtype=float)
        if upper.shape == () and dim > 1:
            upper = np.full(dim, float(upper), dtype=float)

        if lower.shape != (dim,) or upper.shape != (dim,):
            # Attempt to broadcast/reshape if user gave length-1 arrays.
            lower = np.reshape(lower, (-1,))
            upper = np.reshape(upper, (-1,))
            if lower.size == 1:
                lower = np.full(dim, float(lower[0]), dtype=float)
            if upper.size == 1:
                upper = np.full(dim, float(upper[0]), dtype=float)

        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        # Safety: if any bound is inverted, swap.
        lo = np.minimum(lower, upper)
        hi = np.maximum(lower, upper)
        return lo, hi
