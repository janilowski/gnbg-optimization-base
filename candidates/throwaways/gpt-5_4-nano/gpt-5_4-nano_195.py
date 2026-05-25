import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimization algorithm
# using a population of candidate points and a trust-region-like adaptive step
# size. It combines global exploration with local exploitation by using
# differential-style mutations and a local Gaussian sampler around the current best.
# Search state: Maintains a population of points and their objective values, plus
# the current best solution found so far. Tracks remaining evaluations to ensure
# the budget is never exceeded.
# Candidate generation: For each generation, it creates offspring using either
# (1) a differential evolution-like vector difference mutation among population
# members, or (2) a local Gaussian perturbation centered at the best point. Candidate
# creation uses per-dimension bound scaling and clips to feasible space.
# Selection and replacement: Offspring are evaluated; if an offspring improves over
# the parent it replaces it (greedy replacement). The global best is updated whenever
# a new best is found.
# Adaptation: The local step size shrinks when improvements are rare and expands
# slightly when improvements occur, based on a simple improvement counter.
# Exploration mechanisms: Differential-style mutations provide broad exploration,
# especially with a decaying exploration probability to shift toward exploitation.
# Exploitation mechanisms: Local sampling around the best point with an adaptive
# step size improves convergence on smooth-ish problems.
# Boundary handling: Uses clipping to the provided lower/upper bounds (or bounds.lb/ub).
# Budget strategy: Computes a safe maximum number of evaluations up-front and
# allocates them across initial population evaluation and subsequent generations.
# It strictly checks evaluation count before every objective call to never exceed
# the provided budget.
# Closest known influences: Inspired by differential evolution and CMA-ES-like
# step-size control, simplified for a small, standard-library/NumPy-only module.
# Novelty or unusual aspects: A hybrid offspring generator that mixes DE-style and
# local best-centered sampling with a lightweight adaptive schedule tuned to remaining
# budget and observed improvement frequency.
# Failure modes: If the objective is extremely noisy or highly discontinuous, the
# greedy replacement and step-size adaptation may stagnate; if bounds are very tight,
# diversity may collapse quickly due to clipping.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _get_bounds(self, func):
        # Support: func.lower/func.upper or func.bounds.lb/func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub.")
        if lb.shape == () or ub.shape == ():
            lb = np.full(self.dim, float(lb))
            ub = np.full(self.dim, float(ub))
        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError("Bounds must match the provided dimension.")
        return lb, ub

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed: return a valid point but unknown value.
            # The harness expects (best_x, best_y); we choose best_y as +inf.
            lb, ub = self._get_bounds(func)
            x0 = np.clip(np.random.rand(dim) * (ub - lb) + lb, lb, ub)
            return x0, float("inf")

        lb, ub = self._get_bounds(func)
        span = ub - lb
        # Handle degenerate dimensions where span=0
        span_safe = np.where(span == 0, 1.0, span)

        evals = 0

        # Population size: small enough to keep evaluations under budget.
        # Try to scale with dimension while remaining compact.
        # At minimum keep 4 for DE-like mutation.
        pop_size = int(np.clip(4 + 2 * int(np.sqrt(dim)), 4, 30))
        # Ensure we can evaluate at least the initial population within budget.
        pop_size = min(pop_size, budget)

        # Generate initial population uniformly in bounds.
        pop = lb + span * np.random.rand(pop_size, dim)
        pop = np.clip(pop, lb, ub)

        # Evaluate initial population
        vals = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            if evals >= budget:
                break
            vals[i] = float(func(pop[i]))
            evals += 1

        # Track global best
        best_idx = int(np.argmin(vals))
        best_x = pop[best_idx].copy()
        best_y = float(vals[best_idx])

        # If budget exhausted during initial eval
        if evals >= budget:
            return best_x, best_y

        # Remaining evaluations for generations
        # Use a generation scheme with one offspring per individual.
        # But adjust offspring count as budget shrinks.
        max_generations = 1 + (budget - evals) // pop_size

        # Step size for local exploitation (relative to span)
        # Start moderately and adapt with improvements.
        step = 0.25 * span_safe
        min_step = 1e-12 * span_safe

        # Parameters controlling exploration/exploitation balance
        # exploration_prob decreases as budget is used up
        de_F = 0.7
        de_CR = 0.9  # crossover rate for DE-style trial
        local_sigma_shrink = 0.85
        local_sigma_expand = 1.1
        improvements_in_window = 0
        window = max(4, dim // 2)

        # Main loop
        generation = 0
        while evals < budget and generation < max_generations:
            generation += 1

            # Probability of using DE-like mutation vs local best sampling
            frac_used = evals / float(budget)
            exploration_prob = max(0.15, 0.7 * (1.0 - frac_used))

            for i in range(pop_size):
                if evals >= budget:
                    break

                x_parent = pop[i]
                y_parent = vals[i]

                use_exploration = (np.random.rand() < exploration_prob)
                x_trial = None

                if use_exploration and pop_size >= 3:
                    # Differential evolution style:
                    # choose a,b,c distinct indices excluding i
                    idxs = np.arange(pop_size)
                    # Exclude i
                    mask = idxs != i
                    candidates = idxs[mask]
                    # In rare cases, fallback to random selection with replacement
                    if candidates.size >= 3:
                        a, b, c = np.random.choice(candidates, size=3, replace=False)
                    else:
                        # fallback: allow replacement if population too small
                        a, b, c = np.random.choice(idxs, size=3, replace=True)
                        if a == i or b == i or c == i:
                            # still acceptable; DE can handle some duplicates
                            pass

                    # Mutation
                    y_mut = pop[a] + de_F * (pop[b] - pop[c])

                    # Binomial crossover with parent
                    cross = np.random.rand(dim) < de_CR
                    if not np.any(cross):
                        cross[np.random.randint(0, dim)] = True
                    x_trial = np.where(cross, y_mut, x_parent)

                else:
                    # Local exploitation around current best
                    # Mix a Gaussian perturbation with occasional coordinate-wise resampling
                    # to help escape shallow local basins.
                    # Adaptive step size
                    sigma = np.maximum(step, min_step)

                    # Gaussian around best
                    x_trial = best_x + np.random.randn(dim) * sigma

                    # With small probability, resample some coordinates uniformly
                    if np.random.rand() < 0.25:
                        k = max(1, int(0.1 * dim))
                        coords = np.random.choice(dim, size=k, replace=False)
                        x_trial[coords] = lb[coords] + span[coords] * np.random.rand(k)

                # Bound handling: clip to feasible space
                x_trial = np.clip(x_trial, lb, ub)

                # Evaluate safely within remaining budget
                if evals >= budget:
                    break
                y_trial = float(func(x_trial))
                evals += 1

                # Greedy replacement
                if y_trial < y_parent:
                    pop[i] = x_trial
                    vals[i] = y_trial
                    y_parent = y_trial

                    # Update global best
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = x_trial.copy()
                        improvements_in_window += 1

            # Adaptation based on progress (simple windowed scheme)
            if improvements_in_window > 0:
                # Improvement occurred: slightly expand step to continue exploration/exploitation balance
                step = np.minimum(span_safe * 0.5, step * local_sigma_expand)
            else:
                # No improvement in this generation: shrink step to refine around best
                step = np.maximum(min_step, step * local_sigma_shrink)

            # Decay improvement counter
            if improvements_in_window >= window:
                improvements_in_window = 0
            else:
                # Keep but slowly decrease influence
                improvements_in_window = max(0, improvements_in_window - 1)

        return best_x, best_y
