# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free global minimizer for black-box
# functions using an adaptive population of candidate points, with rank-based
# selection and occasional exploratory restarts. It is designed to respect a
# strict evaluation budget while remaining robust across dimensions.
# Search state: Maintains a population of candidate vectors (x) and their
# objective values (y), along with an evaluation counter and the current best
# solution seen so far.
# Candidate generation: Uses a mixture of (1) Gaussian perturbations around
# the best point and (2) differential-style jumps using differences between
# population members. Step sizes adapt based on observed improvements.
# Selection and replacement: After evaluating new candidates, it merges them with
# the existing population and keeps the best individuals by objective value.
# Adaptation: Maintains a step scale that shrinks after stagnation and expands
# slightly after improvements, allowing transitions between exploration and
# exploitation.
# Exploration mechanisms: Includes random restarts of part of the population
# when progress stalls, and uses relatively large moves via differential jumps.
# Exploitation mechanisms: Uses small Gaussian steps around the current best
# and uses differential jumps that can pull candidates toward better regions.
# Boundary handling: New points are clipped to the provided bounds (lower/upper)
# to ensure feasibility in all coordinates.
# Budget strategy: Every objective call increments a counter; the algorithm
# stops generating/evaluating candidates once the budget would be exceeded.
# Closest known influences: Inspired by CMA/DE-style hybrids and rank-based
# selection, but implemented in a minimal, budget-safe form.
# Novelty or unusual aspects: Combines rank-based environmental selection with a
# budget-aware evaluation loop and adaptive restarts, emphasizing robustness
# and simplicity.
# Failure modes: If the objective is extremely noisy or has deceptive local
# minima, the algorithm may stagnate; restarts and step adaptation mitigate but
# cannot guarantee global optimality under tight budgets.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = max(0, int(self.budget))
        if budget == 0:
            # No evaluations allowed; return a feasible point arbitrarily.
            x0 = self._sample_within_bounds(func, rng=np.random)
            return x0, np.inf

        lb, ub = self._get_bounds(func)
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != dim or ub.size != dim:
            # If bounds are scalars, broadcast.
            if lb.size == 1 and ub.size == 1:
                lb = np.full(dim, float(lb[0]))
                ub = np.full(dim, float(ub[0]))
            else:
                raise ValueError("Bounds dimension mismatch with dim.")

        if np.any(ub < lb):
            raise ValueError("Invalid bounds: some ub < lb.")

        rng = np.random

        # Evaluation wrapper to guarantee budget compliance.
        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                # Should not happen due to checks, but keep robust.
                return np.inf
            y = float(func(x))
            evals += 1
            return y

        # Population size: modest scaling with dimension but capped.
        # Keeps iterations budget-aware and computationally light.
        pop_size = int(min(12 + dim, 40))
        pop_size = max(4, pop_size)
        # Number of new candidates per generation.
        new_per_gen = max(1, min(pop_size, dim // 2 + 2))

        # Initialize population uniformly in bounds.
        pop = self._random_population(pop_size, lb, ub, rng=rng)
        y_pop = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            if evals >= budget:
                break
            y_pop[i] = evaluate(pop[i])

        # Track best so far.
        best_idx = int(np.argmin(y_pop))
        best_x = pop[best_idx].copy()
        best_y = float(y_pop[best_idx])

        # Adaptive step size: based on bounds scale.
        span = np.maximum(ub - lb, 1e-12)
        sigma = 0.35 * span  # vector step scale

        # Stagnation counter and restarts.
        no_improve = 0
        stall_limit = max(5, int(2 + 0.1 * dim))

        # Budget-aware loop: generate/evaluate until budget runs out.
        # Each "iteration" evaluates up to new_per_gen candidates.
        max_iters = max(1, (budget - pop_size) // max(1, new_per_gen) + 5)
        for _ in range(max_iters):
            if evals >= budget:
                break

            # Sort population by fitness (ascending).
            order = np.argsort(y_pop)
            pop = pop[order]
            y_pop = y_pop[order]

            # Use best and some elites for candidate generation.
            elite = pop[: max(2, pop_size // 4)]
            elite_y = y_pop[: elite.shape[0]]
            cur_best = pop[0]
            cur_best_y = float(y_pop[0])

            # Determine whether to explore more (stagnation).
            progress = (best_y - cur_best_y)  # <= 0 typically
            if progress < -1e-12:
                no_improve = 0
                best_y = cur_best_y
                best_x = cur_best.copy()
                # Slightly increase sigma after improvement to keep motion.
                sigma = np.minimum(sigma * 1.05, 0.75 * span)
            else:
                no_improve += 1
                # Shrink sigma on stagnation to exploit locally.
                if no_improve >= 1:
                    sigma = np.maximum(sigma * 0.92, 1e-12)

            # If stalled, restart part of the population to regain diversity.
            # (Only affects candidate generation, not immediate replacement.)
            do_restart = no_improve >= stall_limit

            candidates = []
            # Mix strategies:
            # - Gaussian around best (exploitation)
            # - Differential jump using differences (exploration-like but directed)
            # - Random reinitializations upon restart
            for j in range(new_per_gen):
                if evals + len(candidates) >= budget:
                    break

                r = rng.rand()
                if do_restart and j < max(1, new_per_gen // 3):
                    # Restart: sample uniformly.
                    x = lb + rng.rand(dim) * (ub - lb)
                elif r < 0.55:
                    # Gaussian step around current best.
                    x = cur_best + rng.normal(0.0, 1.0, size=dim) * sigma
                else:
                    # Differential-style jump:
                    # pick three individuals a,b,c and set x = best + F*(b-c) + noise
                    # Choose indices from population.
                    a, b, c = rng.randint(0, pop.shape[0], size=3)
                    base = pop[a]
                    F = 0.6 + 0.4 * rng.rand()
                    x = base + F * (pop[b] - pop[c]) + rng.normal(0.0, 0.5, size=dim) * sigma * 0.25
                    # Bias toward best occasionally
                    if rng.rand() < 0.5:
                        x = 0.7 * x + 0.3 * cur_best

                # Boundary handling: clip to bounds.
                x = np.clip(x, lb, ub)
                candidates.append(x)

            if not candidates:
                break

            cand = np.asarray(candidates, dtype=float)
            y_cand = np.empty(cand.shape[0], dtype=float)

            # Evaluate candidates with budget checks.
            for i in range(cand.shape[0]):
                if evals >= budget:
                    y_cand = y_cand[:i]
                    cand = cand[:i]
                    break
                y_cand[i] = evaluate(cand[i])

            if cand.shape[0] == 0:
                break

            # Update best.
            idx = int(np.argmin(y_cand))
            if float(y_cand[idx]) < best_y:
                best_y = float(y_cand[idx])
                best_x = cand[idx].copy()
                no_improve = 0

            # Environmental selection: merge and keep best pop_size.
            merged = np.vstack([pop, cand])
            merged_y = np.concatenate([y_pop, y_cand])

            keep = min(pop_size, merged.shape[0])
            order2 = np.argsort(merged_y)[:keep]
            pop = merged[order2]
            y_pop = merged_y[order2]

            if do_restart:
                # After a restart trigger, reset stagnation counter so we don't restart every time.
                no_improve = max(0, no_improve - stall_limit // 2)
                # Increase exploration slightly after restart.
                sigma = np.minimum(sigma * 1.2, 0.9 * span)

        return best_x, best_y

    def _get_bounds(self, func):
        # Accept either func.lower/upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = getattr(func, "lower")
            ub = getattr(func, "upper")
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = getattr(func.bounds, "lb")
            ub = getattr(func.bounds, "ub")
        else:
            raise AttributeError("Function does not provide bounds as lower/upper or bounds.lb/bounds.ub.")
        return lb, ub

    def _random_population(self, pop_size, lb, ub, rng=np.random):
        # Uniform random points within bounds.
        r = rng.rand(pop_size, lb.size)
        return lb + r * (ub - lb)

    def _sample_within_bounds(self, func, rng=np.random):
        lb, ub = self._get_bounds(func)
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size == 1 and ub.size == 1:
            lb = np.full(self.dim, float(lb[0]))
            ub = np.full(self.dim, float(ub[0]))
        elif lb.size != self.dim:
            # If mismatch, attempt to broadcast using first dim values.
            lb = np.resize(lb, self.dim)
            ub = np.resize(ub, self.dim)
        x = lb + rng.rand(self.dim) * (ub - lb)
        return np.clip(x, lb, ub)
