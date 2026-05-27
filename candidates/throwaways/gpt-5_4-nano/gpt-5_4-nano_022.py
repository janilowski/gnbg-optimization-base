# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# (a global-to-local evolutionary search with restarts). It keeps a small
# population of candidate points inside the provided box bounds, evaluates
# them against the black-box objective, then iteratively generates offspring
# through Gaussian mutations whose scale adapts to the observed progress.
# Search state: The algorithm tracks the best-so-far solution (best_x, best_y),
# current evaluation budget remaining, and a population of candidate points and
# their fitness values.
# Candidate generation: Each iteration selects a few parents biased toward the
# better individuals and creates offspring by (1) mixing parents (weighted
# recombination) and (2) adding Gaussian noise scaled by an adaptive step
# size. Periodically, it also injects new random points (restart/diversify)
# to reduce premature convergence.
# Selection and replacement: Offspring are evaluated and compete with the
# existing population. Replacement uses a simple elitist strategy: the best
# individuals are retained, ensuring the population size stays constant while
# best-so-far never worsens.
# Adaptation: The mutation step size adapts based on relative improvement:
# it shrinks after good progress (to refine) and grows modestly after stagnation
# (to escape). The algorithm also adjusts diversity via restart injections.
# Exploration mechanisms: Gaussian mutations with a step size, weighted
# recombination, and occasional random restarts across the full domain.
# Exploitation mechanisms: Selection pressure toward the best individuals,
# shrinking step size on improvement, and local refinement around the current
# best through recombination.
# Boundary handling: All candidate points are clipped to the provided
# lower/upper bounds after every generation step.
# Budget strategy: The algorithm never exceeds the provided evaluation budget.
# It pre-allocates an initial population and then performs a fixed number of
# evaluation batches, stopping early if the budget would be exceeded.
# Closest known influences: A blend of CMA-ES-like step-size control (simplified),
# evolutionary strategy selection with elitism, and global restart behavior.
# Novelty or unusual aspects: The implementation is intentionally minimal while
# still robust: it uses dimension-scaled step sizes, a generic parent mixing
# scheme, and a safe budget accounting model.
# Failure modes: If the budget is extremely small (e.g., too small for initial
# population), the search degrades to near-random sampling. In highly
# adversarial landscapes, step-size adaptation may oscillate; restarts mitigate
# but cannot guarantee optimality.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")

    def __call__(self, func):
        # Read bounds from func
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective must provide bounds via lower/upper or bounds.lb/bounds.ub")

        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            lb = np.asarray(lb, dtype=float).reshape(-1)
            ub = np.asarray(ub, dtype=float).reshape(-1)
            if lb.size != self.dim or ub.size != self.dim:
                raise ValueError("Bounds must match the provided dimension")

        # Ensure valid ordering
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        lb = lb2
        ub = ub2

        rng = np.random

        # Budget-safe evaluation wrapper
        remaining = self.budget
        eval_calls = 0

        def evaluate(x):
            nonlocal remaining, eval_calls
            if remaining <= 0:
                # Should never happen; guard for robustness.
                # Return a large value to avoid crashing.
                return float("inf")
            remaining -= 1
            eval_calls += 1
            return float(func(np.asarray(x, dtype=float)))

        # Scale for initialization and mutations
        span = ub - lb
        # Avoid zero-span dimensions by setting a minimal scale
        span_safe = np.where(span > 0, span, 1.0)

        # Population size: small enough to stay budget-safe and fast in higher dims.
        # We want at least 2 and at most 16 or budget (if budget is tiny).
        pop_size = int(min(16, max(2, self.dim + 1)))
        pop_size = int(min(pop_size, self.budget))

        # Initial step size based on the domain size
        step = 0.35 * span_safe
        # If domain is very small, ensure nonzero mutation
        step = np.where(step > 0, step, 1e-3)

        # Initialize population uniformly in bounds
        X = lb + rng.random((pop_size, self.dim)) * (ub - lb)
        # Evaluate initial population
        Y = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            Y[i] = evaluate(X[i])

        # Track best-so-far
        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])

        # Elitism: keep this many individuals each iteration
        elite = int(max(2, min(pop_size, 4)))

        # Main loop:
        # Each iteration generates up to batch_size offspring, while respecting budget.
        # We use evaluation-batches to keep accounting simple.
        # Offspring count chosen to balance progress and evaluations.
        # Do not exceed remaining budget.
        stagnation = 0
        prev_best_y = best_y

        while remaining > 0:
            # If we already used all evaluations, stop (handled by remaining condition).
            # Determine batch size.
            # 1/2 to full population evaluations per iteration (bounded by remaining and pop_size)
            batch_size = int(min(pop_size, max(1, pop_size // 2)))
            batch_size = int(min(batch_size, remaining))
            if batch_size <= 0:
                break

            # Sort by fitness (lower is better)
            order = np.argsort(Y)
            Xs = X[order]
            Ys = Y[order]

            # Compute parent selection probabilities biased towards best
            # Use ranks to avoid extreme weights.
            ranks = np.arange(pop_size, dtype=float)
            # Better individuals get higher weight; epsilon to avoid zero division.
            weights = (pop_size - ranks) + 1.0
            weights /= weights.sum()

            # Optionally update adaptive step size
            # If best improved, shrink; else grow slightly.
            if best_y < prev_best_y - 1e-12:
                stagnation = 0
                # shrink: refine
                step *= 0.82
                prev_best_y = best_y
            else:
                stagnation += 1
                # grow: escape stagnation
                step *= 1.08

            # Mild lower bound on step to keep exploration alive
            step_min = 1e-12 + 1e-3 * span_safe
            step = np.maximum(step, step_min)
            # Upper bound: not too large
            step_max = 0.9 * span_safe + 1e-12
            step = np.minimum(step, step_max)

            # Diversity injection/restart:
            # If stagnating for a few iterations, sample some new random points.
            diversify = 0
            if stagnation >= 6:
                diversify = min(batch_size, max(1, batch_size // 3))
                stagnation = 0  # reset after restart action
                step *= 1.2  # increase exploration radius

            # Generate offspring
            X_new = np.empty((batch_size, self.dim), dtype=float)
            # First fill with diversified random samples if needed
            if diversify > 0:
                for j in range(diversify):
                    X_new[j] = lb + rng.random(self.dim) * (ub - lb)

            # Remaining offspring via recombination + mutation
            start_j = diversify
            if start_j < batch_size:
                for j in range(start_j, batch_size):
                    # Choose a couple of parents from the current population
                    # Weighted by rank to bias selection.
                    # We use the sorted arrays to interpret ranks easily.
                    # Map selection indices back to Xs.
                    p1 = int(rng.choice(pop_size, p=weights))
                    p2 = int(rng.choice(pop_size, p=weights))
                    # Recombination weights favoring better parent
                    if p1 <= p2:
                        # p1 is likely better due to sorting
                        w = rng.random()
                        # Mix in a way that leans toward the better parent
                        a, b = Xs[p1], Xs[p2]
                        x0 = w * a + (1.0 - w) * b
                    else:
                        w = rng.random()
                        a, b = Xs[p2], Xs[p1]
                        x0 = w * a + (1.0 - w) * b

                    # Local mutation: isotropic-ish Gaussian scaled per dimension.
                    # Add a small attraction term toward best to exploit.
                    gauss = rng.normal(size=self.dim)
                    # Attraction toward best_x (small fraction)
                    attract = 0.05 * (rng.random() ** 2)
                    x = x0 + gauss * step + attract * (best_x - x0)

                    # Boundary handling: clip to bounds
                    x = np.clip(x, lb, ub)
                    X_new[j] = x

            # Evaluate offspring; budget-safe due to batch_size <= remaining
            Y_new = np.empty(batch_size, dtype=float)
            for j in range(batch_size):
                Y_new[j] = evaluate(X_new[j])

            # Update best-so-far
            idx_off_best = int(np.argmin(Y_new))
            if Y_new[idx_off_best] < best_y:
                best_y = float(Y_new[idx_off_best])
                best_x = X_new[idx_off_best].copy()

            # Elitist replacement: combine and keep best pop_size
            # (also ensures population doesn't lose good points)
            X_comb = np.vstack((X, X_new))
            Y_comb = np.concatenate((Y, Y_new))
            ord2 = np.argsort(Y_comb)
            X = X_comb[ord2[:pop_size]]
            Y = Y_comb[ord2[:pop_size]]

            # Quick termination if budget is used (remaining==0 is checked in while condition)

        return best_x, best_y
