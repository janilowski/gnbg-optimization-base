# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer using a
# population-based global-to-local strategy with self-adaptive step sizes.
# Search state: Maintains a small set of candidate points (population) plus
# per-candidate step sizes (mutation scales). Tracks the best-so-far solution.
# Candidate generation: Uses Gaussian mutations around each population member,
# with step sizes scaled by a success/failure update rule. Also performs occasional
# differential-like recombination using two random population members to
# diversify.
# Selection and replacement: For each offspring, evaluates the objective and
# applies greedy replacement if it improves (minimization). The global best is
# updated whenever a new best is found.
# Adaptation: Step sizes are adapted using a simple 1/5th success-inspired rule:
# if a mutation succeeds, increase the step; if it fails, decrease it.
# Exploration mechanisms: Random initialization within bounds, stochastic
# recombination, and relatively large initial step sizes encourage global search.
# Exploitation mechanisms: Greedy replacement and shrinking step sizes around
# successful individuals drive local improvement.
# Boundary handling: All mutated candidates are clipped to the provided bounds.
# Budget strategy: Uses a strict evaluation counter; never exceeds the provided
# budget. Initial population and subsequent offspring evaluations stop early if
# the budget is exhausted. Each objective call is made via a wrapper that
# increments the counter.
# Closest known influences: Combines ideas from Evolution Strategies (ES) with
# success-based step adaptation and DE-like mutation/recombination, adapted to
# the provided budget and bounds.
# Novelty or unusual aspects: Uses a small, adaptive population with both
# Gaussian ES-style moves and lightweight recombination, while keeping the code
# minimal and robust across dimensions. Also attempts to infer bounds from a few
# common attribute layouts.
# Failure modes: If the objective is very noisy or adversarially structured,
# step-size adaptation may oscillate; boundary clipping can also concentrate
# points at the edges. With extremely small budgets, global exploration is
# limited and performance may degrade.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        n_evals = 0

        lb, ub = self._get_bounds(func)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        if lb.shape != (dim,) or ub.shape != (dim,):
            raise ValueError("Bounds must match dimension; expected shape (dim,).")

        # Ensure numerical sanity
        lb = np.minimum(lb, ub)
        ub = np.maximum(lb, ub)
        span = ub - lb
        span = np.where(span > 0, span, 1.0)  # avoid zero-span issues

        def eval_wrapped(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                # Should not happen if checks are correct, but keep safe.
                return float("inf")
            n_evals += 1
            y = func(np.asarray(x, dtype=float))
            # In case the function returns numpy scalar/array
            return float(np.asarray(y).reshape(()))

        # Candidate initialization
        # Choose a modest population size that scales with dimension.
        # Keep it small to respect strict budgets.
        pop = max(4, min(12, 2 + dim))
        pop = min(pop, self.budget) if self.budget > 0 else 0

        rng = np.random

        if pop == 0:
            x0 = lb.copy()
            return x0, float(eval_wrapped(x0))

        # Initial step size: a fraction of the box size.
        step0 = 0.3 * span
        # Sample initial population uniformly within bounds.
        X = lb + rng.rand(pop, dim) * (ub - lb)
        sigma = step0.copy()
        # Keep sigma per individual (small random variation)
        sigma = sigma * (0.5 + rng.rand(pop, dim))  # (pop, dim)
        sigma = np.maximum(sigma, 1e-12)

        # Evaluate initial population
        best_x = None
        best_y = float("inf")
        ys = np.empty(pop, dtype=float)
        for i in range(pop):
            ys[i] = eval_wrapped(X[i])
            if ys[i] < best_y:
                best_y = ys[i]
                best_x = X[i].copy()

            if n_evals >= self.budget:
                return best_x, best_y

        # Main loop: create offspring one batch at a time until budget exhausted.
        # Offspring count per iteration is min(pop, remaining budget).
        iter_guard = 0
        while n_evals < self.budget:
            iter_guard += 1
            if iter_guard > 10_000:
                break

            remaining = self.budget - n_evals
            k = min(pop, remaining)

            # Precompute some random indices for recombination
            idx_a = rng.randint(0, pop, size=k)
            idx_b = rng.randint(0, pop, size=k)

            # Offspring creation
            X_off = np.empty((k, dim), dtype=float)
            y_off = np.empty(k, dtype=float)
            succ = np.zeros(k, dtype=bool)

            for j in range(k):
                a = idx_a[j]
                b = idx_b[j]
                base = X[a]

                # ES-like Gaussian move using individual sigma
                # sigma[j] corresponds to base index a.
                s = sigma[a]
                # Exploration/exploitation blend:
                # With some probability use DE-like direction to diversify.
                if rng.rand() < 0.5:
                    # Lightweight DE: base + F*(X[a]-X[b]) scaled
                    # plus Gaussian noise proportional to sigma.
                    F = 0.5 + 0.7 * rng.rand()
                    diff = X[a] - X[b]
                    mut = base + F * diff + s * rng.randn(dim)
                else:
                    mut = base + s * rng.randn(dim)

                # Boundary handling: clip to feasible box
                mut = np.clip(mut, lb, ub)
                X_off[j] = mut

                # Evaluate
                y = eval_wrapped(X_off[j])
                y_off[j] = y

                if y < best_y:
                    best_y = y
                    best_x = X_off[j].copy()

                # Greedy replacement & success signal
                if y < ys[a]:
                    succ[j] = True
                    X[a] = X_off[j]
                    ys[a] = y

                # Step-size adaptation for individual a
                # (success -> increase, failure -> decrease)
                # Keep adaptation stable and bounded.
                if succ[j]:
                    sigma[a] *= 1.05 + 0.15 * rng.rand(dim)
                else:
                    sigma[a] *= 0.82  # stronger decrease on failure

                # Clamp sigma to reasonable ranges based on span
                # (avoid collapsing to zero or exploding)
                min_s = 1e-12
                max_s = 0.8 * span
                sigma[a] = np.clip(sigma[a], min_s, max_s)

                if n_evals >= self.budget:
                    break

            if n_evals >= self.budget:
                break

            # Optional additional exploitation: small local sampling around best
            # to spend remaining evaluations more effectively, if room exists.
            remaining = self.budget - n_evals
            if remaining <= 0:
                break
            if rng.rand() < 0.35 and remaining > 0:
                # One local probe centered at best
                s_best = np.clip(0.2 * span, 1e-12, None)
                x_probe = best_x + s_best * rng.randn(dim)
                x_probe = np.clip(x_probe, lb, ub)
                y_probe = eval_wrapped(x_probe)
                if y_probe < best_y:
                    best_y = y_probe
                    best_x = x_probe.copy()

                # If it improves, also try to insert into population
                if y_probe < best_y:
                    pass
                # Update population greedily: find worst and replace if better
                if y_probe < ys.max():
                    worst = int(np.argmax(ys))
                    X[worst] = x_probe
                    ys[worst] = y_probe

        # If for some reason best_x is None (shouldn't happen)
        if best_x is None:
            best_x = lb.copy()
            best_y = float(eval_wrapped(best_x))

        return best_x, best_y

    @staticmethod
    def _get_bounds(func):
        # Priority:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return func.lower, func.upper

        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return b.lb, b.ub

        # Some harnesses might use different attribute names; keep minimal but helpful
        if hasattr(func, "lb") and hasattr(func, "ub"):
            return func.lb, func.ub

        raise AttributeError(
            "Cannot infer bounds. Provide func.lower/func.upper or func.bounds.lb/func.bounds.ub."
        )
