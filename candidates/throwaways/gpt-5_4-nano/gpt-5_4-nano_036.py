# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization strategy
# (a noisy evolutionary / covariance-like variant) using only numpy. It
# maintains a population of candidate solutions within given bounds and
# iteratively improves them by mutating a mixture of the current best and
# other promising points.
# Search state: The algorithm keeps a population X of size pop_size, the
# corresponding objective values Y, the best found solution (best_x,
# best_y), and a step-size sigma controlling mutation strength. A simple
# success counter is used to adapt sigma.
# Candidate generation: Each generation samples new points around selected
# parents via Gaussian steps with occasional heavier-tailed perturbations.
# Parent selection is based on ranking (favoring lower objective values).
# Selection and replacement: Offspring are evaluated and then combined with
# the current population; the best individuals replace the worst ones
# (elitist generational replacement). The global best is tracked throughout.
# Adaptation: The mutation step-size sigma adapts based on recent success
# (how often offspring improve the current best). If improvements occur
# frequently, sigma decreases slower; otherwise it shrinks to refine.
# Exploration mechanisms: Early on, sigma is larger and heavier-tailed
# perturbations (scaled Gaussian) are used more often, encouraging broader
# exploration.
# Exploitation mechanisms: Later, sigma shrinks and parent selection becomes
# more strongly biased toward the current best individuals, focusing on
# local refinement.
# Boundary handling: Candidate vectors are clipped to the provided box
# constraints after mutation.
# Budget strategy: The algorithm strictly tracks remaining evaluations.
# It stops as soon as it would exceed the evaluation budget.
# Closest known influences: The design is inspired by simple evolution
# strategies / CMA-like patterns (rank-based selection, step-size
# adaptation), but implemented compactly without covariance estimation.
# Novelty or unusual aspects: Uses an adaptive mixture of Gaussian and
# occasional large jumps, plus a straightforward rank-based parent sampler.
# Failure modes: In very flat or deceptive landscapes, success-based sigma
# adaptation may shrink too quickly, leading to premature convergence. If
# bounds are extremely tight, clipping can reduce effective diversity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations possible; return zeros within bounds if available.
            lb, ub = self._get_bounds(func, dim)
            x0 = np.clip(np.zeros(dim, dtype=float), lb, ub)
            return x0, float(func(x0)) if budget > 0 else float("inf")

        lb, ub = self._get_bounds(func, dim)
        pop_size = int(np.clip(4 + 3 * np.log(max(2, dim)), 8, 32))
        pop_size = min(pop_size, budget) if budget > 0 else pop_size

        # Initialize population uniformly in bounds.
        X = lb + (ub - lb) * np.random.rand(pop_size, dim)

        evals = 0
        Y = np.empty(pop_size, dtype=float)

        # Evaluate initial population.
        for i in range(pop_size):
            if evals >= budget:
                break
            Y[i] = float(func(X[i]))
            evals += 1

        # If budget < pop_size, trim.
        if evals < pop_size:
            X = X[:evals].copy()
            Y = Y[:evals].copy()
            pop_size = evals

        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])

        # Initial step-size proportional to bounds span.
        span = np.maximum(ub - lb, 1e-12)
        sigma = 0.3 * float(np.mean(span))
        sigma = max(sigma, 1e-12)

        # Success-based adaptation.
        success = 0
        success_window = 0

        # Remaining evaluation budget controls number of generations.
        # Each generation evaluates up to pop_size offspring (or fewer).
        # We'll run until budget is exhausted.
        while evals < budget:
            remaining = budget - evals
            m = min(pop_size, remaining)  # number of offspring to evaluate this generation
            # Rank-based parent selection probabilities (lower Y -> higher prob).
            # Use a soft bias to keep diversity.
            order = np.argsort(Y)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(pop_size)

            # Convert ranks to probabilities: exp(-alpha * rank)
            # alpha increases as search progresses (more exploitation later).
            progress = evals / budget
            alpha = 1.0 + 4.0 * progress
            weights = np.exp(-alpha * ranks.astype(float))
            weights /= np.sum(weights)

            # Adapt exploration/exploitation mixture.
            # More exploration early, more exploitation later.
            heavy_prob = 0.25 * (1.0 - progress)  # heavier tail earlier
            best_bias = 0.25 + 0.5 * progress     # more direct best targeting later

            offspring = np.empty((m, dim), dtype=float)
            parent_indices = np.empty(m, dtype=int)

            # Selection and candidate generation.
            for j in range(m):
                r = np.random.rand()
                if r < best_bias:
                    parent = best_x
                    parent_indices[j] = best_idx
                else:
                    # Sample parent index from current population.
                    parent_idx = int(np.random.choice(pop_size, p=weights))
                    parent_indices[j] = parent_idx
                    parent = X[parent_idx]

                # Mutation: gaussian step plus occasional larger jump.
                if np.random.rand() < heavy_prob:
                    # heavier-tailed behavior: scale a Gaussian with larger factor
                    # Use 2.0 * standard deviation effect.
                    step = sigma * np.random.randn(dim) * (2.0 + 2.0 * np.random.rand())
                else:
                    step = sigma * np.random.randn(dim)

                # Add small "directional" term from best to encourage progress.
                # This is subtle: helps when parent is not best.
                mix = 0.1 * (1.0 - progress)
                if parent is not best_x:
                    step += mix * np.random.randn(dim) * (best_x - parent)

                x_new = parent + step
                x_new = np.clip(x_new, lb, ub)
                offspring[j] = x_new

            # Evaluate offspring (strictly within budget).
            off_y = np.empty(m, dtype=float)
            for j in range(m):
                if evals >= budget:
                    break
                off_y[j] = float(func(offspring[j]))
                evals += 1

            # Compute success relative to current best.
            improved = off_y[:m] < best_y
            if np.any(improved):
                success += int(np.sum(improved))
                success_window += m
                # Update global best.
                jbest = int(np.argmin(off_y[:m]))
                if float(off_y[jbest]) < best_y:
                    best_y = float(off_y[jbest])
                    best_x = offspring[jbest].copy()
            else:
                success_window += m

            # Adapt sigma:
            # - if improvements frequent => gently reduce sigma (keep ability to explore)
            # - if none => shrink sigma more to exploit local basin
            if success_window >= pop_size:
                rate = success / max(1, success_window)
                # Target success rate range depends on dimension (higher dim may need more exploration).
                target = 0.2 + 0.1 * (np.log(max(2, dim)) / np.log(10))
                if rate > target:
                    # Too many successes: slightly expand or keep.
                    sigma *= (1.05 - 0.2 * progress)
                else:
                    # Too few successes: shrink to refine.
                    sigma *= (0.7 ** (1.0 + progress))
                sigma = float(np.clip(sigma, 1e-12, float(np.mean(span)) * 2.0))
                success = 0
                success_window = 0

            # Elitist replacement: combine population with offspring and keep best pop_size.
            # (Budget allows evaluation of m, so replace only if population size > 0)
            if m > 0:
                X_comb = np.vstack([X, offspring[:m]])
                Y_comb = np.concatenate([Y, off_y[:m]])

                # Keep the best pop_size.
                idx = np.argsort(Y_comb)[:pop_size]
                X = X_comb[idx].copy()
                Y = Y_comb[idx].copy()
                best_idx = int(np.argmin(Y))

        return best_x, best_y

    @staticmethod
    def _get_bounds(func, dim):
        # Read bounds from one of the two expected conventions.
        # Prefer func.lower/func.upper; otherwise func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.array(func.lower, dtype=float).reshape(-1)
            ub = np.array(func.upper, dtype=float).reshape(-1)
        else:
            b = getattr(func, "bounds", None)
            if b is None or not (hasattr(b, "lb") and hasattr(b, "ub")):
                raise AttributeError(
                    "Objective must provide either func.lower/func.upper "
                    "or func.bounds.lb/func.bounds.ub."
                )
            lb = np.array(b.lb, dtype=float).reshape(-1)
            ub = np.array(b.ub, dtype=float).reshape(-1)

        if lb.size != dim or ub.size != dim:
            # Try broadcasting if given as scalars.
            if lb.size == 1:
                lb = np.full(dim, float(lb.item()), dtype=float)
            if ub.size == 1:
                ub = np.full(dim, float(ub.item()), dtype=float)

        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)
        if lb.size != dim or ub.size != dim:
            raise ValueError(f"Bounds size mismatch: expected dim={dim}, got {lb.size} and {ub.size}.")
        # Ensure lb <= ub
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        return lb2, ub2
