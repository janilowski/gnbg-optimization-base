# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimization algorithm
# using a population-based evolutionary strategy with a global-to-local transition.
# It maintains a small archive-free population and iteratively proposes candidate
# solutions via Gaussian sampling around an evolving center.
# Search state: Keeps a set of candidate points, their fitness values, the best
# found so far, and an adaptive step-size (sigma). The algorithm also tracks
# evaluations used to strictly respect the provided budget.
# Candidate generation: Each iteration samples offspring by adding isotropic
# Gaussian noise to the current center, with a per-iteration sigma that adapts
# based on whether new points improve the best. Additionally, it performs occasional
# "differential" moves using two random population members to improve exploration
# without gradients.
# Selection and replacement: Selects the best offspring and updates the center toward
# promising solutions (using a weighted recombination of elites). It uses elitism
# by never discarding the current best.
# Adaptation: Uses a simple success-based rule for sigma: if an iteration yields
# an improvement in the best fitness, sigma is reduced (more local search); otherwise
# sigma is increased slightly (more exploration). Also bounds sigma to a reasonable range
# relative to the domain size.
# Exploration mechanisms: Gaussian sampling with a relatively large initial sigma,
# occasional differential mutation, and occasional random re-centering if progress stalls.
# Exploitation mechanisms: Elitist recombination and shrinking sigma after improvements
# to focus around the best region.
# Boundary handling: After proposing points, clips them to the provided bounds (box constraints).
# Budget strategy: All objective evaluations are counted and the algorithm never exceeds
# the evaluation budget. It estimates how many full iterations fit and performs a final
# partial batch if needed.
# Closest known influences: Inspired by (1+lambda)/(mu+lambda)-style ES and by CMA-like
# ideas (step-size adaptation and recombination), but intentionally kept lightweight.
# Novelty or unusual aspects: Uses a global center updated by elite-weighted averaging
# and includes a small differential jump using population members to escape stagnation,
# while remaining compact and budget-safe.
# Failure modes: For extremely flat objectives or very tight budgets, improvement may be
# limited; if the best point is at/near bounds, the clipping may reduce effective
# exploration. The stochastic nature can lead to variance across runs.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return zeros within bounds if possible.
            lb, ub = self._get_bounds(func, dim)
            x = np.zeros(dim, dtype=float)
            x = self._clip(x, lb, ub)
            return x, float("inf")

        lb, ub = self._get_bounds(func, dim)
        span = np.maximum(ub - lb, 1e-12)
        center = (lb + ub) / 2.0

        # Population size (must be >= 2 for differential mutation).
        # Chosen to be small enough to fit most budgets but still informative.
        pop_size = int(np.clip(4 + dim // 3, 4, 24))
        pop_size = min(pop_size, budget) if budget > 0 else pop_size

        # Initialize sigma relative to domain.
        sigma_min = 1e-12 * np.max(span)
        sigma_max = 0.5 * np.max(span)
        sigma = 0.25 * np.max(span)
        sigma = float(np.clip(sigma, sigma_min, sigma_max))

        evals = 0
        best_x = None
        best_y = float("inf")

        # Helper for evaluation with budget counting
        def eval_one(x):
            nonlocal evals, best_x, best_y
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.array(x, dtype=float, copy=True)
            return y

        # Initial sample: create a starting population around the center.
        # Use a smaller step initially so we likely get a decent starting best.
        init_count = min(pop_size, budget)
        population = np.empty((init_count, dim), dtype=float)
        fitness = np.empty(init_count, dtype=float)

        for i in range(init_count):
            # Mix uniform and Gaussian to handle unknown landscapes
            if np.random.rand() < 0.35:
                x = lb + span * np.random.rand(dim)
            else:
                x = center + sigma * np.random.randn(dim)
            x = self._clip(x, lb, ub)
            population[i] = x
            fitness[i] = eval_one(x)

        # Set center to a weighted average of current elites.
        elite_k = max(2, min(init_count, 4))
        elite_idx = np.argsort(fitness)[:elite_k]
        w = np.linspace(1.0, 2.0, elite_k)
        w /= w.sum()
        center = np.sum(population[elite_idx] * w[:, None], axis=0)

        # Stagnation tracking to occasionally re-center.
        no_improve_iters = 0
        best_hist = best_y

        # Compute remaining evaluations and iterate in batches of pop_size.
        remaining = budget - evals
        if remaining <= 0:
            return np.array(best_x, dtype=float), float(best_y)

        # Maximum iterations by budget.
        # We'll generate offspring batches of size batch_size.
        batch_size = min(pop_size, remaining)

        # Main optimization loop
        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break
            batch_size = min(pop_size, remaining)

            # Decide exploration vs exploitation
            # If sigma is small and we have stagnation, rely more on differential jumps.
            explore_prob = 0.25 + 0.25 * (sigma / (sigma_max + 1e-30))
            explore_prob = float(np.clip(explore_prob, 0.15, 0.6))

            # Differential vectors from current population
            # (if we have less than 2 points, fallback to random noise)
            if population.shape[0] >= 2:
                a_idx = np.random.randint(0, population.shape[0], size=batch_size)
                b_idx = np.random.randint(0, population.shape[0], size=batch_size)
            else:
                a_idx = b_idx = np.zeros(batch_size, dtype=int)

            offspring = np.empty((batch_size, dim), dtype=float)
            off_fit = np.empty(batch_size, dtype=float)

            # Create offspring
            for i in range(batch_size):
                x = np.array(center, copy=True)

                if np.random.rand() < explore_prob:
                    # Gaussian exploration/exploitation around center
                    x = x + sigma * np.random.randn(dim)
                else:
                    # Differential jump: move along difference of two points
                    # plus small noise to keep it stochastic.
                    if population.shape[0] >= 2:
                        diff = population[a_idx[i]] - population[b_idx[i]]
                        scale = 0.5 + 0.5 * np.random.rand()
                        x = x + scale * sigma * diff / (np.linalg.norm(diff) + 1e-12)
                    else:
                        x = x + sigma * np.random.randn(dim)

                    # Occasionally incorporate a uniform move to escape local traps
                    if np.random.rand() < 0.08:
                        x = 0.6 * x + 0.4 * (lb + span * np.random.rand(dim))

                # Boundary handling: clip to the box.
                x = self._clip(x, lb, ub)
                offspring[i] = x
                off_fit[i] = eval_one(x)

            # Selection: update population with best individuals from combined set
            # to keep memory limited but adaptive.
            combined_x = np.vstack([population, offspring]) if population.size else offspring
            combined_f = np.concatenate([fitness, off_fit]) if fitness.size else off_fit

            # Elitism
            order = np.argsort(combined_f)
            keep = min(len(order), pop_size)
            elite = order[:keep]
            population = combined_x[elite]
            fitness = combined_f[elite]

            # Center recombination from top elites of the current kept set
            elite_k = max(2, min(4, keep))
            elite_idx = np.argsort(fitness)[:elite_k]
            w = np.linspace(1.0, 2.0, elite_k)
            w /= w.sum()
            center = np.sum(population[elite_idx] * w[:, None], axis=0)

            # Adapt sigma based on improvement
            if best_y < best_hist - 1e-18:
                # Progress: shrink sigma to exploit more locally
                sigma *= 0.82
                no_improve_iters = 0
                best_hist = best_y
            else:
                # No progress: expand sigma slightly
                sigma *= 1.07
                no_improve_iters += 1

            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # Stagnation escape: if no improvement for several iterations, re-center.
            # This is lightweight and budget-safe.
            # Estimate number of iterations from eval usage: if pop_size is small,
            # we trigger after a few batches.
            if no_improve_iters >= 4:
                # Re-center by mixing best with a random point from the domain
                r = lb + span * np.random.rand(dim)
                center = 0.7 * center + 0.3 * r
                # Reset sigma upward a bit to restart exploration
                sigma = float(np.clip(sigma * 1.35, sigma_min, sigma_max))
                no_improve_iters = 0

        return np.array(best_x, dtype=float), float(best_y)

    @staticmethod
    def _clip(x, lb, ub):
        return np.minimum(np.maximum(x, lb), ub)

    @staticmethod
    def _get_bounds(func, dim):
        # Bounds can be provided either as func.lower/func.upper or via func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # If bounds are not provided, fall back to a default symmetric range.
            # (The benchmark spec implies bounds exist, but this prevents crashes.)
            lb = -5.0 * np.ones(dim, dtype=float)
            ub = 5.0 * np.ones(dim, dtype=float)

        # Ensure correct shape
        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != dim or ub.size != dim:
            # Broadcast if possible
            if lb.size == 1:
                lb = np.full(dim, float(lb[0]), dtype=float)
            if ub.size == 1:
                ub = np.full(dim, float(ub[0]), dtype=float)
            if lb.size != dim or ub.size != dim:
                raise ValueError("Bounds dimensionality does not match dim.")

        return lb, ub
