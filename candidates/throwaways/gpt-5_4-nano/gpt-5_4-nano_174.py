# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimizer using a
# (1+λ)-ES strategy with self-adaptive step sizes (log-normal mutation). It maintains a
# single incumbent solution and generates multiple offspring each generation, selecting the
# best candidate to replace the incumbent (elitist selection for minimization).
# Search state: The algorithm stores the current best point x_best, its objective value
# f_best, and a per-dimension mutation step size sigma. It also tracks how many function
# evaluations have been used to ensure the provided budget is never exceeded.
# Candidate generation: For each generation, it samples λ offspring by adding Gaussian noise
# scaled by sigma to the incumbent. Offspring sigma is adapted by multiplying by exp(τ*N(0,1)),
# where τ is dimension-dependent (standard self-adaptation from evolution strategies).
# Selection and replacement: After evaluating offspring, the best (minimum objective) among
# incumbent and offspring is chosen as the next incumbent (μ=1, λ+1 elitist scheme).
# Adaptation: Step sizes sigma are adapted each generation using the best offspring's
# adapted sigma (or conservatively kept) to encourage progress while maintaining stability.
# Exploration mechanisms: Early and generally throughout the run, the Gaussian perturbations
# provide global/local exploration, with sigma controlling the exploration radius.
# Exploitation mechanisms: As sigma shrinks via adaptation and selection focuses on improving
# incumbents, the search gradually exploits around promising regions.
# Boundary handling: Proposed offspring are clipped to the problem bounds on every coordinate
# to guarantee feasibility. (No additional penalty is used since the objective is treated as
# a black box.)
# Budget strategy: The total number of objective evaluations is tracked and offspring
# generation stops when the remaining budget is insufficient for additional evaluations.
# Closest known influences: The design is inspired by (1+λ)-Evolution Strategies with
# self-adaptation of step sizes (log-normal mutation).
# Novelty or unusual aspects: The implementation dynamically chooses λ based on dimension
# and remaining budget to make it robust across dimensions while staying within budget.
# Failure modes: If the objective is extremely noisy or flat, sigma adaptation may drift;
# additionally, hard clipping can reduce effective diversity near boundaries. The algorithm
# still remains budget-safe and returns the best seen point.
# ALGORITHM_ANALYSIS_NOTE_END

from typing import Callable, Tuple, Optional
import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func: Callable) -> Tuple[np.ndarray, float]:
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return a deterministic point (all zeros).
            x0 = np.zeros(dim, dtype=float)
            return x0, float("inf")

        # --- Read bounds from func ---
        lb, ub = self._get_bounds(func, dim)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        if lb.shape != (dim,) or ub.shape != (dim,):
            raise ValueError("Bounds must match dimension.")
        # Ensure proper ordering (robustness).
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)

        rng = np.random

        # --- Initialization ---
        # Pick a feasible starting point uniformly at random within bounds.
        x_best = lo + (hi - lo) * rng.rand(dim)
        f_best, evals = self._eval(func, x_best, budget, 0)

        # Step size initialization: fraction of the box size (avoid zeros).
        box = hi - lo
        # If box has zeros, add a small epsilon for those dimensions.
        eps = 1e-12
        sigma = 0.2 * np.where(box > 0, box, 1.0) + eps

        # Choose λ based on dimension; cap to keep evaluations under budget.
        # Common ES heuristic: λ ≈ 4 + 3*log(d)
        lam = int(max(4, 3 * np.log(max(2, dim)) + 4))
        lam = max(2, lam)

        # ES parameters for self-adaptation (dimension-dependent).
        # τ' = 1/sqrt(2d), τ = 1/sqrt(2*sqrt(d)) commonly.
        tau_prime = 1.0 / np.sqrt(2.0 * dim)
        tau = 1.0 / np.sqrt(2.0 * np.sqrt(dim))

        # --- Main loop: generate offspring until budget is exhausted ---
        # We spend evaluations on offspring; keep an incumbent and update elitist.
        while evals < budget:
            # Determine how many offspring we can still evaluate.
            remaining = budget - evals
            curr_lam = min(lam, remaining)
            if curr_lam <= 0:
                break

            # Sample global normal for each offspring.
            # Offspring sampling is vectorized for speed and simplicity.
            # Also adapt sigma per offspring.
            # Shape: (curr_lam, dim)
            z = rng.randn(curr_lam, dim)
            # Self-adaptation: global and coordinate-wise components
            g = rng.randn(curr_lam)  # global
            # coordinate-wise
            n = rng.randn(curr_lam, dim)

            # Compute offspring sigmas (log-normal update)
            # sigma_off = sigma * exp(tau' * g + tau * n)
            # Broadcasting: (curr_lam,) and (curr_lam,dim)
            sigma_off = sigma * np.exp(tau_prime * g[:, None] + tau * n)

            # Generate offspring candidates and clip to bounds.
            x_off = x_best[None, :] + sigma_off * z
            x_off = np.clip(x_off, lo[None, :], hi[None, :])

            # Evaluate offspring one by one to remain budget-safe.
            best_off_y = None
            best_off_x = None
            best_off_sigma = None

            for i in range(curr_lam):
                y, evals = self._eval(func, x_off[i], budget, evals)
                if best_off_y is None or y < best_off_y:
                    best_off_y = y
                    best_off_x = x_off[i].copy()
                    best_off_sigma = sigma_off[i].copy()
                if evals >= budget:
                    break

            # Decide next incumbent using elitist replacement (include incumbent already known).
            if best_off_y is not None and best_off_y < f_best:
                x_best = best_off_x
                f_best = best_off_y
                # Adapt sigma towards the winner.
                # Blend to avoid abrupt changes; keep stability.
                sigma = 0.7 * sigma + 0.3 * best_off_sigma
            else:
                # If no improvement, slightly decrease sigma to encourage local search,
                # but keep some exploration.
                sigma = 0.9 * sigma
                # Also enforce a minimal sigma based on box and eps to avoid stalling.
                sigma_min = 1e-12 + 1e-3 * np.where(box > 0, box, 1.0)
                sigma = np.maximum(sigma, sigma_min)

        return x_best, float(f_best)

    def _eval(self, func: Callable, x: np.ndarray, budget: int, evals_used: int) -> Tuple[float, int]:
        # Assumes evals_used < budget when called.
        if evals_used >= budget:
            return float("inf"), evals_used
        y = func(x)
        evals_used += 1
        return float(y), evals_used

    def _get_bounds(self, func: Callable, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        # Preferred: func.lower / func.upper or func.bounds.lb / func.bounds.ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = func.bounds.lb
            ub = func.bounds.ub
        else:
            raise AttributeError(
                "Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        # Allow scalar bounds by broadcasting to dim.
        if lb.ndim == 0:
            lb = np.full(dim, float(lb))
        if ub.ndim == 0:
            ub = np.full(dim, float(ub))

        return lb, ub
