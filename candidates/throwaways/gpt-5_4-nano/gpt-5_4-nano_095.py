# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimization
# algorithm (a simple Evolution Strategy / coordinate-random search hybrid).
# It maintains a population of candidate points, repeatedly samples new points around
# a current center using an adaptive step-size, and replaces the center with the
# best individual found so far.
# Search state: Keeps a running evaluation counter, current center x_best,
# best objective value y_best, and a mutable step size sigma.
# Candidate generation: At each iteration, generates a small population by adding
# isotropic Gaussian perturbations to the current center: x = center + sigma * N(0, I).
# Selection and replacement: Evaluates all candidates (without exceeding the budget),
# selects the lowest objective value, and updates the center to that best candidate.
# Adaptation: Step size sigma adapts based on relative improvement; it decays on
# stagnation and modestly increases when improvement is observed to help escape local minima.
# Exploration mechanisms: Random Gaussian sampling around the center plus periodic
# "recenter" from the best found population maintains global exploration early on.
# Exploitation mechanisms: As sigma shrinks with lack of improvement, the search focuses
# near the best point, effectively performing local exploitation.
# Boundary handling: Candidates are clipped to the provided bounds (lower/upper).
# Budget strategy: Each evaluation call consumes exactly one budget unit; the algorithm
# stops once the budget is exhausted (or immediately after evaluating the initial point).
# Closest known influences: Inspired by basic (μ+λ)-style ES and trust-region-like
# step-size adaptation, implemented in a minimal, robust way.
# Novelty or unusual aspects: Uses budget-aware population sizing and a simple
# improvement-driven sigma schedule to remain effective across dimensions.
# Failure modes: If the objective is very noisy or highly irregular, sigma adaptation
# may oscillate or converge prematurely; strict clipping can also cause boundary
# crowding in constrained problems.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # Read bounds from func lower/upper or func.bounds.lb/ub
        lb, ub = _get_bounds(func, self.dim)
        lb = lb.astype(float)
        ub = ub.astype(float)

        # Handle degenerate bounds safely
        span = ub - lb
        span = np.where(np.isfinite(span), span, 0.0)
        span = np.maximum(span, 0.0)

        # Initialization:
        # Start at the middle of the box (deterministic). If span is zero everywhere,
        # this will be constant and the search ends quickly.
        center = lb + 0.5 * span

        # Evaluate initial point (must count toward budget)
        evals_used = 0
        best_x = np.array(center, copy=True)

        # Budget guard
        if self.budget <= 0:
            return best_x, float("inf")

        best_y = float(func(best_x))
        evals_used += 1

        # Initial step size:
        # Use a fraction of the average span; fallback to 1.0 if span is tiny.
        avg_span = float(np.mean(span)) if self.dim > 0 else 0.0
        sigma = 0.3 * avg_span if avg_span > 0 else 0.3

        # Population size heuristic:
        # Small λ keeps calls low and fits many harness budgets.
        # Ensure at least 1 to make progress.
        lam_base = max(4, min(16, 2 * (self.dim if self.dim > 0 else 1)))
        # We'll adjust lam down when nearing budget.
        improvement_threshold = 1e-12

        # Main loop: budget-aware
        while evals_used < self.budget:
            remaining = self.budget - evals_used
            # Choose λ such that we don't exceed remaining budget.
            lam = min(lam_base, remaining)
            if lam <= 0:
                break

            # Generate λ candidates around center using isotropic Gaussian perturbations.
            # Shape: (lam, dim)
            noise = np.random.randn(lam, self.dim)
            cand = center[None, :] + sigma * noise

            # Clip to bounds
            # (Works whether lb/ub are scalars or vectors; we ensure vectors above.)
            cand = np.minimum(np.maximum(cand, lb[None, :]), ub[None, :])

            # Evaluate candidates and pick best
            # Evaluate one-by-one to keep strict budget counting and generic func signature.
            cur_best_y = best_y
            cur_best_x = best_x

            for i in range(lam):
                y = float(func(cand[i]))
                evals_used += 1
                if y < cur_best_y:
                    cur_best_y = y
                    cur_best_x = cand[i].copy()
                if evals_used >= self.budget:
                    break

            # Update center if improved
            prev_y = best_y
            best_y = cur_best_y
            best_x = cur_best_x
            improved = prev_y - best_y > improvement_threshold

            if improved:
                center = best_x
                # Increase sigma slightly to encourage continued exploration when improving
                sigma = sigma * 1.05
            else:
                # Decay sigma on stagnation to exploit locally
                sigma = sigma * 0.85

            # Keep sigma within reasonable limits
            # Lower bound prevents numerical collapse; upper bound prevents huge jumps.
            # Use span to scale upper bound, but keep safe defaults.
            span_scale = float(np.mean(span)) if self.dim > 0 else 0.0
            if span_scale <= 0.0:
                sigma_max = 1.0
            else:
                sigma_max = 0.5 * span_scale
            sigma = float(np.clip(sigma, 1e-12, sigma_max if sigma_max > 0 else 1.0))

        return best_x, best_y


def _get_bounds(func, dim: int):
    # Priority:
    # 1) func.lower/func.upper
    # 2) func.bounds.lb / func.bounds.ub
    if hasattr(func, "lower") and hasattr(func, "upper"):
        lb = getattr(func, "lower")
        ub = getattr(func, "upper")
    elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
        lb = func.bounds.lb
        ub = func.bounds.ub
    else:
        raise AttributeError(
            "Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
        )

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    # Broadcast scalars to dim or validate shapes
    if lb.ndim == 0:
        lb = np.full(dim, float(lb))
    elif lb.shape[0] != dim:
        lb = np.resize(lb, dim)

    if ub.ndim == 0:
        ub = np.full(dim, float(ub))
    elif ub.shape[0] != dim:
        ub = np.resize(ub, dim)

    # Ensure lb <= ub; if swapped, correct.
    lb2 = np.minimum(lb, ub)
    ub2 = np.maximum(lb, ub)
    return lb2, ub2
