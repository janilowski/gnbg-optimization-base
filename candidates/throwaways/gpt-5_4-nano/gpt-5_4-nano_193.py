# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimizer
# inspired by a randomized coordinate/step search with periodic global restarts.
# It maintains a small population of candidate points around the current best,
# evaluates them, and uses the best result to guide the next step. Steps shrink
# after unsuccessful iterations and are occasionally reset to encourage exploration.
#
# Search state: The algorithm tracks the incumbent best point x_best and its
# function value y_best, along with a current step scale (sigma) that controls
# random perturbation magnitude. It also keeps a remaining evaluation budget and
# stops exactly when the budget would be exceeded.
#
# Candidate generation: At each iteration, it generates several candidates by
# adding random Gaussian noise scaled by sigma. Additionally, it includes a
# coordinate-wise perturbation candidate that nudges along a random coordinate.
# Candidate points are clipped to the provided bounds.
#
# Selection and replacement: All candidates are evaluated (within the budget).
# If any candidate improves upon the incumbent, the algorithm updates x_best and
# y_best accordingly. The step scale is decreased after no improvement.
#
# Adaptation: The step scale (sigma) adapts using a simple success rule:
# it shrinks by a factor when iterations fail and grows slightly when improvement
# is found, staying within a reasonable range based on the domain size.
#
# Exploration mechanisms: sigma occasionally undergoes an exploration reset
# (random restart) after a configurable number of consecutive failures to escape
# local minima.
#
# Exploitation mechanisms: near the best point, candidates are generated with
# decreasing sigma, concentrating search around the incumbent.
#
# Boundary handling: Candidate vectors are projected back into the feasible box
# using clipping to [lb, ub].
#
# Budget strategy: Every call consumes at most the remaining budget. The algorithm
# computes the number of batches/iterations based on the budget and a fixed
# per-iteration evaluation count, and it never evaluates beyond the remaining
# evaluations.
#
# Closest known influences: Combines ideas from random local search, evolution-like
# sampling around an incumbent, and coordinate perturbations with step-size control.
#
# Novelty or unusual aspects: Uses both isotropic (Gaussian) and anisotropic
# (single-coordinate) proposals in each batch with simple step-size adaptation,
# designed to be robust and dimension-agnostic while remaining compact.
#
# Failure modes: On highly irregular objectives, improvements may be rare; the
# algorithm relies on budget-driven exploration resets and will otherwise converge
# slowly by shrinking sigma. If bounds are very tight or ill-specified, clipping
# may dominate and reduce effective movement.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim

        # Read bounds from func or func.bounds
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
        else:
            b = getattr(func, "bounds", None)
            if b is None or not (hasattr(b, "lb") and hasattr(b, "ub")):
                raise AttributeError(
                    "Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
                )
            lb = np.asarray(b.lb, dtype=float).reshape(-1)
            ub = np.asarray(b.ub, dtype=float).reshape(-1)

        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Ensure lb <= ub
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        lb = lb2
        ub = ub2

        rng = np.random

        # Helper for safe evaluation with exact budget accounting
        remaining = self.budget
        if remaining <= 0:
            # No evaluations allowed: return a feasible point and a dummy value
            x0 = lb + 0.5 * (ub - lb)
            return x0, float("inf")

        def project(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_one(x):
            nonlocal remaining
            if remaining <= 0:
                # Should never happen if budget logic is correct
                return float("inf")
            y = float(func(x))
            remaining -= 1
            return y

        # Choose an initial point uniformly within bounds (feasible)
        x_best = lb + rng.random(dim) * (ub - lb)
        y_best = eval_one(x_best)

        # Step size initial based on domain size
        domain = ub - lb
        # Avoid zero range -> sigma becomes small but nonzero to allow perturbations
        dom_norm = np.linalg.norm(domain) / np.sqrt(max(1, dim))
        sigma = 0.25 * dom_norm if dom_norm > 0 else 1e-3

        # Cap sigma based on domain to avoid wasting evaluations outside effective region
        sigma_min = 1e-12
        sigma_max = max(1e-3, dom_norm) if dom_norm > 0 else 1.0

        # Batch settings: per iteration evaluate k candidates including incumbent update candidates
        # Keep it small for robustness and budget control.
        k = 2 + min(6, dim)  # isotropic proposals + coordinate proposal + optional extra
        # We already evaluated the initial point, so we have remaining-1 evaluations left
        # We'll iterate in batches of size k while we can afford them.
        consecutive_failures = 0
        failure_restart = 6  # reset exploration after this many consecutive failures

        # Precompute a coordinate perturbation basis indices
        while remaining > 0:
            # Determine how many candidates we can still evaluate in this batch
            # Each batch uses k evaluations.
            batch_evals = min(k, remaining)

            candidates = []

            # Always include a random isotropic Gaussian perturbation around incumbent
            # Candidate 1..m are Gaussian, last one is coordinate perturbation (if budget allows).
            for _ in range(max(0, batch_evals - 1)):
                # Isotropic Gaussian perturbation
                z = rng.standard_normal(dim)
                x = x_best + sigma * z
                x = project(x)
                candidates.append(x)

            if batch_evals > 0:
                # Coordinate perturbation candidate (anisotropic, exploitation-friendly)
                idx = int(rng.randint(0, dim))
                step = sigma * (rng.standard_normal() if rng.rand() < 0.5 else 1.0)
                e = np.zeros(dim, dtype=float)
                e[idx] = step
                x = project(x_best + e)
                candidates.append(x)

            # Evaluate candidates (never exceeding budget)
            improved = False
            # Evaluate and track best among candidates
            for x in candidates[:batch_evals]:
                y = eval_one(x)
                if y < y_best:
                    x_best, y_best = x, y
                    improved = True

            # If batch_evals == 0, stop (should not occur because remaining > 0)
            if batch_evals <= 0:
                break

            if improved:
                consecutive_failures = 0
                # Mild step-size increase to better exploit promising region
                sigma = min(sigma_max, sigma * 1.15)
            else:
                consecutive_failures += 1
                # Shrink sigma after failure to refine around incumbent
                sigma = max(sigma_min, sigma * 0.7)

            # Exploration reset after repeated failures
            if consecutive_failures >= failure_restart and remaining > 0:
                # Jump to a new random point and re-seed sigma
                x_new = lb + rng.random(dim) * (ub - lb)
                y_new = eval_one(x_new)  # uses remaining budget (respects limit)
                if y_new < y_best:
                    x_best, y_best = x_new, y_new
                # Reset sigma based on domain scale (promote exploration)
                sigma = min(sigma_max, 0.5 * sigma_max)
                consecutive_failures = 0

            # If sigma becomes extremely small, still continue if budget remains
            # (the budget will naturally end the search).

        return np.asarray(x_best, dtype=float), float(y_best)
