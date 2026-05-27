# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimizer (GNBG-style)
# using a budgeted variant of iterative random directions with a greedy line search
# around the current best. It maintains a small population of candidate points and uses
# sampled directional steps to explore, then exploits by refining along the best direction.
#
# Search state: The algorithm tracks the current best point x_best and its objective
# value y_best. It also keeps a set of "active" direction vectors sampled from a normal
# distribution and normalized for stable step scaling. A step size (sigma) controls
# the magnitude of exploration and is adapted based on success.
#
# Candidate generation: Each iteration proposes candidates by taking x_best plus a
# scaled directional perturbation (both + and - along each direction). Additionally,
# it performs a lightweight 1D refinement along the best-performing direction among
# the sampled candidates by testing a couple of fractions of that step.
#
# Selection and replacement: All evaluated candidates are compared to the current
# best; when a candidate improves y_best, it replaces x_best and may update the
# "best direction" used for exploitation in later steps.
#
# Adaptation: If improvement happens, sigma is reduced moderately (to zoom in) and
# occasionally refreshed with more random directions. If no improvement happens,
# sigma is increased slightly (to escape local basins) and direction diversity is
# maintained.
#
# Exploration mechanisms: Random unit directions (normal-based) with bidirectional
# steps explore the space globally within the bounds. Direction sets are re-sampled
# periodically to preserve diversity.
#
# Exploitation mechanisms: After finding a good direction, the algorithm tests
# additional points along that direction at smaller step fractions to locally refine.
#
# Boundary handling: All candidates are clipped to the provided bounds. This keeps
# evaluations inside the feasible region without assuming convexity.
#
# Budget strategy: The algorithm carefully tracks remaining evaluations. It never
# calls the objective more times than the budget specified to Algorithm.__init__.
# The number of candidates per iteration adapts to dimension and remaining budget.
#
# Closest known influences: The design is reminiscent of coordinate-free evolution
# strategies / pattern search hybrids: directional sampling, greedy replacement,
# and step-size adaptation. It is intentionally simple and robust across dimensions.
#
# Novelty or unusual aspects: A small two-stage candidate approach (direction sampling
# then fraction-based refinement along the best direction) offers a balance between
# exploration and exploitation while remaining compact.
#
# Failure modes: If the objective has extremely narrow minima or is highly noisy,
# sigma adaptation may not converge well. Clipping can bias searches near boundaries.
# With very small budgets, the algorithm may only do a few exploratory evaluations.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._get_bounds(func)
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        d = self.dim
        n_evals_max = max(0, self.budget)

        # If budget is 0, return something deterministic without evaluating.
        if n_evals_max == 0:
            x0 = (lb + ub) * 0.5
            return x0, float("inf")

        # Initial point: center of bounds (stable across dimensions).
        x_best = np.clip((lb + ub) * 0.5, lb, ub)
        y_best = self._eval(func, x_best, n_evals_max, eval_counter=[0])
        eval_counter = [1]  # keep local counter in sync with _eval

        # Remaining evaluations guard
        def remaining():
            return n_evals_max - eval_counter[0]

        # Step size: start proportional to box size / sqrt(d)
        box = ub - lb
        # Prevent degenerate scales; if box is 0, exploration is irrelevant.
        scale = np.where(box > 0, box, 1.0)
        sigma = float(0.3 * np.median(scale) / max(1.0, np.sqrt(d)))

        # Direction sampling: maintain a rolling set of directions
        rng = np.random

        best_dir = None

        # Iteration loop: we decide how many candidates we can afford each time.
        # Each candidate is one objective evaluation.
        # We'll do: K directions * {+,-} then optionally 2 refinements along best direction.
        while remaining() > 0:
            # Choose K so that 2*K (+ refinements) fits in budget.
            # Keep K small for compactness; increase slightly with dimension.
            # Base K around max(2, min(8, 2*sqrt(d))) but budget-aware.
            K = int(max(2, min(8, 2.0 * np.sqrt(d))))
            # Feasible evaluations: 2*K for +/-; plus up to 2 refinements.
            max_K = remaining() // 2
            if max_K <= 0:
                # Only one evaluation left; just probe along a random direction.
                v = rng.normal(size=d)
                v_norm = np.linalg.norm(v)
                if v_norm == 0:
                    v = np.zeros(d)
                    v[0] = 1.0
                else:
                    v = v / v_norm
                step = sigma if sigma != 0 else 0.0
                x_try = np.clip(x_best + step * v, lb, ub)
                y_try = func(x_try)
                eval_counter[0] += 1
                if y_try < y_best:
                    x_best, y_best = x_try, float(y_try)
                    best_dir = v.copy()
                break

            K = min(K, max_K)

            # Generate random unit directions
            dirs = rng.normal(size=(K, d))
            norms = np.linalg.norm(dirs, axis=1)
            # Handle rare zero vectors
            zero_mask = norms == 0
            if np.any(zero_mask):
                dirs[zero_mask, 0] = 1.0
                norms[zero_mask] = 1.0
            dirs = dirs / norms[:, None]

            # Exploration candidates: +/- sigma along each direction
            improved = False
            best_local = y_best
            best_candidate = x_best
            best_local_dir = best_dir

            for i in range(K):
                # + direction
                if remaining() <= 0:
                    break
                v = dirs[i]
                step = sigma
                x_plus = np.clip(x_best + step * v, lb, ub)
                y_plus = func(x_plus)
                eval_counter[0] += 1
                if y_plus < best_local:
                    best_local = float(y_plus)
                    best_candidate = x_plus
                    best_local_dir = v.copy()

                # - direction
                if remaining() <= 0:
                    break
                x_minus = np.clip(x_best - step * v, lb, ub)
                y_minus = func(x_minus)
                eval_counter[0] += 1
                if y_minus < best_local:
                    best_local = float(y_minus)
                    best_candidate = x_minus
                    best_local_dir = (-v).copy()

            # Exploitation: refine along the best direction found among candidates
            # using smaller fractions. Two refinement points at most.
            if remaining() > 0 and best_local_dir is not None and not np.allclose(best_candidate, x_best):
                # If best_local improved, refine more aggressively; else smaller zoom.
                # Use fractions relative to sigma.
                refine_fracs = (0.5, 0.25) if best_local < y_best else (0.35, 0.2)
                v = best_local_dir
                for frac in refine_fracs:
                    if remaining() <= 0:
                        break
                    x_ref = np.clip(x_best + frac * sigma * v, lb, ub)
                    y_ref = func(x_ref)
                    eval_counter[0] += 1
                    if y_ref < best_local:
                        best_local = float(y_ref)
                        best_candidate = x_ref
                        # Keep direction as is
                # (No further changes if refinement didn't beat best_local)

            # Adapt and select
            if best_local < y_best:
                x_best, y_best = best_candidate, best_local
                best_dir = best_local_dir
                improved = True

            # Step size adaptation
            if improved:
                # Zoom in on success.
                sigma *= 0.82
                # Occasionally refresh best_dir by reusing direction but doesn't matter;
                # keep it simple.
            else:
                # No improvement: broaden search.
                sigma *= 1.05
                # If sigma becomes too small relative to scale, bump it.
                sigma_min = 1e-12
                if abs(sigma) < sigma_min:
                    sigma = sigma_min

            # If sigma is effectively zero, we can't improve unless objective at x_best already minimal.
            if sigma == 0 or np.all(box == 0):
                break

        return x_best, float(y_best)

    @staticmethod
    def _get_bounds(func):
        # Accept multiple possible locations for bounds as specified by prompt.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError(
                "Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )
        return lb, ub

    @staticmethod
    def _eval(func, x, budget, eval_counter):
        # Helper to evaluate within budget, used only for the initial evaluation.
        if eval_counter[0] >= budget:
            return float("inf")
        y = func(x)
        eval_counter[0] += 1
        return float(y)
