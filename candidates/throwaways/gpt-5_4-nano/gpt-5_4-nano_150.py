# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization strategy
# based on a multi-start, fitness-guided coordinate search with a shrinking
# step-size schedule and periodic random reinitialization. It maintains the
# best solution found so far and iteratively samples candidate points by
# probing along coordinate directions and, occasionally, by global random
# sampling.
#
# Search state: The algorithm tracks the best_x/best_y, a per-run step size
# (delta), and a remaining evaluation budget. It also maintains the current
# search center (x_center) which is aligned to the best solution found in the
# current phase.
#
# Candidate generation: In each iteration, it generates candidates by
# evaluating points at +/- delta along a randomly permuted set of coordinates.
# Additionally, it can generate a few random candidates within the bounds to
# escape stagnation.
#
# Selection and replacement: Among evaluated candidates, it picks the best
# (lowest objective) and replaces the center with that point if it improves
# the incumbent. The global best is updated whenever an even better value is
# discovered.
#
# Adaptation: The step size delta shrinks whenever no improvement is observed,
# using a multiplicative factor. The algorithm also resets or increases
# exploration periodically via random restarts.
#
# Exploration mechanisms: Random restarts (and small sets of uniformly random
# candidates)
# are used when improvement stalls, helping it move away from local minima.
#
# Exploitation mechanisms: Coordinate probing around the current best/center
# performs local exploitation, gradually tightening the search as delta shrinks.
#
# Boundary handling: All candidate points are clipped to the problem bounds
# after adding perturbations. If a candidate collapses to the current center
# due to clipping, it still counts as an evaluation but may not improve.
#
# Budget strategy: The algorithm never exceeds the evaluation budget. It uses
# a remaining-budget guard around every function call and chooses the number
# of probes based on the leftover evaluations. It stops immediately when the
# budget is exhausted.
#
# Closest known influences: The design is reminiscent of “pattern search” /
# coordinate descent with step-size adaptation, augmented with occasional
# random restart to mitigate premature convergence.
#
# Novelty or unusual aspects: It uses an adaptive budget-aware probing schedule:
# each iteration’s number of coordinate probes is selected so that the remaining
# evaluations are respected, rather than assuming a fixed iteration count.
#
# Failure modes: On highly non-separable landscapes or when gradients are
# essential, coordinate probing may be slow. In very noisy objectives, the
# algorithm might overfit to transient improvements; however, the conservative
# step-size shrink and restart help somewhat. In extremely tight bounds or
# flat functions, many probes may clip to the same point, wasting evaluations.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._get_bounds(func, self.dim)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        dim = self.dim

        # Safety: ensure valid bounds
        lb, ub = self._sanitize_bounds(lb, ub, dim)

        # Evaluation budget management
        n_eval = 0
        max_eval = max(0, self.budget)

        def eval_f(x):
            nonlocal n_eval
            if n_eval >= max_eval:
                # Budget exhausted: should not happen if guarded properly.
                return np.inf
            n_eval += 1
            return float(func(np.asarray(x, dtype=float)))

        # Initialization: choose starting point uniformly in bounds
        if max_eval <= 0:
            # No evaluations allowed; return something deterministic-ish.
            x0 = lb.copy()
            return x0, float("inf")

        rng = np.random

        # Step size based on average range; robust across dimensions.
        span = np.maximum(ub - lb, 0.0)
        avg_span = float(np.mean(span)) if dim > 0 else 0.0
        # If span is zero, keep delta small but nonzero to avoid stagnation from exact ties.
        delta0 = 0.25 * avg_span if avg_span > 0 else 0.0
        if delta0 == 0.0:
            delta0 = 1e-12

        # Create an initial center and evaluate
        x_center = lb + rng.rand(dim) * (ub - lb) if dim > 0 else np.zeros(0)
        y_center = eval_f(x_center)
        best_x = x_center.copy()
        best_y = y_center

        # Parameters
        delta = delta0
        shrink = 0.7
        min_delta = 1e-12 * (1.0 + avg_span)
        # When stuck, do a random restart; frequency depends on budget.
        # We keep it small to avoid wasting evaluations.
        restart_patience = 3 + (dim // 4)
        stuck = 0

        # To keep it compact, do a loop that always checks remaining budget.
        # The number of probes per iteration is adaptive to remaining budget.
        while n_eval < max_eval and (delta > min_delta or max_eval - n_eval <= 5):
            remaining = max_eval - n_eval
            if remaining <= 0:
                break

            # Determine how many coordinate probes we can afford.
            # Each probe evaluates +delta and -delta (two evals) for each coordinate.
            # We'll allocate probes such that 1 evaluation for center update is not needed.
            # We already have the center value; we only evaluate candidates.
            # Candidate eval count = 2 * k + random_evals
            # We'll set random_evals small unless stuck.
            stuck_factor = 1 if stuck < restart_patience else 2
            base_random = 0 if stuck < 2 else 1
            random_evals = min(3, remaining // 4) if stuck_factor >= 2 else min(base_random, remaining // 8)

            # Choose k so total fits.
            # Ensure at least 1 probe if possible.
            # Avoid division by zero for small remaining.
            max_k_by_budget = max(0, (remaining - random_evals) // 2)
            k = min(dim, max(1, max_k_by_budget)) if max_k_by_budget > 0 else 0

            # Generate candidates
            improved = False
            best_local_y = best_y
            best_local_x = None

            # If we're stuck, do a random restart around best_x with larger delta,
            # otherwise local coordinate probing.
            if stuck >= restart_patience and remaining > 0:
                # Random restart: pick a new center uniformly in bounds
                # (counts as one evaluation via y_center-like).
                x_restart = lb + rng.rand(dim) * (ub - lb) if dim > 0 else np.zeros(0)
                y_restart = eval_f(x_restart)
                if y_restart < best_y:
                    best_y = y_restart
                    best_x = x_restart.copy()
                    improved = True
                x_center = best_x.copy()
                y_center = best_y
                stuck = 0
                # After a restart, set delta somewhat larger to re-explore.
                delta = max(delta, 0.5 * delta0)
                continue

            if k > 0:
                # Coordinate probing: randomly permute coordinate indices each iteration
                # so that different dimensions get attention over time.
                idx = rng.permutation(dim)[:k] if dim > 0 else np.zeros(0, dtype=int)

                # Try +/- along each chosen coordinate
                # Use clipping for boundary handling.
                for j in idx:
                    if n_eval >= max_eval:
                        break
                    x1 = x_center.copy()
                    x1[j] = np.clip(x1[j] + delta, lb[j], ub[j])
                    y1 = eval_f(x1)

                    if y1 < best_local_y:
                        best_local_y = y1
                        best_local_x = x1.copy()

                    if n_eval >= max_eval:
                        break

                    x2 = x_center.copy()
                    x2[j] = np.clip(x2[j] - delta, lb[j], ub[j])
                    y2 = eval_f(x2)

                    if y2 < best_local_y:
                        best_local_y = y2
                        best_local_x = x2.copy()

            # Add a few random candidates when stuck to escape local minima.
            if random_evals > 0 and n_eval < max_eval:
                for _ in range(random_evals):
                    if n_eval >= max_eval:
                        break
                    xr = lb + rng.rand(dim) * (ub - lb) if dim > 0 else np.zeros(0)
                    yr = eval_f(xr)
                    if yr < best_local_y:
                        best_local_y = yr
                        best_local_x = xr.copy()

            # Selection and replacement
            if best_local_x is not None and best_local_y < y_center:
                x_center = best_local_x
                y_center = best_local_y
            if best_local_x is not None and best_local_y < best_y:
                best_y = best_local_y
                best_x = best_local_x.copy()
                improved = True

            # Adaptation: shrink delta when no improvement in center; restart patience uses stuck counter.
            if improved:
                stuck = 0
                # Slightly relax delta to continue local exploitation at a useful scale.
                delta = max(delta * 0.95, min_delta)
            else:
                stuck += 1
                delta *= shrink

            # Ensure delta remains sensible and avoid too many tiny steps
            if avg_span > 0 and delta < 1e-15 * avg_span:
                delta = 1e-15 * avg_span

        return np.asarray(best_x, dtype=float), float(best_y)

    @staticmethod
    def _get_bounds(func, dim):
        # Priority:
        # 1) func.lower/func.upper
        # 2) func.bounds.lb/func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = func.bounds.lb
            ub = func.bounds.ub
        else:
            # Fallback: assume func has already embedded bounds elsewhere; if not,
            # use [0, 1] box to remain robust.
            lb = np.zeros(dim, dtype=float)
            ub = np.ones(dim, dtype=float)
        return lb, ub

    @staticmethod
    def _sanitize_bounds(lb, ub, dim):
        lb = np.asarray(lb, dtype=float).reshape(-1) if dim > 0 else np.asarray(lb, dtype=float).reshape(0)
        ub = np.asarray(ub, dtype=float).reshape(-1) if dim > 0 else np.asarray(ub, dtype=float).reshape(0)

        if dim == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        # Broadcast if scalar provided
        if lb.size == 1:
            lb = np.full(dim, float(lb[0]))
        if ub.size == 1:
            ub = np.full(dim, float(ub[0]))

        # If lengths mismatch, take first dim (robust over malformed inputs)
        if lb.size != dim:
            lb = (np.resize(lb, dim)).astype(float, copy=False)
        if ub.size != dim:
            ub = (np.resize(ub, dim)).astype(float, copy=False)

        # Ensure lb <= ub
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        return lo, hi
