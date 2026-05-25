# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimization
# algorithm using a mixture of local coordinate searches and global random restarts.
# It works with a strict evaluation budget and is designed to be robust across many
# dimensions.
# Search state: The algorithm maintains a current best point x_best, its objective
# value y_best, and a step-size sigma that shrinks when improvement is found and
# grows modestly when stagnation occurs. It also keeps an incumbent x_cand for
# evaluating trial moves.
# Candidate generation: Each iteration draws a small set of candidate points.
# With probability based on progress, it generates either:
#  (a) Gaussian perturbations around x_best scaled by sigma (global/local mixture),
#  (b) coordinate-wise proposals (sign flips and one-step offsets) using sigma.
# It uses randomness from NumPy (the harness can control determinism via np.random.seed).
# Selection and replacement: Among evaluated candidates (including the incumbents
# themselves where relevant), the algorithm keeps the best found. If a candidate
# improves y_best, it replaces x_best and records a success streak.
# Adaptation: sigma is adapted: it decreases after improvements (finer local search)
# and increases after several iterations without improvement (to escape local minima).
# Exploration mechanisms: Random restarts are triggered after prolonged stagnation.
# During normal operation, Gaussian perturbations provide exploration.
# Exploitation mechanisms: Coordinate-wise proposals and shrink-on-improvement
# concentrate search around promising regions.
# Boundary handling: After proposing points, it clips them to the provided bounds
# (from func.lower/func.upper or func.bounds.lb/ub). This ensures feasibility.
# Budget strategy: It strictly tracks remaining evaluations. Each candidate evaluation
# decreases the budget; the code never calls the objective after budget is exhausted.
# Closest known influences: The behavior is inspired by common budgeted derivative-free
# patterns (e.g., coordinate search + success-based step-size adaptation) but is
# implemented compactly without external dependencies.
# Novelty or unusual aspects: The algorithm blends vector Gaussian trials with
# cheap coordinate proposals and uses a stagnation-based sigma restart schedule,
# tuned to remain budget-aware.
# Failure modes: If the objective is extremely noisy or highly non-smooth, step-size
# adaptation can oscillate; clipping may also trap search on boundaries. The
# algorithm mitigates this with restarts and sigma growth, but cannot guarantee
# optimality on all problems.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget_total = self.budget
        if budget_total <= 0:
            # No evaluations allowed: return zeros within bounds if possible.
            lb, ub = self._get_bounds(func, dim)
            return np.clip(np.zeros(dim, dtype=float), lb, ub), float("inf")

        lb, ub = self._get_bounds(func, dim)
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        # Ensure bounds are valid even if provided strangely.
        # (Swap if lb > ub elementwise)
        swap_mask = lb > ub
        if np.any(swap_mask):
            lb2 = lb.copy()
            ub2 = ub.copy()
            lb2[swap_mask], ub2[swap_mask] = ub2[swap_mask], lb2[swap_mask]
            lb, ub = lb2, ub2

        # Evaluation counter / wrapper
        evals = 0
        best_x = None
        best_y = float("inf")

        def eval_at(x):
            nonlocal evals, best_x, best_y
            if evals >= budget_total:
                return best_y
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.array(x, dtype=float, copy=True)
            return y

        # Initial sigma: fraction of typical range
        ranges = ub - lb
        # Prevent zeros scale; if a component has zero range, keep proposals stable.
        # The overall sigma should still be reasonable.
        base_range = np.median(ranges[ranges > 0]) if np.any(ranges > 0) else 1.0
        if not np.isfinite(base_range) or base_range <= 0:
            base_range = 1.0

        # Start from a feasible random point and also try the midpoint.
        x_mid = (lb + ub) / 2.0
        x0 = lb + (ub - lb) * np.random.rand(dim)
        x0 = np.clip(x0, lb, ub)

        # Evaluate both if budget permits
        eval_at(x_mid)
        if evals < budget_total:
            eval_at(x0)

        if best_x is None:
            # Should not happen, but keep safe
            best_x = x0

        # Stagnation control
        sigma = 0.3 * base_range  # initial step-size
        success_streak = 0
        no_improve = 0
        # budget-aware iteration count
        iters = max(1, budget_total // max(1, (2 + min(10, dim))))  # rough heuristic

        # Coordinate sampling parameters
        # cap number of coordinate trials to keep each iteration cheap in high dims
        coord_trials = min(dim, 10)

        # Main loop: each iteration uses a fixed small budget slice
        for _ in range(iters):
            if evals >= budget_total:
                break

            # Decide exploration vs exploitation based on recent progress.
            # When improving, exploit more; otherwise explore more.
            stagnation_ratio = no_improve / max(1, 4 + (budget_total // max(1, iters)))
            p_explore = float(np.clip(0.25 + 0.65 * stagnation_ratio, 0.15, 0.9))

            candidates = []

            # Gaussian perturbation candidates (global/local mixture)
            # Use 2 or 3 candidates depending on budget.
            g_count = 2 if dim <= 20 else 1
            g_count = 3 if (budget_total - evals) >= 3 else max(1, g_count)
            for _k in range(g_count):
                step = sigma * np.random.randn(dim)
                x = np.clip(best_x + step, lb, ub)
                candidates.append(x)

            # Coordinate-wise exploitation: try a few coordinate sign directions
            # Choose random coordinates each iteration.
            if dim > 0 and (budget_total - evals) > 0:
                idxs = np.random.choice(dim, size=coord_trials, replace=False) if coord_trials < dim else np.arange(dim)
                # Choose step magnitudes: one uses sigma, one uses 0.5*sigma
                for i in idxs:
                    for sgn in (1.0, -1.0):
                        x = best_x.copy()
                        # If range is tiny, sigma may overshoot; still clipped.
                        x[i] = x[i] + sgn * sigma
                        x = np.clip(x, lb, ub)
                        candidates.append(x)
                        if len(candidates) >= 2 * coord_trials + g_count + 3:
                            break
                    if len(candidates) >= 2 * coord_trials + g_count + 3:
                        break

            # Ensure we don't exceed remaining budget:
            # We'll evaluate in order of "likely value" by mixing exploration then exploitation.
            # Heuristic: perturbations closer to best_x are likely; coordinate moves often too.
            # We'll simply cap number of evaluations.
            remaining = budget_total - evals
            if remaining <= 0:
                break
            max_evals_this_iter = min(6 + coord_trials, remaining)
            # Prefer Gaussian first when exploring, coordinate first when exploiting.
            if np.random.rand() < p_explore:
                # Explore: keep initial Gaussian candidates first
                ordered = candidates
            else:
                # Exploit: coordinate candidates first; keep stable order by re-splitting
                # (Gaussian candidates are the first g_count in our construction)
                ordered = candidates[g_count:] + candidates[:g_count] if len(candidates) > g_count else candidates

            # If sigma too large and everything clips to boundaries, reduce step proactively.
            # (Quick diagnostic: proposed points differ from best_x)
            # We'll measure after ordering and cap, but before evaluating all.
            improved_before = best_y

            # Evaluate capped candidates until budget is consumed.
            local_evaluations = 0
            for x in ordered[:max_evals_this_iter]:
                if evals >= budget_total:
                    break
                y = eval_at(x)
                local_evaluations += 1

            # Adapt sigma based on whether we improved
            if best_y < improved_before:
                success_streak += 1
                no_improve = 0
                # Shrink step size: faster convergence near optima
                sigma = max(sigma * (0.7 ** min(3, success_streak)), 1e-12 * base_range)
            else:
                no_improve += 1
                success_streak = 0
                # Grow step size to escape stagnation
                sigma = min(sigma * 1.25 + 1e-12, (ub - lb).max() if np.any(ub > lb) else 1e6)

            # Restarts when heavily stagnated and budget remains
            if no_improve >= 5 and (budget_total - evals) > 2 * dim:
                # Restart from a fresh random feasible point, but still keep best seen.
                x_restart = lb + (ub - lb) * np.random.rand(dim)
                x_restart = np.clip(x_restart, lb, ub)
                # Small evaluation burst around restart to avoid a single random hit
                # Cap by remaining evaluations
                remaining = budget_total - evals
                if remaining > 0:
                    eval_at(x_restart)
                if evals < budget_total and remaining > 1:
                    eval_at(np.clip(x_restart + 0.1 * sigma * np.random.randn(dim), lb, ub))
                if evals < budget_total and remaining > 2:
                    eval_at(np.clip(x_restart - 0.1 * sigma * np.random.randn(dim), lb, ub))
                # Reset sigma after restart
                sigma = 0.3 * base_range
                no_improve = 0
                success_streak = 0

        # Safety: if never evaluated (shouldn't), pick midpoint
        if best_x is None:
            best_x = np.clip(x_mid, lb, ub)
            best_y = float(func(best_x))
        return best_x, best_y

    @staticmethod
    def _get_bounds(func, dim):
        # Try: func.lower / func.upper
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(getattr(func, "lower"), dtype=float)
            ub = np.asarray(getattr(func, "upper"), dtype=float)
            if lb.shape == () or ub.shape == ():
                lb = np.full(dim, float(lb))
                ub = np.full(dim, float(ub))
            return lb, ub

        # Try: func.bounds.lb / func.bounds.ub
        if hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(getattr(b, "lb"), dtype=float)
                ub = np.asarray(getattr(b, "ub"), dtype=float)
                if lb.shape == () or ub.shape == ():
                    lb = np.full(dim, float(lb))
                    ub = np.full(dim, float(ub))
                return lb, ub

        # Fallback: if no bounds exist, use a standard box centered at 0.
        # Note: The benchmark should provide bounds; this is just robustness.
        lb = np.full(dim, -5.0, dtype=float)
        ub = np.full(dim, 5.0, dtype=float)
        return lb, ub
