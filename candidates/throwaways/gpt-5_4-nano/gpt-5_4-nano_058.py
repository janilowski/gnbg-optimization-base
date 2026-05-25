# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm for continuous domains.
# It combines random sampling, short local coordinate-wise refinement, and periodic restarts using a
# step-size schedule. The algorithm maintains the best-so-far solution (lowest objective value).
# Search state: The algorithm tracks remaining function evaluations, current best point/value,
# a current step-size (global scale for proposing moves), and a small working candidate each round.
# Candidate generation: Candidates are created either by (1) uniform random points in bounds,
# (2) adding Gaussian perturbations to the best point, and (3) doing coordinate-wise tweaks
# around the current best by +/- step_size on one dimension.
# Selection and replacement: Every evaluated candidate is compared to the incumbent best; if it
# improves, it becomes the new best. For local refinement, successful coordinate moves reduce
# step-size; failures increase step-size slightly to escape plateaus.
# Adaptation: Step-size is adapted based on whether local/perturbation moves improve the incumbent.
# The step-size decays over time to shift from exploration to exploitation.
# Exploration mechanisms: Initial random sampling and periodic restart when improvement stalls.
# Exploitation mechanisms: Gaussian perturbations around the best point and coordinate-wise local
# search that attempts to refine each dimension.
# Boundary handling: All proposed points are clipped to the provided bounds. If bounds are
# degenerate (lower == upper), the dimension remains fixed.
# Budget strategy: The algorithm never exceeds the provided evaluation budget. It carefully
# accounts for how many objective calls are left and truncates the plan accordingly.
# Closest known influences: This resembles a lightweight evolution-strategy / stochastic direct-search
# hybrid with coordinate-wise local refinement (ideas from ES and coordinate search), adapted to
# strict evaluation budgets.
# Novelty or unusual aspects: The coordinate-wise refinement is budget-aware and uses an adaptive
# per-iteration step schedule, blended with restart-triggered exploration.
# Failure modes: In very noisy or highly non-smooth objectives, coordinate refinement may be misled.
# In extremely narrow feasible regions, random exploration may be inefficient (though clipping
# guarantees feasibility). If the budget is tiny, it may act mostly as random search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # --- Resolve bounds ---
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub.")

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        d = self.dim
        if lb.size != d or ub.size != d:
            # Be tolerant if dim matches but shapes differ slightly; otherwise fail loudly.
            raise ValueError(f"Bounds dimension mismatch: expected {d}, got {lb.size} and {ub.size}.")

        # Ensure numeric stability for degenerate bounds.
        width = ub - lb
        eps = 1e-12
        width_safe = np.where(np.abs(width) > eps, width, 0.0)
        is_fixed = np.isclose(width_safe, 0.0)

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        # --- Budget control ---
        remaining = self.budget
        if remaining <= 0:
            # No evaluations allowed; return something deterministic within bounds.
            x0 = np.where(is_fixed, lb, (lb + ub) / 2.0)
            return x0, float("inf")

        eval_count = 0

        def eval_f(x):
            nonlocal remaining, eval_count
            if remaining <= 0:
                return float("inf")
            y = func(x)
            remaining -= 1
            eval_count += 1
            return float(y)

        # --- Initialization ---
        # Start from midpoint, and also do a small random burn-in.
        x_best = clip(np.where(is_fixed, lb, (lb + ub) / 2.0))
        y_best = eval_f(x_best)

        # Exploration amount adapts to budget.
        # Use a small fraction for random points, but at least 1 if possible.
        # cap at dim + 3 to limit overhead.
        burn_in = min( max(1, d + 1), remaining )
        # Random uniform burn-in
        if burn_in > 0:
            for _ in range(burn_in):
                r = np.random.random(d)
                x = clip(lb + r * width_safe)
                y = eval_f(x)
                if y < y_best:
                    y_best, x_best = y, x

        # Initial step size: fraction of typical width (ignore fixed dims)
        typical = np.median(np.abs(width_safe[~is_fixed])) if np.any(~is_fixed) else 0.0
        if typical <= 0:
            # All dims fixed => nothing to do
            return x_best, y_best

        # Scale factor tries to work across dimensions.
        step = 0.25 * typical
        step_min = 1e-12 * typical
        step_max = 0.75 * typical

        # Helper to schedule decays
        def decay_factor(t, total):
            # Gentle exponential decay towards exploitation.
            if total <= 0:
                return 1.0
            return np.exp(-3.0 * (t / total))

        # --- Main loop: restart + refinement cycles ---
        # Number of cycles based on budget.
        # Each cycle uses a few evaluations; keep the plan simple and budget-aware.
        # At most ~12 cycles or enough to spend budget with safety.
        max_cycles = 12
        # We will allocate approximately remaining evaluations across cycles.
        cycles = min(max_cycles, max(1, remaining // max(1, d // 2 + 3)))

        # Track improvement for restart logic
        no_improve = 0
        stall_limit = 3

        for cycle in range(cycles):
            if remaining <= 0:
                break

            t_total = max(1, cycles)
            step = float(np.clip(step * decay_factor(cycle, t_total), step_min, step_max))

            improved_this_cycle = False

            # --- Exploitation: gaussian perturbation around best ---
            # Evaluate a few candidates towards the best.
            # Number of perturbations depends on budget and dimension.
            perturbs = min(4, remaining)
            for _ in range(perturbs):
                if remaining <= 0:
                    break
                # Gaussian step; ignore fixed dims by zeroing their perturbation
                z = np.random.randn(d)
                if np.any(is_fixed):
                    z[is_fixed] = 0.0
                x = clip(x_best + step * z)
                y = eval_f(x)
                if y < y_best:
                    y_best, x_best = y, x
                    improved_this_cycle = True

            # --- Local refinement: coordinate-wise tweaks ---
            # Budget-aware: refine up to k coordinates.
            if remaining > 0 and not np.all(is_fixed):
                # Choose number of coordinate attempts.
                # Try more coordinates when we have enough budget.
                k = min(d, max(2, d // 2))
                # Use a permutation to avoid bias.
                coords = np.random.permutation(d)[:k]

                # Coordinate step magnitude: slightly smaller than global step
                cstep = step * 0.7
                for i in coords:
                    if remaining <= 0:
                        break
                    if is_fixed[i]:
                        continue

                    # Try + and - (2 evaluations). Budget-aware truncation.
                    # If only 1 eval remains, try one direction only.
                    if remaining >= 2:
                        x1 = x_best.copy()
                        x2 = x_best.copy()
                        x1[i] = x1[i] + cstep
                        x2[i] = x2[i] - cstep
                        x1 = clip(x1)
                        x2 = clip(x2)
                        y1 = eval_f(x1)
                        if y1 < y_best:
                            y_best, x_best = y1, x1
                            improved_this_cycle = True
                        y2 = eval_f(x2)
                        if y2 < y_best:
                            y_best, x_best = y2, x2
                            improved_this_cycle = True
                    else:
                        # One evaluation left: choose a random direction
                        direction = 1.0 if np.random.random() < 0.5 else -1.0
                        x = x_best.copy()
                        x[i] = x[i] + direction * cstep
                        x = clip(x)
                        y = eval_f(x)
                        if y < y_best:
                            y_best, x_best = y, x
                            improved_this_cycle = True

            # --- Adaptation / restart logic ---
            if improved_this_cycle:
                # Successful cycle: reduce step to exploit more.
                step = max(step_min, step * 0.7)
                no_improve = 0
            else:
                no_improve += 1
                # If no improvement, increase step moderately to escape.
                step = min(step_max, step * 1.15)

            # If stalled, do a restart with random samples around a random point
            if no_improve >= stall_limit and remaining > 0:
                no_improve = 0
                # Create a new center: either uniform random or a mix with best.
                # Using mix helps when best is already good.
                mix = 0.5
                r = np.random.random(d)
                x_rand = clip(lb + r * width_safe)
                x_restart = clip(mix * x_best + (1.0 - mix) * x_rand)

                # Evaluate x_restart and a few small perturbations around it.
                y_restart = eval_f(x_restart)
                if y_restart < y_best:
                    y_best, x_best = y_restart, x_restart

                # A handful of restart perturbations
                if remaining > 0:
                    n = min(3, remaining)
                    for _ in range(n):
                        z = np.random.randn(d)
                        if np.any(is_fixed):
                            z[is_fixed] = 0.0
                        x = clip(x_restart + 0.5 * step * z)
                        y = eval_f(x)
                        if y < y_best:
                            y_best, x_best = y, x
                # Continue loop with new step size
                step = min(step_max, max(step_min, 0.6 * typical))

        # Final clamp and return
        x_best = clip(np.asarray(x_best, dtype=float))
        return x_best, float(y_best)
