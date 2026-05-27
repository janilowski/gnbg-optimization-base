# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer using a
# multi-start local search with a coordinate-wise pattern search and
# occasional stochastic restarts. It maintains the best-so-far solution and
# dynamically reduces step sizes while respecting a strict evaluation budget.
# Search state: Keeps current point x, current step size (per-coordinate
# scale), and the globally best point found (best_x, best_y). It also tracks
# remaining evaluations to guarantee the budget is never exceeded.
# Candidate generation: Forms trial points by probing along positive/negative
# coordinate directions (coordinate pattern search) around the current best.
# It additionally proposes random candidates for exploration during restarts.
# Selection and replacement: Accepts a trial if it improves (strictly lowers)
# the objective. If no improvement occurs after a full coordinate sweep,
# it shrinks the step size; if the step becomes very small, it triggers a
# restart from a new random point.
# Adaptation: Step size shrinks geometrically when progress stalls, and expands
# slightly upon successful improvements to accelerate convergence.
# Exploration mechanisms: Random restarts and occasional random directions
# when progress stalls, to escape local minima.
# Exploitation mechanisms: Deterministic coordinate pattern search around the
# best point to refine solutions.
# Boundary handling: All candidates are clipped to the provided bounds.
# Budget strategy: Every objective call decrements a shared counter; the
# algorithm never performs calls beyond the given budget, and always returns
# the best point found so far.
# Closest known influences: Inspired by coordinate pattern search / Hooke-Jeeves
# style iterations combined with multi-start stochastic restarts.
# Novelty or unusual aspects: Uses a per-dimension step scale and a simple
# schedule that adapts both the sweep step magnitude and restart frequency based
# on remaining budget to stay robust across dimensions.
# Failure modes: For highly noisy, deceptive, or extremely flat objectives,
# progress may be slow and the algorithm may rely more on restarts. With very
# small budgets, only a limited number of probes are performed.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Read bounds from func lower/upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective function must provide bounds via func.lower/func.upper or func.bounds.lb/ub.")

        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        # Handle scalar/shape mismatches robustly
        lb = np.broadcast_to(lb, (dim,)).astype(float, copy=False)
        ub = np.broadcast_to(ub, (dim,)).astype(float, copy=False)

        # Ensure valid bounds ordering (clip range if needed)
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        lb, ub = lo, hi

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        # Evaluation wrapper with hard budget
        remaining = budget

        def evaluate(x):
            nonlocal remaining
            if remaining <= 0:
                # Should never happen if budget accounting is correct
                return None
            remaining -= 1
            y = func(x)
            return float(y)

        # If budget is extremely small, still evaluate once at a feasible point
        rng = np.random
        # Initialize best by evaluating a random feasible point if possible
        # (We may evaluate a couple points depending on budget)
        def random_point():
            if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)) and np.any(ub > lb):
                u = rng.rand(dim)
                return lb + u * (ub - lb)
            # Fallback: center of bounds
            return 0.5 * (lb + ub)

        best_x = None
        best_y = np.inf

        # Create initial point(s)
        x0 = random_point()
        y0 = evaluate(x0) if budget > 0 else np.inf
        if y0 is not None:
            best_x, best_y = x0.copy(), y0

        # Early exit if no more evaluations allowed
        if remaining <= 0 or budget <= 1:
            return best_x, best_y

        # Choose initial step size as a fraction of the box width
        box_width = ub - lb
        # Avoid zero-width dimensions: they effectively can't move
        # Use a small epsilon fraction of typical width or 1.0 if width is zero.
        finite_width = box_width[np.isfinite(box_width)]
        typical = float(np.median(finite_width)) if finite_width.size else 1.0
        eps_floor = 1e-12
        base_step = 0.25 * typical
        # Per-dimension step scale; if a dimension is fixed, keep step at 0
        step = np.where(box_width > 0, 0.25 * box_width, 0.0)
        # If everything is fixed, return best_x
        if np.all(step == 0):
            return best_x, best_y

        # Multi-start schedule: allocate a few restarts depending on budget
        # Each restart includes at least one evaluation (already have one).
        # Coordinate sweeps are expensive: 2*dim probes per sweep.
        sweep_cost = 2 * dim
        # How many full sweeps can we afford in total, conservatively
        max_sweeps = max(1, (remaining // max(1, sweep_cost)))
        # Limit restarts so we don't waste budget
        max_restarts = 1
        if remaining > sweep_cost:
            # More budget allows additional restarts
            max_restarts = int(min(5, remaining // (max(1, 2 * sweep_cost))))
        max_restarts = max(max_restarts, 1)

        # State variables for current local search
        current_x = best_x.copy()
        current_y = best_y

        # Pattern search parameters
        shrink = 0.5
        expand = 1.2
        min_step_ratio = 1e-7  # relative to typical box width

        # Remaining evaluation safety
        def can_probe():
            return remaining > 0

        # Local search loop with budget checks
        restarts_done = 0
        while remaining > 0 and restarts_done < max_restarts:
            # If step is too small, restart or stop local refinement
            if np.all(step <= eps_floor + min_step_ratio * abs(typical)):
                break

            improved_in_sweep = False

            # Coordinate-wise pattern: try +/- step in each dimension
            # Use a randomized coordinate order to reduce bias
            order = np.arange(dim)
            rng.shuffle(order)

            for j in order:
                if not can_probe():
                    break

                if step[j] == 0:
                    continue

                # Try negative
                trial = current_x.copy()
                trial[j] = trial[j] - step[j]
                trial = clip(trial)
                y_trial = evaluate(trial)
                if y_trial is not None and y_trial < current_y:
                    current_x, current_y = trial, y_trial
                    if current_y < best_y:
                        best_x, best_y = current_x.copy(), current_y
                    improved_in_sweep = True
                    # Small expansion to intensify exploitation
                    step[j] = step[j] * expand
                    continue

                # Try positive
                if not can_probe():
                    break
                trial = current_x.copy()
                trial[j] = trial[j] + step[j]
                trial = clip(trial)
                y_trial = evaluate(trial)
                if y_trial is not None and y_trial < current_y:
                    current_x, current_y = trial, y_trial
                    if current_y < best_y:
                        best_x, best_y = current_x.copy(), current_y
                    improved_in_sweep = True
                    step[j] = step[j] * expand

            # If no coordinate improved, shrink and optionally jitter
            if not improved_in_sweep:
                step = step * shrink

                # Occasional exploration: random move around best_x scaled by step
                # Only if budget remains for at least a couple evaluations
                if remaining > min(10, 2 * dim) and rng.rand() < 0.35:
                    # Use a Gaussian perturbation biased toward current step size
                    # (strongly clipped to stay in bounds)
                    if can_probe():
                        z = rng.randn(dim)
                        trial = best_x + z * (0.5 * step)
                        trial = clip(trial)
                        y_trial = evaluate(trial)
                        if y_trial is not None and y_trial < best_y:
                            best_x, best_y = trial, y_trial
                            current_x, current_y = best_x.copy(), best_y
                            step = np.maximum(step, 1e-18) * 1.05  # mild re-growth
                    # Try a second random point if we have budget
                    if remaining > 0 and can_probe():
                        trial = random_point()
                        y_trial = evaluate(trial)
                        if y_trial is not None and y_trial < best_y:
                            best_x, best_y = trial, y_trial
                            current_x, current_y = best_x.copy(), best_y
                            step = np.where(box_width > 0, 0.25 * box_width, 0.0)
                # If step is shrinking too far, attempt a restart
                if np.all(step <= eps_floor + min_step_ratio * abs(typical)):
                    break

            # Update current based on best (helps if current drifted)
            # but keep some local continuity by only syncing when better found.
            if current_y > best_y:
                current_x, current_y = best_x.copy(), best_y

            # After a sweep, maybe restart from best but with a new point if progress was limited
            restarts_done += 1
            if remaining > 0 and (not improved_in_sweep) and restarts_done < max_restarts:
                # Restart near best_x (guided) rather than purely random
                if np.any(step > 0):
                    z = rng.randn(dim)
                    trial = best_x + z * (0.8 * step)
                    trial = clip(trial)
                else:
                    trial = random_point()
                y_trial = evaluate(trial)
                if y_trial is not None and y_trial <= best_y:
                    current_x, current_y = trial, y_trial
                    if y_trial < best_y:
                        best_x, best_y = trial, y_trial
                else:
                    # fallback: pure random restart if guided failed
                    if remaining > 0:
                        trial = random_point()
                        y_trial = evaluate(trial)
                        if y_trial is not None and y_trial < best_y:
                            best_x, best_y = trial, y_trial
                            current_x, current_y = best_x.copy(), best_y

        # Final safety: ensure best_x is valid
        if best_x is None:
            best_x = random_point()
            if budget > 0:
                best_y = evaluate(best_x) if remaining >= 0 else best_y
        return best_x, best_y
