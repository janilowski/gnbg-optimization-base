# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimizer
# for the GNBG-style benchmark setting. It uses a repeated multi-start
# strategy combined with local “coordinate-like” exploration and an adaptive
# step-size controlled by success in reducing the objective.
# Search state: The algorithm maintains a current best point (x_best, y_best),
# a per-stage step size (sigma), and keeps track of how many function evaluations
# have been consumed to never exceed the provided budget.
# Candidate generation: At each stage, it proposes candidates by perturbing
# the current point along random Gaussian directions scaled by sigma, plus
# a small coordinate-like refinement by using random coordinate signs.
# Selection and replacement: Each proposed candidate that improves the objective
# replaces the incumbent (and may update the global best). If no improvement is
# found for several attempts, the step size is reduced.
# Adaptation: sigma adapts based on recent success rate: success increases sigma
# slightly (encourage broader search), while repeated failures shrink sigma
# (promote local refinement).
# Exploration mechanisms: Early/mid stages include more exploratory sampling
# (larger sigma and multiple random directions).
# Exploitation mechanisms: Later stages rely more on small perturbations around
# the best-so-far and coordinate-like moves.
# Boundary handling: All candidate points are projected back into the feasible
# bounds by clipping to [lb, ub].
# Budget strategy: The budget is enforced by wrapping all evaluations through
# a counter; every candidate costs exactly one evaluation. The remaining budget
# determines how many iterations/tries can be executed.
# Closest known influences: The design is reminiscent of simple evolution strategies
# (ES)/CMA-free strategies and adaptive random search with step-size control.
# Novelty or unusual aspects: It mixes Gaussian-direction sampling with lightweight
# coordinate-like refinement while keeping bookkeeping minimal and robust.
# Failure modes: In highly constrained or extremely ill-scaled problems, step-size
# adaptation might stagnate; the multi-start mechanism mitigates this but cannot
# guarantee global optimality within limited budgets.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # Resolve bounds from either func.lower/upper or func.bounds.lb/ub
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            # support both b.lb/b.ub and lower/upper naming inside bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            elif hasattr(b, "lower") and hasattr(b, "upper"):
                lb = np.asarray(b.lower, dtype=float)
                ub = np.asarray(b.upper, dtype=float)
        if lb is None or ub is None:
            raise AttributeError("Could not determine bounds from func. Provide func.lower/upper or func.bounds.lb/ub.")

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError(f"Bounds dimension mismatch: expected dim={self.dim}, got lb={lb.size}, ub={ub.size}")

        # Ensure proper ordering
        swapped = lb > ub
        if np.any(swapped):
            lb2 = lb.copy()
            ub2 = ub.copy()
            lb2[swapped] = ub[swapped]
            ub2[swapped] = lb[swapped]
            lb, ub = lb2, ub2

        # If budget is tiny, still evaluate safely.
        n_eval = 0

        def clamp(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_one(x):
            nonlocal n_eval
            if n_eval >= self.budget:
                # Never exceed budget: if called too many times, return +inf to avoid breaking.
                return float("inf")
            x = clamp(np.asarray(x, dtype=float))
            y = float(func(x))
            n_eval += 1
            return y

        # Initialize: pick starting points via Latin-ish random affine combination
        # based on bounds; include current center as one candidate if budget allows.
        span = ub - lb
        # Avoid zero span causing sigma=0 everywhere: provide a small fallback scale.
        fallback_scale = 1.0
        span_norm = np.where(span != 0, np.abs(span), fallback_scale)
        # A reasonable initial sigma: 0.2 * average span, bounded away from 0.
        sigma0 = 0.2 * float(np.mean(span_norm))
        sigma0 = max(sigma0, 1e-8)

        x_best = None
        y_best = float("inf")

        # Helper to update incumbent
        def consider(x, y):
            nonlocal x_best, y_best
            if y < y_best:
                x_best = np.asarray(x, dtype=float).copy()
                y_best = y

        # Compute a few initial points
        center = lb + 0.5 * (ub - lb)

        if self.budget <= 0:
            # No evaluations allowed; return something deterministic within bounds
            x_best = clamp(center)
            return x_best, float("inf")

        # How many multi-starts? Keep conservative so we leave budget for local search.
        # At least 1 start, at most ~5.
        max_starts = min(5, self.budget)
        starts = max(1, max_starts)

        # Evaluate center if possible
        if n_eval < self.budget:
            y = eval_one(center)
            x_best = clamp(center)
            y_best = y

        # Additional starts
        for _ in range(starts - 1):
            if n_eval >= self.budget:
                break
            # Random point in box: uniform in bounds
            u = np.random.rand(self.dim)
            x0 = lb + u * (ub - lb)
            y0 = eval_one(x0)
            consider(x0, y0)

        if x_best is None:
            x_best = clamp(center)

        # Budget left determines number of "stages" and tries.
        remaining = self.budget - n_eval
        if remaining <= 0:
            return x_best, y_best

        # Stages: logarithmic-ish so it works across dims/budgets.
        # Each stage performs several candidate evaluations.
        # More stages for larger budgets.
        stages = int(np.clip(np.log2(remaining + 1.0) + 2, 2, 12))
        # Convert stages into per-stage number of evaluations.
        # Keep at least 3 tries per stage when possible.
        tries_per_stage = max(3, remaining // stages)

        sigma = sigma0
        # Maintain a small success history to adapt sigma
        success_streak = 0
        failure_streak = 0

        # Local search loop
        # Uses x_best as incumbent; explores around it.
        for stage in range(stages):
            if n_eval >= self.budget:
                break

            # Exploration/exploitation blend:
            # Early stages: more random directions. Later stages: smaller step and more focused.
            t = stage / max(1, (stages - 1))
            # Decrease sigma over time but let adaptation react too.
            sigma_stage = sigma * (1.0 - 0.5 * t)

            # For each stage, attempt multiple improvements
            stage_budget = min(tries_per_stage, self.budget - n_eval)
            any_improved = False

            # Candidate count: a mix of Gaussian directions and coordinate-like moves
            # Ensure we cover both for robustness.
            n_gauss = int(np.clip(stage_budget * (0.7 - 0.2 * t), 1, stage_budget))
            n_coord = stage_budget - n_gauss

            # Gaussian direction sampling
            for _ in range(n_gauss):
                if n_eval >= self.budget:
                    break
                # Random direction with Gaussian components
                d = np.random.randn(self.dim)
                # Normalize to avoid extremely large steps in high dimensions
                dn = np.linalg.norm(d)
                if dn == 0:
                    d = np.ones(self.dim)
                    dn = np.sqrt(self.dim)
                d = d / dn

                # Step: sigma_stage scaled plus a small additional noise for escape
                step = sigma_stage * np.random.randn()
                x_cand = x_best + step * d

                y_cand = eval_one(x_cand)
                if y_cand < y_best:
                    consider(x_cand, y_cand)
                    any_improved = True

            # Coordinate-like refinement
            if n_coord > 0 and n_eval < self.budget:
                # pick random coordinates; try +/- in random order
                for _ in range(n_coord):
                    if n_eval >= self.budget:
                        break
                    coord = np.random.randint(0, self.dim)
                    sign = 1.0 if np.random.rand() < 0.5 else -1.0
                    # Use a per-coordinate step based on sigma_stage and box size.
                    # If span is tiny, fall back to sigma_stage.
                    per_coord_scale = span_norm[coord]
                    step_mag = sigma_stage * (0.5 + 0.5 * np.random.rand())
                    # Scale to match coordinate magnitude loosely
                    step = sign * step_mag * (1.0 if per_coord_scale == fallback_scale else (per_coord_scale / (fallback_scale)))
                    x_cand = np.array(x_best, copy=True)
                    x_cand[coord] += step
                    y_cand = eval_one(x_cand)
                    if y_cand < y_best:
                        consider(x_cand, y_cand)
                        any_improved = True

            # Adapt sigma based on stage improvement
            if any_improved:
                success_streak += 1
                failure_streak = 0
                # If consistently improving, slightly increase exploratory power
                if success_streak >= 2:
                    sigma *= 1.15
                else:
                    sigma *= 1.05
            else:
                failure_streak += 1
                success_streak = 0
                # On repeated failure, shrink step aggressively
                shrink = 0.75 if failure_streak >= 2 else 0.85
                sigma *= shrink

            # Hard guard: sigma should not collapse to zero
            min_sigma = 1e-12 * (1.0 + np.mean(span_norm))
            if sigma < min_sigma:
                sigma = min_sigma

        return x_best, y_best
