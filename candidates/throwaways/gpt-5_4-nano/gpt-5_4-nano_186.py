# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, budget-aware black-box minimizer
# for continuous domains using a mix of global random sampling and local
# coordinate-wise refinement. It works for any dimension and only relies on
# function evaluations.
# Search state: The algorithm maintains a single incumbent best solution
# (best_x, best_y) and a step-size vector controlling the size of local
# perturbations.
# Candidate generation: It generates candidates by adding uniform random
# steps scaled by the current step size, plus a few structured coordinate
# moves around the incumbent. Most proposals are random, ensuring exploration.
# Selection and replacement: Every newly evaluated point is compared to the
# incumbent. If it improves the objective (lower value), it replaces the
# incumbent and the local step size is reduced; otherwise, the step size is
# gradually increased after repeated non-improvements.
# Adaptation: The step size is adapted based on whether recent trials improved.
# Exploration mechanisms: Global exploration is performed by repeatedly
# sampling random points across the bounds early in the budget.
# Exploitation mechanisms: Once a decent incumbent exists, the algorithm
# performs localized perturbations and coordinate-wise probing to refine.
# Boundary handling: All candidate points are clipped to the provided lower and
# upper bounds to stay feasible.
# Budget strategy: The total number of objective evaluations is capped at the
# provided budget. The implementation tracks remaining evaluations and stops
# immediately when the budget is exhausted.
# Closest known influences: The design is inspired by simple derivative-free
# strategies combining random search with local step-size adaptation (similar
# in spirit to evolution strategies / pattern search), but implemented in a
# minimal, robust way for black-box benchmarking.
# Novelty or unusual aspects: The code uses a hybrid of random global sampling,
# isotropic random steps, and lightweight coordinate moves with a unified,
# vector step-size adaptation.
# Failure modes: If the objective is very noisy or has extremely narrow
# minima, random probing may miss the region. In such cases, the algorithm
# still returns the best point found within the budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # --- Extract bounds ---
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and func.bounds is not None:
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        if lb is None or ub is None:
            raise AttributeError(
                "Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )

        lb = np.broadcast_to(lb, (self.dim,)).astype(float)
        ub = np.broadcast_to(ub, (self.dim,)).astype(float)
        if np.any(ub < lb):
            raise ValueError("Upper bounds must be >= lower bounds for all dimensions.")

        rng = np.random

        # --- Budget accounting ---
        remaining = self.budget

        def eval_at(x):
            nonlocal remaining
            if remaining <= 0:
                # Should never happen if used correctly; keep safe.
                return np.inf
            remaining -= 1
            y = func(x)
            return float(y)

        # Degenerate bounds: width may be zero.
        width = ub - lb
        # Avoid all-zero width step: keep minimal epsilon scale.
        base_scale = float(np.max(width)) if np.max(width) > 0 else 1.0
        step = 0.3 * base_scale * np.ones(self.dim, dtype=float)

        # --- Initial sampling / incumbent selection ---
        best_x = lb.copy()
        best_y = np.inf

        # Evaluate a few initial points: incumbent candidates.
        # Choose count to spend enough evaluations but keep within budget.
        init_count = min(max(2, self.dim + 2), max(1, self.budget))
        for _ in range(init_count):
            # If bounds are degenerate, random sampling will collapse to the same point.
            u = rng.random(self.dim)
            x = lb + u * width
            y = eval_at(x)
            if y < best_y:
                best_y = y
                best_x = x

        # If budget is already exhausted, return best so far.
        if remaining <= 0:
            return best_x, best_y

        # --- Main optimization loop ---
        # We adapt step size: reduce on improvement, increase on stagnation.
        # Keep loop lightweight: random trials dominate, with occasional
        # coordinate probes for exploitation.
        non_improve = 0
        improve = 0

        # Hyperparameters scaled for robustness across dimensions.
        # More dimensions -> slightly fewer coordinate probes.
        coord_probe_prob = 0.3 if self.dim <= 10 else 0.15
        coord_candidates = max(1, int(round(0.2 * self.dim)))  # average probes per iteration
        # Clamp factors to avoid numerical issues.
        shrink_factor = 0.85
        grow_factor = 1.08

        while remaining > 0:
            # Decide number of trials per "iteration" based on remaining budget.
            # This keeps selection/replacement simple and bounded.
            trials = 1
            if remaining >= 10:
                trials = 3
            elif remaining >= 3:
                trials = 2

            for _t in range(trials):
                if remaining <= 0:
                    break

                # Candidate generation:
                # 1) Random isotropic move around incumbent (primary exploration/exploitation).
                # 2) With some probability, perform coordinate-wise probe moves (pattern search-ish).
                r = rng.random()
                if r < coord_probe_prob:
                    # Coordinate-wise exploitation:
                    # Probe a subset of coordinates with +/- step components.
                    # Randomly pick coordinate indices.
                    k = min(self.dim, coord_candidates)
                    idx = rng.choice(self.dim, size=k, replace=False)
                    # Direction choice: +/- with equal probability, possibly different per coord.
                    signs = rng.choice([-1.0, 1.0], size=k)

                    # Evaluate a small set of coordinate moves. Pick one coordinate move
                    # at a time to keep evaluation count precise.
                    j = int(idx[rng.randint(0, k)])
                    # Use a component-scaled step, but ensure nonzero move if possible.
                    comp_step = step[j] if step[j] > 0 else 0.0
                    x = best_x.copy()
                    x[j] = x[j] + signs[list(idx).index(j)] * comp_step

                    # If comp_step is zero, fall back to a small random move.
                    if comp_step == 0.0:
                        u = rng.random(self.dim) - 0.5
                        x = best_x + 0.05 * base_scale * u

                else:
                    # Random step in a box around incumbent.
                    # Use a mix of uniform noise and Gaussian noise for stability.
                    # Clipped to bounds afterwards.
                    # Scale by step, and occasionally use larger steps early in the run.
                    noise = (rng.standard_normal(self.dim) * 0.5 + (rng.random(self.dim) - 0.5) * 0.5)
                    scale = 1.0
                    if non_improve == 0 and improve > 0:
                        # if improving, don't overshoot much
                        scale = 0.9
                    elif non_improve >= 2:
                        # stagnation: increase exploration
                        scale = 1.1
                    x = best_x + scale * step * noise

                # Boundary handling: clip to feasible bounds.
                x = np.minimum(ub, np.maximum(lb, x))

                # If clipping collapses to incumbent, consider a random jitter attempt.
                # (Still cheap; ensures we can make progress in degenerate cases.)
                if np.all(x == best_x):
                    jitter = (rng.random(self.dim) - 0.5) * 0.02 * base_scale
                    x = np.minimum(ub, np.maximum(lb, best_x + jitter))

                y = eval_at(x)
                if y < best_y:
                    best_y = y
                    best_x = x
                    improve += 1
                    non_improve = 0
                    # Exploitation: shrink step size to refine around new best.
                    step = step * shrink_factor
                else:
                    non_improve += 1
                    # Exploration: gradually grow step size after failures.
                    if non_improve >= 2:
                        step = step * grow_factor
                        non_improve = 1  # avoid runaway growth too fast

                # Optional: stop early if step becomes tiny and bounds are tight.
                # This doesn't affect correctness; it only helps efficiency, but budget
                # is the primary limit.
                if remaining <= 0:
                    break

            # If budget remains, but step becomes extremely small relative to widths,
            # we can re-inject some exploration to avoid total stagnation.
            if remaining > 0:
                effective_width = np.maximum(width, 1e-12)
                if np.max(step / effective_width) < 1e-6:
                    # Reheat step based on widths.
                    step = 0.2 * effective_width

        return best_x, best_y
