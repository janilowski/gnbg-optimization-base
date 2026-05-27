# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimization
# algorithm using a population of candidate points sampled around a moving center.
# It performs iterative global-to-local search by shrinking/expanding the search radius
# based on observed improvements.
# Search state: The algorithm keeps a current center point, a step size (radius),
# a small population of sampled candidates, and tracks the best-so-far solution.
# Candidate generation: Each iteration samples candidates as center + step * noise
# where noise is drawn from a zero-mean normal distribution and also includes occasional
# coordinate-wise perturbations to help escape shallow local minima.
# Selection and replacement: The best candidate in the sampled population (lowest objective)
# becomes the new center; other candidates help validate progress. If no improvement is
# found, the step size is reduced to promote local refinement.
# Adaptation: Step size adapts using a simple success rule: improvements increase
# aggressiveness slightly, failures shrink the step size more.
# Exploration mechanisms: Randomized sampling around the center and periodic coordinate
# perturbations provide exploration.
# Exploitation mechanisms: When improvements occur, the step size is increased modestly;
# when they don't, the step size shrinks, focusing the search locally.
# Boundary handling: Candidates are clamped to the provided bounds after each perturbation.
# Budget strategy: The total number of objective evaluations never exceeds the provided
# budget; it divides the budget into an initial warm start and then repeated iterations,
# stopping when the remaining budget is insufficient.
# Closest known influences: This is loosely inspired by evolution strategies / CMA-style
# local search, but simplified to remain compact and budget-safe.
# Novelty or unusual aspects: The algorithm uses a minimal state and a deterministic
# accounting of evaluations per iteration, plus a hybrid of isotropic Gaussian sampling
# and coordinate-wise exploration.
# Failure modes: If bounds are extremely tight, the algorithm may stagnate due to
# frequent clamping; if the budget is too small, it may behave like random search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        if budget <= 0:
            raise ValueError("budget must be positive")

        # Read bounds from func.lower/upper or func.bounds.lb/ub
        if hasattr(func, "bounds") and func.bounds is not None:
            lb = np.asarray(getattr(func.bounds, "lb"), dtype=float)
            ub = np.asarray(getattr(func.bounds, "ub"), dtype=float)
        else:
            lb = np.asarray(getattr(func, "lower"), dtype=float)
            ub = np.asarray(getattr(func, "upper"), dtype=float)

        if lb.shape != (dim,) or ub.shape != (dim,):
            lb = np.broadcast_to(lb, (dim,)).astype(float, copy=False)
            ub = np.broadcast_to(ub, (dim,)).astype(float, copy=False)

        # Ensure finite bounds
        # If any non-finite bounds exist, fall back to a wide default range.
        span = ub - lb
        nonfinite = ~np.isfinite(span)
        if np.any(nonfinite):
            # Use a conservative span estimate for problematic dimensions.
            # If bounds are non-finite, approximate using absolute of finite parts or 1.
            finite_lb = np.where(np.isfinite(lb), lb, 0.0)
            finite_ub = np.where(np.isfinite(ub), ub, finite_lb + 1.0)
            lb = finite_lb
            ub = finite_ub
            span = ub - lb

        # Replace zero span dims with tiny span to avoid stuck radius
        span = np.where(span > 0, span, 1e-12)

        def project(x):
            # Clamp to bounds
            return np.minimum(ub, np.maximum(lb, x))

        evals = 0

        # Determine evaluation allocation.
        # Use a small population size tied to dimension but keep it compact.
        # This avoids exceeding budget and keeps per-iteration overhead low.
        # pop_size should be >= 2 to provide selection pressure.
        pop_size = int(max(2, min(8, 2 + dim // 3)))
        # Warm start: evaluate a few random points around center.
        warm = int(min(pop_size, max(1, budget // 10)))
        remaining = budget

        # Initialize center as midpoint with a small random perturbation.
        center = (lb + ub) * 0.5
        # Initial step: fraction of span (robust even for high dim).
        step = 0.25 * span

        best_x = None
        best_y = np.inf

        def eval_candidate(x):
            nonlocal evals, best_x, best_y
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Warm start: evaluate center and some random candidates.
        # Always count these evaluations.
        if remaining <= 0:
            return center, best_y
        center = project(center)
        eval_candidate(center)

        remaining -= 1
        for _ in range(min(warm - 1, remaining)):
            # Gaussian around center with isotropic noise scaled by step.
            x = center + np.random.randn(dim) * step
            x = project(x)
            eval_candidate(x)
        remaining = budget - evals

        # Main loop: each iteration evaluates up to pop_size candidates.
        # Always ensure we don't exceed budget.
        # Success rule updates step size based on whether improvement occurs.
        # Keep step bounded to avoid numerical issues.
        min_step = 1e-12 * span
        max_step = 0.6 * span

        # Coordinate perturbation schedule (every few iterations).
        # This helps escape along axes when isotropic noise is too smooth.
        iter_count = 0
        while evals < budget and remaining > 0:
            iter_count += 1

            # Determine how many candidates we can still evaluate
            k = min(pop_size, budget - evals)
            if k <= 0:
                break

            # Generate candidates
            # Candidate 0: always include current center to guarantee at least one baseline
            # and allow selection even if k=1 (though k>=2 typically).
            candidates = np.empty((k, dim), dtype=float)
            candidates[0] = center

            # For reproducibility and robustness across dimensions:
            # Use Gaussian sampling; occasionally add coordinate-wise jitter.
            for i in range(1, k):
                if (iter_count % 3 == 0) and (i % 2 == 0):
                    # Coordinate perturbation: pick a random coordinate
                    j = np.random.randint(0, dim)
                    # Small to moderate jitter on one dimension
                    coord_noise = np.random.randn() * step[j]
                    x = center.copy()
                    x[j] = x[j] + coord_noise
                else:
                    # Isotropic Gaussian
                    x = center + np.random.randn(dim) * step
                candidates[i] = project(x)

            # Evaluate and pick best candidate this iteration
            # (we also maintain global best via eval_candidate).
            best_i_y = np.inf
            best_i_x = None
            for i in range(k):
                y = eval_candidate(candidates[i])
                if y < best_i_y:
                    best_i_y = y
                    best_i_x = candidates[i].copy()

            # Adaptation based on improvement relative to previous center
            # (use objective value of current center if tracked).
            # We don't store center's objective each time; instead, compare against best_y
            # improvement is global; for local step rule compare best candidate vs global best history isn't ideal.
            # We'll approximate by comparing best candidate this iteration to current best at time start:
            # Since global best could already be better than center, we keep a snapshot.
            # Use best candidate of iteration vs current best_y_before to measure success.
            # Implement by comparing best_i_y with "center_y" estimated by reevaluating is budget-unsafe.
            # So we use heuristic: success if best_i_y is less than previous best_x value's objective proxy.
            # We'll track last_center_best_y whenever we update center.
            if not hasattr(self, "_center_best_y"):
                self._center_best_y = best_i_y
            center_before = self._center_best_y

            if best_i_y < center_before - 1e-15:
                # Success: move center to the iteration best and increase step slightly
                center = best_i_x
                self._center_best_y = best_i_y
                step = np.minimum(max_step, step * (1.0 + 0.20 / np.sqrt(dim + 1.0)))
            else:
                # Failure: keep the global best but shrink step to exploit locally
                # Move center toward best_i_x only a bit (conservative).
                # This helps avoid drift when improvements are marginal.
                center = project(0.75 * center + 0.25 * best_i_x)
                self._center_best_y = min(self._center_best_y, best_i_y)
                step = np.maximum(min_step, step * 0.70)

            remaining = budget - evals

        # Fallback if best_x not set (shouldn't happen)
        if best_x is None:
            best_x = project(center)
            best_y = float(func(best_x))

        return best_x, best_y
