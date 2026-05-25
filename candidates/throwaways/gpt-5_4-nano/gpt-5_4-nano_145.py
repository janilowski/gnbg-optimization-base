import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy combining
# random restarts, population-based sampling, and a local derivative-free
# refinement around the current best. Works for any dimension and uses a
# strict evaluation budget.
# Search state: Maintains a current best point (best_x) and best value
# (best_y), plus an adaptive step size (sigma). During each iteration it
# samples candidate points around the best and also uses occasional global
# exploration.
# Candidate generation: Uses a Gaussian distribution around the current best
# with step size sigma and respects bounds via projection. Also generates
# purely random candidates with a small probability to escape local minima.
# Selection and replacement: Evaluates each candidate (within remaining budget)
# and keeps the best among them as the new incumbent. If a better point is
# found, sigma is adapted upward slightly; otherwise it is shrunk.
# Adaptation: sigma is increased when improvement is observed to continue
# exploring; sigma is reduced when iterations fail to improve to refine locally.
# Exploration mechanisms: Includes random sampling (global restart-like
# behavior) controlled by exploration rate and uses occasional wider Gaussian
# proposals.
# Exploitation mechanisms: Primarily relies on Gaussian perturbations around
# the best point to perform local search without gradients.
# Boundary handling: Candidate points are clipped to the provided bounds after
# sampling, ensuring feasibility.
# Budget strategy: Uses an internal evaluation counter and never calls the
# objective more than the given budget; stops early if no evaluations remain.
# Closest known influences: Inspired by evolution strategies / CMA-ES-like
# sampling (without covariance learning) and classic step-size adaptation
# heuristics from derivative-free optimizers.
# Novelty or unusual aspects: Uses a combination of (1) strict budget-aware
# ask/evaluate loops, (2) adaptive step-size based on success/failure rate,
# and (3) a lightweight fallback refinement when progress stalls.
# Failure modes: If the budget is extremely small or the objective is very
# noisy, the method may not sufficiently sample the space; clipping can also
# lead to many boundary-equal points on tightly bounded problems.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim

        # Read bounds from either func.lower/func.upper or func.bounds.lb/ub.
        if hasattr(func, "bounds") and func.bounds is not None:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            lb = np.asarray(getattr(func, "lower"), dtype=float)
            ub = np.asarray(getattr(func, "upper"), dtype=float)

        # Ensure shapes and validity.
        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))
        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds size does not match dim.")

        # Handle potential infinite bounds by substituting a reasonable scale
        # derived from finite bounds; if all are infinite, default to [-1, 1].
        finite_lb = np.isfinite(lb)
        finite_ub = np.isfinite(ub)
        any_finite = np.any(finite_lb) or np.any(finite_ub)
        if any_finite:
            # Replace infinities using the other side or a default span.
            # (Still allow clipping for finite portions.)
            default_span = 1.0
            span = np.where(np.isfinite(ub) & np.isfinite(lb), ub - lb, default_span)
            span = np.where(np.isfinite(span), span, default_span)

            # Choose fill values for infinities close to the finite side.
            lb_fill = np.where(finite_lb, lb, (ub - span))
            ub_fill = np.where(finite_ub, ub, (lb + span))
            lb = lb_fill
            ub = ub_fill
        else:
            lb = np.full(dim, -1.0)
            ub = np.full(dim, 1.0)

        # If bounds are degenerate, expand slightly to avoid zero sigma.
        spans = ub - lb
        spans = np.where(spans > 0, spans, 1e-12)
        span_mean = float(np.mean(spans))

        rng = np.random

        evals = 0
        best_x = None
        best_y = None

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_once(x):
            nonlocal evals, best_x, best_y
            x = clip(np.asarray(x, dtype=float))
            y = func(x)
            evals += 1
            if best_y is None or y < best_y:
                best_y = float(y)
                best_x = x.copy()
            return y

        # If budget is extremely small, just evaluate at one point.
        if self.budget <= 0:
            raise ValueError("budget must be positive")

        # Initial candidates: a few points to set a baseline.
        # Use at most budget evaluations.
        # Always include a mid-point and a random point.
        mid = lb + 0.5 * (ub - lb)

        # Choose initial sigma based on bounds.
        sigma = 0.35 * span_mean
        sigma = max(sigma, 1e-12)

        # Keep a small population each iteration while respecting budget.
        # This is compact and dimension-independent.
        base_pop = 1 + (10 if dim <= 10 else 6 if dim <= 30 else 4)

        # Evaluate initial points.
        n_init = min(self.budget, 2 + (1 if dim > 1 else 0))
        eval_once(mid)
        if evals < self.budget and n_init >= 2:
            eval_once(lb + rng.random(dim) * (ub - lb))
        if evals < self.budget and n_init >= 3:
            # A center + noise around mid to help on non-symmetric functions.
            x = mid + rng.normal(0.0, 0.1, size=dim) * spans
            eval_once(x)

        # Main loop: budget-aware iterations.
        # Each iteration samples candidates around the best and possibly explores globally.
        # Stop when evaluations run out.
        success_streak = 0
        fail_streak = 0

        while evals < self.budget:
            remaining = self.budget - evals

            # Adapt population size to remaining budget.
            pop = min(base_pop, remaining)

            # Exploration rate decreases as budget gets used.
            # Keeps some chance to escape, but focuses later on exploitation.
            t = evals / self.budget
            explore_rate = 0.25 * (1.0 - t) + 0.05  # between ~0.05 and 0.30

            # Occasionally consider a few purely random points.
            # Determine number of global samples.
            n_global = int(np.floor(pop * explore_rate))
            n_global = min(n_global, pop)
            n_local = pop - n_global

            improved = False
            old_best = best_y

            # Global exploration samples.
            for _ in range(n_global):
                if evals >= self.budget:
                    break
                x = lb + rng.random(dim) * (ub - lb)
                eval_once(x)

            # Local exploitation samples: Gaussian around best with adaptive sigma.
            for i in range(n_local):
                if evals >= self.budget:
                    break

                # Use a mixture of step sizes:
                # - mostly sigma,
                # - occasionally a larger step to avoid premature convergence.
                if rng.random() < 0.15:
                    local_sigma = sigma * rng.uniform(1.5, 3.0)
                else:
                    local_sigma = sigma

                # Perturbation direction: Gaussian.
                z = rng.normal(0.0, 1.0, size=dim)
                # Scale per-dimension by bounds span to be robust.
                x = best_x + (local_sigma / max(span_mean, 1e-12)) * (spans * 0.5) * z

                eval_once(x)
                # Minor early break if we already improved a lot: not necessary,
                # but can save evals on very tight budgets.
                if best_y is not None and old_best is not None and best_y < old_best:
                    improved = True

            if best_y is None:
                # Should never happen.
                best_y = float("inf")
                best_x = mid.copy()

            # Adapt sigma based on improvement.
            if best_y < old_best:
                improved = True
                success_streak += 1
                fail_streak = 0
                # If we keep improving, expand sigma slightly to continue searching.
                sigma *= 1.12 if success_streak < 3 else 1.05
            else:
                fail_streak += 1
                success_streak = 0
                # If we fail, shrink sigma to refine locally.
                sigma *= 0.82

            # Keep sigma within reasonable limits.
            sigma = float(np.clip(sigma, 1e-12, 2.5 * span_mean))

            # Lightweight refinement if stalled:
            # Use a small "coordinate-ish" probing around the best with fewer steps.
            if fail_streak >= 4 and evals < self.budget:
                # Try to reduce failure by probing along a few random directions.
                # This uses remaining budget carefully.
                remaining = self.budget - evals
                k = min(2 + dim // 10, remaining)  # small and budget-aware
                # Reduce step size for refinement.
                refine_sigma = max(sigma * 0.35, 1e-12)
                for _ in range(k):
                    if evals >= self.budget:
                        break
                    d = rng.normal(0.0, 1.0, size=dim)
                    d_norm = np.linalg.norm(d)
                    if d_norm == 0:
                        continue
                    d /= d_norm
                    # Probe both directions.
                    step = rng.choice([-1.0, 1.0]) * refine_sigma * span_mean
                    x = best_x + step * d
                    eval_once(x)

                # After refinement attempt, reset streak.
                fail_streak = 0

        return best_x, best_y
