# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer based on
# Evolution Strategies (ES) with a rotating global best, coordinate-free
# Gaussian sampling, and an optional local refinement step.
#
# Search state: Tracks current best solution x_best and its value y_best,
# a running strategy step size sigma, and a success counter to adapt sigma.
# Each evaluation consumes budget units.
#
# Candidate generation: For each generation, samples lambda candidate points
# by adding isotropic Gaussian noise to the current best (or incumbent).
# Optionally, includes a small local search around the best using a few
# coordinate perturbations (cheap and dimension-agnostic).
#
# Selection and replacement: Evaluates all candidates, selects the lowest
# value as the new incumbent and (if improved) updates global best.
# Uses (mu+lambda) style behavior: the best of parent/offspring survives.
#
# Adaptation: Uses 1/5th success rule-like adaptation: if enough sampled
# points improve the incumbent, sigma increases; otherwise sigma decreases.
# This keeps the method responsive to varying problem scales.
#
# Exploration mechanisms: Wide initial sigma, Gaussian sampling around the
# incumbent, and periodic local perturbations promote exploration.
#
# Exploitation mechanisms: Sampling concentrated around the best-so-far and
# local coordinate perturbations refine promising regions.
#
# Boundary handling: Clamps candidates to the provided bounds to ensure
# feasibility. Sampling continues until evaluation budget is exhausted.
#
# Budget strategy: Never exceeds the provided evaluation budget. The code
# carefully caps the number of generations and candidate evaluations based on
# remaining budget.
#
# Closest known influences: Simplified (mu+lambda)-ES with sigma adaptation
# resembling the classic 1/5 success rule; boundary handling via clamping.
#
# Novelty or unusual aspects: Combines global ES sampling with a lightweight,
# dimension-scaled local “coordinate jitter” step while maintaining strict budget
# accounting.
#
# Failure modes: On extremely noisy or ill-scaled objectives, sigma adaptation
# might oscillate or converge prematurely. For very small budgets, it primarily
# performs a few incumbent-centered samples and may not fully exploit.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        n = self.dim
        budget = max(1, int(self.budget))

        # Read bounds from func in a robust way
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
        if lb is None or ub is None:
            raise AttributeError("Function must provide bounds via func.lower/func.upper or func.bounds.lb/ub.")

        lb = np.broadcast_to(lb, (n,)).copy()
        ub = np.broadcast_to(ub, (n,)).copy()
        if np.any(ub < lb):
            # Swap if bounds are reversed
            tmp = lb.copy()
            lb = np.minimum(lb, ub)
            ub = np.maximum(tmp, ub)

        def clamp(x):
            return np.minimum(np.maximum(x, lb), ub)

        # Pick an initial point: midpoint (stable) + tiny noise if degenerate
        x0 = 0.5 * (lb + ub)
        width = ub - lb
        if np.all(width == 0):
            x_best = x0.copy()
        else:
            # Use a small relative perturbation; harness sets numpy seed.
            eps = 1e-3
            scale = np.where(width > 0, width, 1.0)
            x_best = clamp(x0 + eps * scale * np.random.randn(n))

        evals_used = 0

        def eval_obj(x):
            nonlocal evals_used
            if evals_used >= budget:
                # Should never happen if caller respects budget.
                return np.inf
            y = func(x)
            evals_used += 1
            return float(y)

        y_best = eval_obj(x_best)

        # Initialize sigma relative to domain size; handle tiny widths.
        domain = np.where(width > 0, width, 1.0)
        sigma = 0.25 * np.median(domain)
        if sigma <= 0 or not np.isfinite(sigma):
            sigma = 1.0

        # Generation sizing: keep lambda small for large dims or tiny budgets.
        # Use at least 1 candidate per generation.
        # Aim roughly for O(10*n) evaluations but strictly bounded by budget.
        # lambda should not exceed remaining budget.
        base_lambda = max(4, min(16, 2 + n // 2))
        mu = 1  # not explicitly used; selection is based on best offspring.
        # Success threshold for 1/5 rule approximation
        success_threshold = max(1, int(0.2 * base_lambda + 0.5))

        # Local refinement settings
        do_local = n <= 50  # keep it modest in higher dimensions
        local_trials = 2 * n if n > 1 else 1  # coordinate jitter count
        # Ensure local refinement doesn't blow the budget; we'll cap per generation.

        # Main loop: each iteration evaluates up to lambda candidates plus optional local.
        # Ensure we never exceed the budget.
        while evals_used < budget:
            remaining = budget - evals_used
            lam = min(base_lambda, remaining)
            if lam <= 0:
                break

            # Candidate generation around incumbent (x_best).
            # Isotropic Gaussian step with clamped feasibility.
            # Shape: (lam, n)
            noise = np.random.randn(lam, n)
            # Scale noise by sigma and domain sensitivity (use domain median scaling).
            step = sigma * noise
            X = clamp(x_best[None, :] + step)

            # Evaluate candidates
            best_gen_x = x_best
            best_gen_y = y_best
            improve_count = 0

            for i in range(lam):
                yi = eval_obj(X[i])
                if yi < best_gen_y:
                    best_gen_y = yi
                    best_gen_x = X[i].copy()
                if yi < y_best:
                    improve_count += 1

            # Selection/replacement
            parent_survives = y_best <= best_gen_y
            if not parent_survives:
                x_best = best_gen_x
                y_best = best_gen_y

            # Adapt sigma using success count
            # Increase if enough improvements, otherwise decrease.
            # Factor choices keep sigma stable across dimensions.
            if improve_count >= success_threshold:
                sigma *= 1.4
            else:
                sigma *= 0.85

            # Keep sigma within reasonable bounds relative to the domain
            # so it doesn't collapse or explode.
            dom_med = np.median(domain)
            sigma = float(np.clip(sigma, 1e-12 * dom_med, 2.0 * (dom_med if dom_med > 0 else 1.0) + 1.0))

            # Optional local coordinate jitter around current best
            if do_local and evals_used < budget and (lam == base_lambda or improve_count > 0):
                remaining = budget - evals_used
                t = min(local_trials, remaining)
                if t > 0:
                    # Choose a subset of coordinates to probe
                    coords = np.random.choice(n, size=min(n, max(1, t // 2)), replace=False)
                    # For each chosen coordinate, try +/- a small step
                    # Total trials capped by t.
                    trial_xs = []
                    trial_signs = []
                    small = 0.1 * sigma
                    for c in coords:
                        if len(trial_xs) >= t:
                            break
                        for sgn in (-1.0, 1.0):
                            if len(trial_xs) >= t:
                                break
                            trial_x = x_best.copy()
                            trial_x[c] = x_best[c] + sgn * small
                            trial_xs.append(clamp(trial_x))
                            trial_signs.append(sgn)

                    for xi in trial_xs[:t]:
                        yi = eval_obj(xi)
                        if yi < y_best:
                            y_best = yi
                            x_best = xi.copy()
                    # If local step finds improvements, slightly increase exploitation (sigma down a bit)
                    # to refine; if not, decrease more aggressively.
                    # We infer "found improvement" from last comparison against best before local.
                    # (This is approximate; still robust.)
                    # If any improvement occurred, sigma tends to reduce to focus.
                    # We'll check quickly by recomputing best_gen_y wasn't retained; so just use improve_count.
                    if improve_count > 0:
                        sigma *= 0.95
                    else:
                        sigma *= 0.90

        return x_best, y_best
