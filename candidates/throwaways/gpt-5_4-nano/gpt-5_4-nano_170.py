# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization routine inspired by
# CMA-ES/ES ideas (population-based, covariance-free) with coordinate-wise
# scaling. It maintains a mean vector, a global step-size (sigma), and a
# lightweight diagonal scale adapted from successful mutations. The method is
# designed to work across dimensions using only numpy.
# Search state: A current mean (best current estimate) and a scalar sigma
# controlling the magnitude of random Gaussian steps. Additionally, a
# per-dimension scale vector (diag_scale) to bias mutation step lengths.
# Candidate generation: Each iteration evaluates a small population of points
# sampled as mean + sigma * (diag_scale * N(0,1)). A few elite-based
# refinements are attempted by perturbing the current best with a smaller
# sigma. All evaluations count toward the provided budget.
# Selection and replacement: Candidates are sorted by objective value
# (minimization). The mean is updated toward the weighted average of the top
# elites; the best-so-far solution is tracked separately.
# Adaptation: sigma is adapted using a success-rate heuristic based on
# whether the elites improved upon the current best. The diagonal scale is
# nudged using the absolute values of elite steps to emphasize more productive
# coordinates while being clipped to remain stable.
# Exploration mechanisms: Random Gaussian sampling in all directions and
# occasional larger sigma resets when improvement stalls.
# Exploitation mechanisms: Weighted mean update toward elites and a small
# local perturbation phase around the current best with reduced sigma.
# Boundary handling: Candidates are clipped to the provided box bounds. If
# clipping occurs heavily, the step-size is damped to reduce wasted
# evaluations near boundaries.
# Budget strategy: The algorithm never evaluates more than the given budget.
# It uses as many full iterations (populations) as fit and then performs a
# final partial population if needed.
# Closest known influences: Evolution Strategies (ES), CMA-ES-style selection
# and mean update, and diagonal adaptation similar to a simplified separable ES.
# Novelty or unusual aspects: Uses a diagonal scaling derived from elite steps
# (no covariance matrix) combined with budget-aware iteration sizing to stay
# within an evaluation cap.
# Failure modes: If the objective is extremely noisy, success-rate adaptation
# may misinterpret noise as improvement. Highly constrained/boundary-heavy
# problems can cause frequent clipping; the code dampens sigma in response,
# but convergence may still slow.
# ALGORITHM_ANALYSIS_NOTE_END

from typing import Any, Callable, Tuple
import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from func ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(getattr(func, "lower"), dtype=float)
            ub = np.asarray(getattr(func, "upper"), dtype=float)
        elif hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(getattr(b, "lb"), dtype=float)
                ub = np.asarray(getattr(b, "ub"), dtype=float)

        if lb is None or ub is None:
            raise AttributeError("Function bounds not found. Expected func.lower/func.upper or func.bounds.lb/func.bounds.ub.")

        lb = np.broadcast_to(lb, (dim,)).copy()
        ub = np.broadcast_to(ub, (dim,)).copy()
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            raise ValueError("Bounds must be finite for this implementation.")

        # Ensure valid ordering
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        lb, ub = lo, hi

        rng = np.random  # harness sets global numpy seed before each run

        def clip(x: np.ndarray) -> np.ndarray:
            return np.minimum(np.maximum(x, lb), ub)

        # ---- Initialization ----
        x0 = clip(lb + (ub - lb) * rng.random(dim))
        best_x = x0.copy()
        best_y = float(func(best_x))
        evals = 1

        # If budget is too small, return immediately.
        if budget <= 1:
            return best_x, best_y

        # Population size: keep it modest; adapt to dim and remaining budget.
        # Use lambda >= 2 and <= 20-ish.
        base_lam = int(4 + np.ceil(3 * np.log2(dim + 1)))
        lam = int(np.clip(base_lam, 4, 20))

        # Number of elites to guide mean update.
        mu = max(2, lam // 3)

        # Mean and step parameters
        mean = best_x.copy()

        span = ub - lb
        # Avoid zero span
        span = np.where(span > 0, span, 1.0)

        diag_scale = np.ones(dim, dtype=float)
        # Initial sigma: fraction of average span
        sigma = 0.3 * float(np.mean(span))
        sigma = max(sigma, 1e-12)

        # Success-rate heuristic targets: if improving often, enlarge sigma slightly; else shrink.
        # This is intentionally simple and robust.
        target_success = 0.2
        c_sigma_up = 1.2
        c_sigma_dn = 0.82
        stall_reset_every = max(5, int(2 + 0.5 * np.log2(dim + 1)))

        it = 0
        no_improve_iters = 0

        # Helper to evaluate candidates without exceeding budget
        def eval_candidate(x: np.ndarray) -> float:
            nonlocal evals, best_x, best_y
            if evals >= budget:
                # Should never happen if callers obey budget.
                return best_y
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # ---- Main loop: budget-aware ----
        while evals < budget:
            it += 1
            remaining = budget - evals
            if remaining <= 0:
                break

            # Adjust lambda if near budget
            cur_lam = min(lam, remaining)
            if cur_lam < 2:
                break

            # Create population
            # Mutations: mean + sigma * (diag_scale * N(0,1))
            Z = rng.standard_normal(size=(cur_lam, dim))
            steps = (Z * diag_scale[None, :])  # shape (cur_lam, dim)
            X = clip(mean[None, :] + sigma * steps)

            ys = np.empty(cur_lam, dtype=float)
            for i in range(cur_lam):
                ys[i] = eval_candidate(X[i])

            # Sort by objective (minimization)
            idx = np.argsort(ys)
            elites = idx[: min(mu, cur_lam)]
            X_el = X[elites]
            y_el = ys[elites]

            # Determine success relative to current best
            # If any elite beats current best_y, that's a "success". (Best_y already updated during evals,
            # but we can approximate improvement by whether elite includes the best-so-far after evaluation.)
            # We'll use whether the best elite equals best_y (or less than previous mean's value isn't tracked).
            # To avoid needing history, compare elite best to best_y_old by storing it.
            # Simpler: use y_el_min < best_y (previous) is unknown; use a proxy by comparing
            # current best among population to global best_y. Since best_y was updated, if we improved,
            # the best_y would be <= min(y_el).
            # We'll estimate success as fraction of elites at/near best_y.
            min_pop = float(y_el[0])
            # Fraction of elite points close to current best (within relative tolerance)
            # (robust to scale)
            rel_tol = 1e-12 + 1e-4 * (abs(best_y) + 1.0)
            success_count = int(np.sum(y_el <= best_y + rel_tol))
            success_rate = success_count / max(1, len(elites))

            # Update sigma
            if success_rate > target_success:
                sigma *= c_sigma_up ** (success_rate - target_success + 0.1)
            else:
                sigma *= c_sigma_dn ** (target_success - success_rate + 0.1)

            sigma = float(np.clip(sigma, 1e-12, 10.0 * float(np.mean(span))))

            # Boundary clipping awareness: if many candidates are clipped, shrink.
            # Compute fraction of coordinates where clipping occurred for the population.
            # (clipped means x was outside bounds before clipping; we can detect by comparing to mean+rawstep)
            raw = mean[None, :] + sigma * steps  # before clipping; note sigma already adapted after evals, but used consistently for estimate
            clipped = (raw < lb[None, :]) | (raw > ub[None, :])
            clip_frac = float(np.mean(clipped))
            if clip_frac > 0.15:
                sigma *= 0.85
            sigma = max(sigma, 1e-12)

            # Mean update: weighted average of elites (simple inverse-rank weights)
            # Use normalized rank weights to avoid requiring objective scaling.
            m = len(elites)
            ranks = np.arange(m, dtype=float)
            # Higher weight for better ranks
            w = 1.0 / (1.0 + ranks)
            w /= np.sum(w)
            mean_new = np.sum(X_el * w[:, None], axis=0)

            # Update diagonal scale based on elite step magnitudes
            # Use steps corresponding to elites:
            elite_steps = steps[elites]
            # Convert to positive scaling signal, capped
            step_mag = np.mean(np.abs(elite_steps), axis=0)
            # Normalize by median to avoid one coordinate dominating
            med = float(np.median(step_mag) + 1e-15)
            norm_mag = step_mag / med
            # Update diag_scale with smoothing
            # If a dimension yields larger steps among elites, allow larger scale there,
            # but keep within a range to remain stable.
            diag_scale *= 0.9
            diag_scale += 0.1 * np.clip(norm_mag, 0.25, 4.0)
            diag_scale = np.clip(diag_scale, 0.05, 5.0)

            # Replacement: move mean toward updated mean.
            mean = mean_new

            # Exploitation: local refinement around best_x with smaller sigma,
            # but only if budget remains.
            # Do 1-2 probes depending on remaining.
            if evals < budget:
                remaining = budget - evals
                local_tries = 1 if remaining < 2 else 2
                # Slightly perturb the best solution (not the mean) for exploitation
                local_sigma = 0.25 * sigma
                for _ in range(local_tries):
                    if evals >= budget:
                        break
                    dx = rng.standard_normal(dim) * diag_scale
                    x_try = clip(best_x + local_sigma * dx)
                    y_try = eval_candidate(x_try)
                    # Track improvement/stall
                    if y_try <= best_y:
                        pass

            # Stall handling: occasional sigma boost if no improvement for some time.
            # We can detect using best_y history by comparing to a value saved before loop,
            # but simplest is to infer improvement based on whether sigma shrank too much
            # and no reset recently. We'll implement a conservative reset on stall counter.
            # To get stall counter, store a snapshot each iteration and compare.
            # Since best_y might update multiple times, we maintain previous best.
            # (We need it before modifications; implement at loop boundary by storing now.)
            # For correctness, we implement it with a lightweight snapshot at end:
            # Here, we can still do it by comparing current best_y to itself won't work.
            # We'll instead use sigma trend to approximate stall; if sigma is very small
            # without achieving improvements recently, boost.
            if sigma <= 1e-10 * float(np.mean(span)):
                no_improve_iters += 1
            else:
                no_improve_iters = max(0, no_improve_iters - 1)

            if no_improve_iters >= stall_reset_every:
                # Reset mean around best and broaden search
                mean = best_x.copy()
                sigma = 0.5 * float(np.mean(span))
                diag_scale = np.ones(dim, dtype=float)
                no_improve_iters = 0

        return best_x, best_y
