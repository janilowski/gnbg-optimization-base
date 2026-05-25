# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# for general (box-constrained) domains using a stochastic population of
# candidate points. It combines global exploration (random samples) with
# local exploitation around the current best using a derivative-free
# “step-size shrinking” search similar in spirit to evolution strategies.
# Search state: The algorithm maintains a small population of points, their
# objective values, a global best (best_x, best_y), and a scalar step-size
# (sigma) controlling how far new candidates are sampled from the best.
# Candidate generation: Each iteration generates offspring by perturbing the
# current best with Gaussian noise scaled by sigma, while also periodically
# drawing uniform random points for diversification.
# Selection and replacement: Offspring and current population are merged;
# the best individuals (lowest objective values) are kept as the next
# population. The global best is updated whenever an improved point is found.
# Adaptation: sigma is adapted using a simple success rule: if the new
# offspring improve the best in a given iteration, sigma shrinks or grows
# modestly to balance exploration vs exploitation.
# Exploration mechanisms: Uniform random sampling is used early and
# occasionally to escape local minima, with the probability of exploration
# decreasing as the budget is consumed.
# Exploitation mechanisms: Gaussian perturbations around best_x with
# step-size sigma are used every iteration to refine the solution.
# Boundary handling: All candidates are clipped to the provided bounds to
# ensure they remain feasible. The bounds are read from func.lower/upper or
# func.bounds.lb/ub.
# Budget strategy: The algorithm strictly tracks the number of objective
# evaluations and stops when the budget is exhausted, never evaluating
# beyond the provided limit. It uses small batch sizes per iteration to keep
# overhead low.
# Closest known influences: Evolution strategies / (1+λ)-ES style adaptation
# with tournament-like selection among a small population; no gradients are
# used and everything is purely black-box.
# Novelty or unusual aspects: The code adapts its exploration rate and the
# population-related offspring count based on remaining evaluations, aiming
# to be robust and compact across dimensions.
# Failure modes: In very noisy or highly deceptive landscapes, the simple
# success-based sigma adaptation may converge prematurely; insufficient
# budget or extremely tight bounds can also limit progress. The clipping
# method can cause many points to lie on the boundary in such cases.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from func ----
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            # support common naming conventions: lb/ub
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            else:
                # fallback: try generic attributes
                lb = np.asarray(getattr(b, "lower", None), dtype=float)
                ub = np.asarray(getattr(b, "upper", None), dtype=float)
                if lb.size == 0 or ub.size == 0:
                    raise AttributeError("Cannot read bounds from func.bounds (expected lb/ub).")
        else:
            raise AttributeError("Cannot read bounds from func (expected lower/upper or bounds.lb/bounds.ub).")

        if lb.shape[0] != dim or ub.shape[0] != dim:
            lb = lb.reshape(-1)[:dim]
            ub = ub.reshape(-1)[:dim]

        # Ensure valid ordering and handle degenerate ranges
        lb, ub = np.minimum(lb, ub), np.maximum(lb, ub)
        span = ub - lb
        span = np.where(span > 0, span, 0.0)

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        evals = 0
        best_x = None
        best_y = np.inf

        # Evaluate helper that never exceeds budget.
        def eval_point(x):
            nonlocal evals, best_x, best_y
            if evals >= budget:
                return np.inf
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # If budget is 0, return something deterministic within bounds.
        if budget <= 0:
            mid = lb + 0.5 * (ub - lb)
            return clip(mid), float("inf")

        # ---- Initialization ----
        rng = np.random

        # Choose population/offspring sizes relative to dim and budget
        # Keep it small for compactness.
        pop_size = int(max(2, min(8, 2 + dim // 2)))
        # Initial samples: consume at most pop_size evaluations
        init_tries = min(pop_size, budget)

        # Starting sigma: a fraction of the overall span (robust to degenerate spans)
        total_span = float(np.max(span)) if dim > 0 else 0.0
        sigma = 0.25 * total_span if total_span > 0 else 1.0

        # If span is extremely small, sigma should be tiny to avoid pointless work
        sigma = max(1e-12, sigma)

        # Initialize population arrays
        X = np.empty((init_tries, dim), dtype=float)
        Y = np.empty(init_tries, dtype=float)
        for i in range(init_tries):
            # Uniform random point in bounds
            if np.all(span == 0):
                x = lb.copy()
            else:
                r = rng.random(dim)
                x = lb + r * span
            X[i] = x
            Y[i] = eval_point(x)

        # If we used fewer than budget, we still proceed with a dynamic loop.
        if best_x is None:
            # Shouldn't happen, but keep safe fallback
            idx = int(np.argmin(Y))
            best_x = X[idx].copy()
            best_y = float(Y[idx])

        # Keep a current population of size pop_size (or smaller if budget is tight)
        cur_pop = min(pop_size, budget - evals) if budget - evals > 0 else init_tries
        # Select the best individuals from X/Y to form population
        if init_tries > 0 and cur_pop < init_tries:
            order = np.argsort(Y)[:cur_pop]
            X = X[order]
            Y = Y[order]
        elif init_tries > 0:
            # If init_tries < cur_pop, we'll grow it gradually via offspring
            pass

        # ---- Main loop ----
        # Success-based adaptation thresholds
        success_count = 0
        prev_best_y = best_y

        # Exploration probability decreases with time.
        # Also adapt offspring batch size to remaining budget.
        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break

            t = evals / budget
            # Number of offspring evaluated this iteration
            # Keep it modest to avoid large unproductive batches near end.
            base_lambda = max(2, min(12, 2 + dim))
            lam = min(base_lambda, remaining)

            # Exploration probability: more early, less later
            # Ensure occasional exploration.
            p_explore = 0.35 * (1.0 - t) + 0.05
            p_explore = float(np.clip(p_explore, 0.05, 0.45))

            offspring = np.empty((lam, dim), dtype=float)
            off_y = np.empty(lam, dtype=float)

            # Generate offspring
            # With probability p_explore, sample uniformly; otherwise perturb best.
            # Use per-dimension noise scaled by relative span (helps across scales).
            denom = np.where(span > 0, span, 1.0)
            rel = denom / max(1.0, float(np.max(denom)))
            # If all spans are zero, perturbation collapses; rely on clipping.
            for j in range(lam):
                if rng.random() < p_explore:
                    if np.all(span == 0):
                        x = lb.copy()
                    else:
                        x = lb + rng.random(dim) * span
                else:
                    # Gaussian perturbation around best_x
                    # Scale noise so that typical step is around sigma * relative span.
                    z = rng.standard_normal(dim)
                    step_scale = sigma * rel
                    x = best_x + z * step_scale
                x = clip(x)
                offspring[j] = x
                off_y[j] = eval_point(x)
                if evals >= budget:
                    # Fill remaining (not evaluated) with inf and break cleanly
                    if j + 1 < lam:
                        off_y[j + 1 :] = np.inf
                    break

            # Determine success based on improvement in best_y
            if best_y < prev_best_y - 1e-15:
                success_count += 1
                prev_best_y = best_y
                # Shrink sigma to exploit when improvement occurs
                sigma *= 0.82
            else:
                # Slightly increase sigma if stuck to re-explore
                sigma *= 1.05
            # Keep sigma within a reasonable range relative to bounds
            if total_span > 0:
                sigma = float(np.clip(sigma, 1e-12, 0.5 * total_span))
            else:
                sigma = float(np.clip(sigma, 1e-12, 1.0))

            # ---- Selection / replacement ----
            # Merge current population (if any) with offspring and keep best pop_size.
            # Note: if we haven't built a full population yet, handle sizes safely.
            if X is None or Y is None:
                X = offspring
                Y = off_y
            else:
                # Determine how many current individuals we have
                cur_n = X.shape[0]
                # Keep only evaluated offspring (some may be inf if budget ran out)
                # But off_y already marks non-evaluated as inf, so sorting will drop them.
                X_all = np.vstack([X, offspring]) if cur_n > 0 else offspring
                Y_all = np.concatenate([Y, off_y]) if cur_n > 0 else off_y

                # Keep the best individuals for next iteration
                k = min(pop_size, X_all.shape[0])
                order = np.argsort(Y_all)[:k]
                X = X_all[order]
                Y = Y_all[order]

            # Optional mild diversity: if many individuals are identical to best,
            # increase sigma a bit (helps avoid collapse).
            if np.allclose(X, best_x, rtol=0, atol=1e-14):
                sigma *= 1.1

            # If no further evaluations possible, break
            if evals >= budget:
                break

        return best_x, best_y
