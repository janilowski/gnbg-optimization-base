# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# (a CMA-ES-inspired evolution strategy with simple step-size adaptation) that
# is robust across dimensions while honoring a fixed evaluation budget.
#
# Search state: Maintains a current mean vector (mu) and a global step-size
# (sigma). Also tracks the best-so-far solution found during the run.
#
# Candidate generation: Each generation samples offspring from a Gaussian
# distribution around mu using an isotropic covariance (sigma * N(0, I)).
# It uses symmetric pairing of random directions to reduce variance.
#
# Selection and replacement: Evaluates all offspring in a generation, then
# selects the best subset (by objective value) and recombines them via a
# weighted average to update the mean toward lower objective values.
#
# Adaptation: Adapts sigma using a path-like mechanism based on whether the
# new mean improved relative to previous progress (a simplified form of
# CMA-ES step-size control).
#
# Exploration mechanisms: Random sampling with time-varying sigma provides
# exploration; symmetric directions provide diversity with reduced noise.
#
# Exploitation mechanisms: Selection pressure and weighted recombination
# concentrate samples around promising regions.
#
# Boundary handling: Candidate points are clipped to the provided bounds
# before evaluation, ensuring feasibility.
#
# Budget strategy: The total number of objective evaluations is capped to the
# provided budget. The number of generations and offspring per generation is
# chosen accordingly.
#
# Closest known influences: This design is inspired by evolution strategies and
# CMA-ES-style sigma adaptation, but intentionally kept lightweight by using an
# isotropic covariance (no expensive covariance matrix updates).
#
# Novelty or unusual aspects: Uses symmetric direction sampling to improve
# robustness for small budgets/dimensions while staying compact and standard-
# library-only.
#
# Failure modes: If the objective is extremely noisy or bounds are very tight,
# clipping can distort gradients and slow progress. With very small budgets,
# the algorithm may only perform one or two generations, limiting adaptation.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def _get_bounds(self, func):
        # Priority: func.lower/func.upper, else func.bounds.lb/func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            b = getattr(func, "bounds", None)
            if b is None or not (hasattr(b, "lb") and hasattr(b, "ub")):
                raise AttributeError(
                    "Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
                )
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)

        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            # Try to reshape if possible
            lb = np.asarray(lb).reshape(self.dim)
            ub = np.asarray(ub).reshape(self.dim)
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        # Ensure valid ordering
        swap = lb > ub
        if np.any(swap):
            lb2 = lb.copy()
            ub2 = ub.copy()
            lb2[swap], ub2[swap] = ub2[swap], lb2[swap]
            lb, ub = lb2, ub2

        return lb, ub

    def __call__(self, func):
        lb, ub = self._get_bounds(func)

        # Random initialization around the center with mild spread
        center = (lb + ub) / 2.0
        span = (ub - lb)
        # Prevent sigma from becoming 0 if bounds coincide
        base_scale = np.maximum(np.max(span), 1e-12)
        sigma = 0.25 * base_scale / max(1.0, np.sqrt(self.dim))

        # Mean: start near center with small random perturbation if possible
        mu = center + np.random.uniform(-0.1, 0.1, size=self.dim) * span
        mu = np.clip(mu, lb, ub)

        def eval_point(x, count):
            # Returns (y, new_count)
            y = float(func(x))
            count += 1
            return y, count

        # Budget management
        max_evals = max(1, self.budget)
        evals = 0

        # Evaluate initial point
        best_x = mu.copy()
        best_y, evals = eval_point(best_x, evals)

        # Early exit if budget exhausted
        if evals >= max_evals:
            return best_x, best_y

        # Determine offspring per generation. Keep small for low budgets.
        # Aim for 2-12 generations; otherwise reduce offspring size.
        # Must respect budget: generations * (lambda) <= remaining evals.
        remaining = max_evals - evals
        # Choose lambda based on dimension, but clamp by budget.
        lam = int(np.clip(4 + 2 * self.dim, 4, 64))
        lam = max(2, lam)
        # Use symmetric sampling => effective directions count is lam/2 if lam even.
        # We'll keep lam even for clean pairing.
        if lam % 2 == 1:
            lam += 1
        lam = min(lam, remaining)  # cannot exceed remaining evals
        if lam < 2:
            lam = 2 if remaining >= 2 else remaining
        lam = int(lam)
        if lam % 2 == 1 and lam >= 2:
            lam -= 1
        if lam < 2:
            # Handle extremely small remaining budgets
            y, evals = eval_point(np.clip(mu + sigma * np.random.randn(self.dim), lb, ub), evals)
            if y < best_y:
                best_y = y
                best_x = np.clip(mu + sigma * np.random.randn(self.dim), lb, ub)
            return best_x, best_y

        # Generations based on remaining evals
        gens = remaining // lam
        if gens < 1:
            # Not enough for one full generation: do one batch of size remaining
            lam = remaining
            gens = 1

        # Step-size adaptation parameters (lightweight)
        # Typical CMA-ES: c_sigma and damping, but simplified.
        c_sigma = 0.3
        d_sigma = 1.0 / np.sqrt(self.dim)
        p_sigma = 0.0  # scalar progress indicator
        prev_best_y = best_y

        # Selection: top mu_sel among lambda
        mu_sel = max(2, lam // 2)
        # Recombination weights (linear, normalized)
        w = np.log(mu_sel + 0.5) - np.log(np.arange(1, mu_sel + 1))
        w = w / np.sum(w)

        for _ in range(gens):
            # Stop if budget reached
            if evals >= max_evals:
                break

            # If last generation would exceed budget, reduce lambda
            remaining = max_evals - evals
            if lam > remaining:
                # Recreate an even-ish lambda; but allow odd by falling back to non-paired samples.
                lam_eff = int(remaining)
            else:
                lam_eff = lam

            # Candidate generation with symmetric pairing
            # We generate k pairs + optional extra sample if lam_eff is odd.
            xs = []
            Zs = []  # store direction samples (for progress)
            k = lam_eff // 2
            for _k in range(k):
                z = np.random.randn(self.dim)
                # Normalize for scale stability when sigma is large relative to bounds
                # (does not change isotropy, but improves numerical behavior)
                nz = np.linalg.norm(z)
                if nz > 0:
                    z = z / nz
                # Two symmetric points
                x1 = np.clip(mu + sigma * z * base_scale / max(1e-12, np.sqrt(self.dim)), lb, ub)
                x2 = np.clip(mu - sigma * z * base_scale / max(1e-12, np.sqrt(self.dim)), lb, ub)
                xs.append(x1)
                xs.append(x2)
                Zs.append(z)
            if lam_eff % 2 == 1:
                z = np.random.randn(self.dim)
                nz = np.linalg.norm(z)
                if nz > 0:
                    z = z / nz
                x = np.clip(mu + sigma * z * base_scale / max(1e-12, np.sqrt(self.dim)), lb, ub)
                xs.append(x)

            # Evaluate candidates
            ys = np.empty(len(xs), dtype=float)
            for i, x in enumerate(xs):
                y, evals = eval_point(x, evals)
                ys[i] = y
                if evals >= max_evals:
                    # Trim remaining evaluations if budget runs out mid-generation
                    ys = ys[: i + 1]
                    xs = xs[: i + 1]
                    break

            # Update best-so-far
            min_idx = int(np.argmin(ys))
            if ys[min_idx] < best_y:
                best_y = float(ys[min_idx])
                best_x = np.array(xs[min_idx], copy=True)

            # If not enough points evaluated, stop
            if len(ys) < 2:
                break

            # Sort by fitness (minimization)
            order = np.argsort(ys)
            xs_sorted = [xs[i] for i in order]
            ys_sorted = ys[order]

            # Select elites
            m = min(mu_sel, len(xs_sorted))
            elite = xs_sorted[:m]
            elite = np.asarray(elite, dtype=float)

            # Weighted recombination
            # Mean update: mu <- sum_i w_i * elite_i, using w truncated if m < mu_sel
            if m != mu_sel:
                w_use = np.log(m + 0.5) - np.log(np.arange(1, m + 1))
                w_use = w_use / np.sum(w_use)
            else:
                w_use = w

            mu_new = np.sum(elite * w_use[:, None], axis=0)

            # Step-size adaptation:
            # If we improved vs previous best, increase exploitation (decrease sigma slightly).
            # Otherwise, increase exploration (increase sigma).
            improved = best_y < prev_best_y - 1e-15
            if improved:
                p_sigma = (1 - c_sigma) * p_sigma - c_sigma * 1.0
            else:
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * 1.0
            prev_best_y = best_y

            # Update sigma multiplicatively (CMA-like)
            # Clamp to avoid degeneracy
            sigma *= np.exp((p_sigma * d_sigma))
            sigma = float(np.clip(sigma, 1e-12, 10.0 * base_scale))

            # Additionally, nudge sigma down if elites are clustered (small progress)
            # This is a small heuristic to stabilize on flat functions.
            step_norm = np.linalg.norm(mu_new - mu)
            if step_norm < 1e-12:
                sigma *= 0.99
            else:
                # If mean jump is large relative to sigma, shrink a bit for stability
                if step_norm > 2.0 * sigma * base_scale / max(1e-12, np.sqrt(self.dim)):
                    sigma *= 0.9

            mu = mu_new

        return best_x, best_y
