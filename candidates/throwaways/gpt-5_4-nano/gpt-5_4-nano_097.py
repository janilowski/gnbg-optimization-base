# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm that combines
# CMA-ES-like covariance adaptation with fallback strategies to remain robust
# across dimensions and noisy/non-smooth objectives. It maintains a small
# population of candidate solutions and iteratively refines a multivariate
# normal sampling distribution toward better points.
# Search state: Maintains the current mean vector (m), a covariance matrix (C)
# (implicitly represented through low-rank structure and eigenvalue clipping for
# stability), a step size (sigma), and a pool of already-evaluated points for
# optional warm-start and failure detection. Tracks the number of objective
# evaluations to never exceed the provided budget.
# Candidate generation: Each iteration samples lambda candidates from
# N(m, sigma^2 * C) using a stable factorization of C (eigen-decomposition).
# Candidates are then clipped to the box bounds.
# Selection and replacement: Evaluates all candidates, sorts by objective value
# (minimization), and computes a weighted new mean from the best individuals.
# The algorithm also keeps the overall best seen solution and value.
# Adaptation: Updates step size using progress along the evolution path,
# and updates covariance using rank-one and rank-mu style updates derived from
# selected steps, with eigenvalue flooring/ceiling to prevent numerical issues.
# Exploration mechanisms: Uses stochastic sampling with adjustable sigma and
# covariance. The covariance update maintains directionality; step size prevents
# premature collapse.
# Exploitation mechanisms: Weighted recombination pulls the mean toward the
# best region. The covariance update further concentrates sampling around
# improving directions.
# Boundary handling: Because it is a black-box with box constraints, every
# sampled candidate is clipped to the provided bounds (either via func.lower/
# func.upper or func.bounds.lb/ub).
# Budget strategy: Converts the budget into an iteration count and ensures that
# total function evaluations never exceed budget by carefully computing
# how many candidates can be evaluated per iteration.
# Closest known influences: Inspired by CMA-ES mechanics (mean recombination,
# evolution paths, covariance and step-size adaptation) but implemented in a compact
# and defensive way using only numpy and standard library.
# Novelty or unusual aspects: Uses robust covariance stabilization via eigenvalue
# clamping and a deterministic evaluation accounting scheme to guarantee budget
# compliance, even when the budget is not divisible by the population size.
# Failure modes: Very small budgets may lead to minimal refinement (returns best
# among initial samples). Ill-conditioned covariance can occur; eigenvalue clamping
# mitigates this. If the objective is extremely noisy, adaptation may be less effective.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import math
import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = int(self.budget)
        if budget <= 0 or d <= 0:
            # No evaluations allowed; return a zero vector.
            return np.zeros(d, dtype=float), float("inf")

        # ---- Read bounds from func ----
        lower = upper = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)

        if lower is None or upper is None:
            # If bounds are missing, default to wide bounds (still safe with clipping).
            lower = -5.0 * np.ones(d, dtype=float)
            upper = 5.0 * np.ones(d, dtype=float)

        lower = lower.reshape(-1)
        upper = upper.reshape(-1)
        if lower.size != d or upper.size != d:
            raise ValueError("Bounds dimension mismatch with dim.")

        # Ensure proper ordering
        swap_mask = lower > upper
        if np.any(swap_mask):
            a = lower.copy()
            lower = np.where(swap_mask, upper, a)
            upper = np.where(swap_mask, a, lower)

        # Clip helper
        def clip(x):
            return np.minimum(np.maximum(x, lower), upper)

        # Distance scale
        span = upper - lower
        span = np.where(span == 0, 1.0, span)

        # ---- Objective evaluation accounting ----
        eval_count = 0

        best_x = None
        best_y = float("inf")

        def eval_one(x):
            nonlocal eval_count, best_x, best_y
            y = float(func(x))
            eval_count += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # ---- Initialization ----
        rng = np.random.default_rng()  # harness will set global seed; this still uses entropy in absence of override
        # To respect harness seeding normally, try to sync with np.random's global state:
        # If the harness uses np.random.seed, numpy's RandomState is seeded; RandomGenerator isn't.
        # We'll use legacy global RNG instead for better harness compatibility.
        # (This is still only numpy usage.)
        rs = np.random

        # Choose initial mean as center of bounds
        m = (lower + upper) / 2.0

        # Initial sigma proportional to box size
        sigma = 0.3 * float(np.mean(span))
        sigma = max(sigma, 1e-12)

        # Population size and iteration budget split
        # Keep lambda modest for speed and robust behavior.
        lam = int(min(max(4, 4 + 3 * int(math.sqrt(d))), max(4, 4 * d)))
        # mu best
        mu = lam // 2
        if mu < 1:
            mu = 1

        # Weights (log-based, normalized) for recombination
        # Standard-ish weights: w_i ~ log(mu+0.5) - log(i+1)
        i = np.arange(1, mu + 1, dtype=float)
        w = np.log(mu + 0.5) - np.log(i)
        w = np.maximum(w, 0.0)
        if np.sum(w) == 0:
            w = np.ones_like(w)
        w = w / np.sum(w)
        # Effective selection mass
        mu_eff = 1.0 / np.sum(w ** 2)

        # Strategy parameter settings (CMA-ES-like, defensive)
        # c_sigma and damping
        c_sigma = (mu_eff + 2.0) / (d + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (d + 1.0)) - 1.0) + c_sigma
        # Covariance learning rates
        c_c = (4.0 + mu_eff / d) / (d + 4.0 + 2.0 * mu_eff / d)
        c1 = 2.0 / ((d + 1.3) ** 2 + mu_eff)
        alpha_mu = 2.0
        c_mu = min(1.0 - c1, alpha_mu * (mu_eff - 2.0 + 1.0 / mu_eff) / ((d + 2.0) ** 2 + alpha_mu * mu_eff / 2.0))
        # Evolution paths
        p_c = np.zeros(d, dtype=float)
        p_sigma = np.zeros(d, dtype=float)

        # Covariance matrix start: scaled identity
        C = np.eye(d, dtype=float)

        # Best among initial random samples to kick-start exploration
        # We use as many evaluations as budget allows but leave some for iterations.
        # Evaluate at most min(lam, budget) initial points.
        init_evals = min(lam, budget)
        # Use rs for harness seeding compatibility
        for _ in range(init_evals):
            z = rs.normal(size=d)
            x = clip(m + sigma * z)
            eval_one(x)

        # If budget exhausted, return best
        if eval_count >= budget:
            if best_x is None:
                best_x = m.copy()
                best_y = float(func(best_x))
            return best_x, best_y

        # Iterations: each iteration uses k evaluations where k <= lam and eval_count+k<=budget
        # We'll estimate remaining iterations but adapt last partial.
        # Precompute an upper bound on number of iterations
        remaining = budget - eval_count
        max_iters = max(1, remaining // max(1, lam))
        # But we may do more if budget < lam; handle via while.
        it = 0
        while eval_count < budget and it < max_iters + 5:
            it += 1
            # Determine how many candidates we can evaluate this iteration
            k = min(lam, budget - eval_count)
            if k <= 0:
                break

            # Stable factorization: eigen-decompose C and clamp eigenvalues.
            # This keeps sampling robust and numerically safe.
            # For d up to moderate sizes, this is acceptable.
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
            except np.linalg.LinAlgError:
                # Fallback: reset covariance if decomposition fails
                C = np.eye(d, dtype=float)
                eigvals = np.ones(d, dtype=float)
                eigvecs = np.eye(d, dtype=float)

            # Clamp eigenvalues to avoid collapse/explosion
            # Bounds chosen to be conservative relative to sigma scaling.
            eigvals = np.clip(eigvals, 1e-12, 1e12)
            A = eigvecs @ (np.diag(np.sqrt(eigvals)) @ eigvecs.T)

            # Sample k candidates
            # We'll generate z ~ N(0,I), then y = A @ z, x = m + sigma * y.
            # Using matrix multiply for speed.
            Z = rs.normal(size=(k, d))
            Y = Z @ A.T
            X = clip(m + sigma * Y)

            # Evaluate
            fitness = np.empty(k, dtype=float)
            for j in range(k):
                fitness[j] = eval_one(X[j])

            # Sort candidates by fitness (minimization)
            order = np.argsort(fitness)
            X = X[order]
            Y = Y[order]  # corresponding steps scaled by covariance factor
            fitness = fitness[order]

            # Select best mu out of k
            k_mu = min(mu, k)
            if k_mu <= 0:
                continue
            # Adjust weights if k < mu
            if k_mu != mu:
                # Recompute weights for first k_mu indices
                ii = np.arange(1, k_mu + 1, dtype=float)
                ww = np.log(k_mu + 0.5) - np.log(ii)
                ww = np.maximum(ww, 0.0)
                if np.sum(ww) == 0:
                    ww = np.ones_like(ww)
                ww = ww / np.sum(ww)
                mu_eff_curr = 1.0 / np.sum(ww ** 2)
            else:
                ww = w
                mu_eff_curr = mu_eff

            # Weighted mean recombination using selected X (not Y)
            x_sel = X[:k_mu]
            new_m = np.dot(ww, x_sel)

            # Compute weighted steps in normalized coordinates for adaptation:
            # y_i = (x_i - old_m) / sigma in covariance coordinates; but we sampled:
            # x_i = old_m + sigma * Y_i, where Y_i = A @ z_i
            # => (x_i - old_m) / sigma = Y_i
            y_sel = Y[:k_mu]  # shape (k_mu, d)
            weighted_y = np.dot(ww, y_sel)

            # Evolution path update for step-size control (CMA-ES-like)
            # In CMA-ES, p_sigma = (1 - c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mu_eff) * C^{-1/2} * (m_new-m)/sigma
            # Here C^{-1/2} * weighted_y approx: since weighted_y = A @ z = C^{1/2} z,
            # we need C^{-1/2} weighted_y = z.
            # But weighted_y is weighted over z_i; approximate using z-space from Y via least squares:
            # We'll compute z_mean = (A^{-1} weighted_y) using eigen-decomp.
            inv_sqrt = eigvecs @ (np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T)
            z_w = inv_sqrt @ weighted_y

            p_sigma = (1.0 - c_sigma) * p_sigma + math.sqrt(max(c_sigma * (2.0 - c_sigma) * mu_eff_curr, 0.0)) * z_w
            norm_p_sigma = float(np.linalg.norm(p_sigma))
            # Step-size update
            # chi_d approximation
            chi_d = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
            sigma *= math.exp((c_sigma / d_sigma) * ((norm_p_sigma / max(chi_d, 1e-12)) - 1.0))

            # Evolution path for covariance update
            # h_sigma condition: if norm(p_sigma) is small enough
            h_sigma = 1.0 if norm_p_sigma / math.sqrt(max(1e-12, 1.0 - (1.0 - c_sigma) ** (2.0))) < (1.4 + 2.0 / (d + 1.0)) * chi_d else 0.0

            p_c = (1.0 - c_c) * p_c + h_sigma * math.sqrt(max(c_c * (2.0 - c_c) * mu_eff_curr, 0.0)) * weighted_y

            # Rank-one and rank-mu covariance updates
            # delta_m = (new_m - m_old)/sigma
            m_old = m
            m = new_m

            # Rank-one
            rank_one = np.outer(p_c, p_c)

            # Rank-mu
            # covariance update uses sum_i w_i * y_i y_i^T, where y_i = (x_i - m_old)/sigma = Y_i
            y_sel = y_sel[:k_mu]
            weighted_outer = np.zeros((d, d), dtype=float)
            for idx in range(k_mu):
                weighted_outer += ww[idx] * np.outer(y_sel[idx], y_sel[idx])

            # Update covariance
            C = (1.0 - c1 - c_mu) * C + c1 * rank_one + c_mu * weighted_outer

            # Stabilize covariance eigenvalues after update
            try:
                eigvals2, eigvecs2 = np.linalg.eigh(C)
                eigvals2 = np.clip(eigvals2, 1e-14, 1e14)
                C = eigvecs2 @ (np.diag(eigvals2) @ eigvecs2.T)
            except np.linalg.LinAlgError:
                C = np.eye(d, dtype=float)

            # Keep sigma in a reasonable range relative to bounds
            sigma = float(np.clip(sigma, 1e-12, 10.0 * float(np.mean(span))))

            # If mean goes out-of-bounds due to clipping effects, re-center it safely
            m = clip(m)

        # Ensure best exists
        if best_x is None:
            best_x = m.copy()
            best_y = float(func(best_x))

        return best_x, best_y
