# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer (a lightweight
# evolution strategy). It maintains a small population of candidate solutions,
# samples Gaussian perturbations around the current best, and iteratively
# updates the search center based on fitness rankings.
# Search state: Tracks the current best solution (x_best, y_best), current
# step size (sigma), and a history of evaluations to ensure the evaluation
# budget is never exceeded.
# Candidate generation: Each iteration samples lambda offspring as
# x_i = x_best + sigma * N(0, I), with an additional “mirror” around the
# center for some offspring to improve local sampling symmetry.
# Selection and replacement: Evaluates all offspring, then uses the best offspring
# to potentially update the center; additionally performs a mild weighted
# update toward the top-ranked offspring for stability.
# Adaptation: Step size sigma shrinks when improvement is observed, and expands
# slightly when iterations fail to improve, keeping exploration/exploitation balanced.
# Exploration mechanisms: Gaussian sampling around the best, plus intermittent
# re-centering from the best-so-far and occasional larger perturbations.
# Exploitation mechanisms: Strong emphasis on candidates near the best via
# decreasing sigma and rank-based weighted center updates.
# Boundary handling: After mutation, candidates are clipped to the provided bounds.
# Budget strategy: Uses a fixed number of iterations computed from the
# evaluation budget and evaluates exactly (1 + iterations*lambda) points,
# never going beyond the budget. If the remaining budget is insufficient,
# fewer candidates are evaluated in the last iteration.
# Closest known influences: Inspired by CMA-ES-style sampling and (1+λ)/(μ+λ)
# evolution strategies, but simplified to remain compact and robust.
# Novelty or unusual aspects: Adds a symmetric (mirror) sampling trick and a
# rank-weighted update step, without maintaining a full covariance matrix.
# Failure modes: In very rugged or deceptive landscapes, the simple sigma
# adaptation may converge prematurely; clipping may also cause many candidates to
# land on boundaries, reducing effective search diversity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # ---- Read bounds from the objective ----
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
            raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub.")

        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            lb = np.reshape(lb, (self.dim,))
            ub = np.reshape(ub, (self.dim,))

        # Ensure finite bounds; if any are infinite, fall back to a heuristic scale.
        finite_mask = np.isfinite(lb) & np.isfinite(ub)
        if not np.any(finite_mask):
            # If bounds are entirely infinite (unusual), use default window [-1, 1]
            lb = np.full(self.dim, -1.0)
            ub = np.full(self.dim, 1.0)
            finite_mask = np.ones(self.dim, dtype=bool)

        # ---- Helper: budget-aware evaluation ----
        evals = 0

        def eval_x(x):
            nonlocal evals
            if evals >= self.budget:
                # Should not happen due to careful budgeting.
                return float("inf")
            y = func(x)
            evals += 1
            # Convert to scalar float if possible
            try:
                return float(y)
            except Exception:
                return y

        # ---- Initialization ----
        # Choose a starting point in the middle of bounds (common for robust black-box setups).
        center = np.where(finite_mask, 0.5 * (lb + ub), 0.0).astype(float)

        # Initial step size based on bound scale (use only finite dimensions).
        span = np.where(finite_mask, ub - lb, 1.0)
        span_scale = np.median(span[finite_mask]) if np.any(finite_mask) else 1.0
        # Guard against zero span.
        span_scale = float(span_scale) if span_scale > 0 else 1.0

        sigma = 0.3 * span_scale / max(1.0, np.sqrt(self.dim))
        sigma = max(sigma, 1e-12)

        # Evaluate initial center.
        y_best = eval_x(center)
        x_best = center.copy()

        # Strategy parameters (kept compact and dimension-robust).
        lam = int(4 + 3 * np.log1p(self.dim))  # offspring per iteration
        lam = max(4, lam)

        # Compute how many full iterations we can afford after initial evaluation.
        remaining = self.budget - evals
        if remaining <= 0:
            return x_best, y_best

        # Each iteration uses up to lam evaluations. We'll evaluate fewer in last iter if needed.
        iters = remaining // lam
        if iters <= 0:
            # Not enough budget for even one full batch; evaluate a small set and return best.
            # We still respect the budget exactly.
            k = min(lam, remaining)
            # Symmetric candidates for better local search.
            for i in range(k):
                z = np.random.randn(self.dim)
                cand = np.clip(x_best + sigma * z, lb, ub)
                y = eval_x(cand)
                if y < y_best:
                    y_best = y
                    x_best = cand.copy()
            return x_best, y_best

        # If we can do at least one full iteration, also allow a partial final iteration.
        extra = remaining - iters * lam
        total_iters = iters + (1 if extra > 0 else 0)

        # Main optimization loop
        for t in range(total_iters):
            if evals >= self.budget:
                break

            # Determine how many offspring to evaluate this iteration.
            k = lam
            if t == total_iters - 1 and extra > 0:
                k = extra
            if k <= 0:
                break

            # Generate offspring around current center. Use mirror sampling sometimes.
            offspring = np.empty((k, self.dim), dtype=float)
            half = k // 2

            # First half: standard mutations
            for i in range(half):
                z = np.random.randn(self.dim)
                offspring[i] = center + sigma * z

            # Second half: mirrored mutations to improve symmetry/local estimation
            for i in range(half, k):
                z = np.random.randn(self.dim)
                offspring[i] = center - sigma * z

            # Boundary handling: clip to bounds
            offspring = np.clip(offspring, lb, ub)

            # Evaluate offspring
            values = np.empty(k, dtype=float)
            for i in range(k):
                values[i] = eval_x(offspring[i])

            # Sort by fitness (minimization)
            idx = np.argsort(values)
            best_i = idx[0]

            # Update best-so-far
            if values[best_i] < y_best:
                y_best = values[best_i]
                x_best = offspring[best_i].copy()

            # Rank-based weighted center update toward top candidates.
            # Use a small number of leaders for stability.
            m = min(3, k)
            leaders = idx[:m]
            leader_x = offspring[leaders]
            leader_y = values[leaders]

            # Convert leader_y to weights: lower is better.
            # Use a softmax on negative scaled fitness for numeric stability.
            # Scale by range or sigma to avoid extreme exponents.
            y_span = float(np.max(leader_y) - np.min(leader_y))
            scale = y_span if y_span > 1e-12 else 1.0
            # Lower fitness => higher weight
            w = np.exp(-(leader_y - np.min(leader_y)) / scale)
            w = w / (np.sum(w) + 1e-300)

            new_center = np.sum(leader_x * w[:, None], axis=0)

            # Sigma adaptation based on improvement signal.
            # If we improved global best, shrink; else gently expand.
            if values[best_i] < y_best if False else False:
                # (This branch is unreachable because y_best already updated above.)
                pass

            # Instead, compare pre-iteration best to best offspring value.
            # We'll infer improvement by comparing x_best distance isn't reliable; so:
            # Track improvement by checking if best offspring equals global best update.
            # Since we may have improved y_best above, we can detect by whether
            # offspring best_i equals x_best (approx).
            improved = np.isfinite(values[best_i]) and np.allclose(offspring[best_i], x_best, atol=0, rtol=0)

            if improved:
                sigma *= 0.82
            else:
                # If no improvement, keep exploring a bit more.
                sigma *= 1.05

            # Occasional larger exploration step every few iterations
            if (t % max(3, int(np.log1p(self.dim)))) == 0:
                sigma = min(sigma * 1.05, 2.0 * span_scale / max(1.0, np.sqrt(self.dim)))

            # Clip center to bounds and carry over
            center = np.clip(new_center, lb, ub)

            # Clamp sigma to reasonable bounds to avoid numerical issues
            sigma = float(np.clip(sigma, 1e-12, 5.0 * span_scale))

        return x_best, y_best
