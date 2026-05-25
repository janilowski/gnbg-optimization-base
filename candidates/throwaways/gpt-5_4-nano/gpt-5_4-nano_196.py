# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm combining
# randomized population search with local refinement from the best points.
# The method is derivative-free, works for any dimension, and respects a
# strict evaluation budget.
#
# Search state: Maintains a current population of candidate solutions and
# tracks the best solution/value found so far. It also tracks the remaining
# number of evaluations to ensure the budget is never exceeded.
#
# Candidate generation: Each iteration creates new candidates by sampling
# Gaussian perturbations around elite points, with step sizes that shrink
# as the budget is consumed. A small additional uniform “jitter” is added
# early on for robustness.
#
# Selection and replacement: The new candidates are evaluated; the best
# candidates replace the worst ones (elitist replacement). The global best
# is updated whenever a better objective value is found.
#
# Adaptation: The global step size is adapted based on progress (improvement
# rate). If improvement stalls, the algorithm reduces step size less
# aggressively and keeps some exploration; if it improves, it gradually
# tightens.
#
# Exploration mechanisms: Early-stage larger perturbations and uniform jitter
# across the domain encourage broad search and help avoid poor starting points.
#
# Exploitation mechanisms: Sampling around the current best (and several elites)
# supports local refinement. The algorithm repeatedly re-centers perturbations
# at the best-so-far.
#
# Boundary handling: Candidates are clipped to the provided bounds after each
# perturbation. This keeps solutions feasible even if sampling goes out of
# range.
#
# Budget strategy: The algorithm uses the budget parameter as an evaluation
# limit. It allocates an initial batch of evaluations, then runs a loop where
# each iteration evaluates a fixed number of candidates, capped so the total
# evaluations never exceed the provided budget.
#
# Closest known influences: Inspired by simple evolutionary strategies and
# CMA-ES-like intuition (elite-guided Gaussian sampling with step-size control),
# but implemented in a minimal, budget-aware form without covariance matrices.
#
# Novelty or unusual aspects: Uses a deterministic allocation of evaluation
# batches and a small “best-of-elites” local search that activates more
# strongly near the end of the budget, improving robustness with little code.
#
# Failure modes: On extremely ill-conditioned functions or very small budgets,
# the random search component may dominate and local refinement may have too
# few evaluations to converge.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        n = self.dim
        lb, ub = self._get_bounds(func, n)
        lb = lb.astype(float)
        ub = ub.astype(float)
        span = ub - lb

        # If bounds are degenerate, return a feasible point (clipped) and evaluate once.
        # This ensures we never exceed budget and handle weird inputs robustly.
        evals_used = 0

        def eval_x(x):
            nonlocal evals_used
            if evals_used >= self.budget:
                # Should not happen if budget logic is correct.
                return float("inf")
            y = func(x)
            evals_used += 1
            return float(y)

        # Choose initial step scale relative to bounds.
        # If span is zero in some dimensions, perturbations there have no effect.
        # This keeps behavior stable across different scaling.
        base_sigma = 0.2 * np.maximum(span, 1e-12)
        base_sigma = np.where(span == 0, 0.0, base_sigma)

        # Heuristic population/iteration planning.
        # Keep it compact but ensure meaningful search in moderate dimensions.
        # If budget is tiny, we degrade gracefully to random sampling.
        B = max(1, self.budget)
        d = max(1, n)

        # Population size: small for high dims or tiny budgets
        pop = int(np.clip(8 + d // 2, 8, 48))
        pop = min(pop, B)

        # Initial candidates: uniform over the domain (feasible by construction).
        X = lb + np.random.rand(pop, n) * span
        Y = np.empty(pop, dtype=float)

        # Evaluate initial population (cap to budget).
        m = min(pop, B)
        for i in range(m):
            Y[i] = eval_x(X[i])
        if m < pop:
            # In case budget < pop (only possible when B < pop), truncate.
            X = X[:m]
            Y = Y[:m]
            pop = m

        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])

        # Elite set size for guiding sampling
        elite_k = int(np.clip(3 + d // 5, 3, min(10, pop)))
        elite_k = min(elite_k, pop)

        # Budget-aware loop. Each iteration evaluates up to eval_batch candidates.
        # Choose a batch size that amortizes overhead but respects budget.
        eval_batch = int(np.clip(pop, 4, 32))
        eval_batch = min(eval_batch, B - evals_used) if B - evals_used > 0 else 0

        # If no remaining evaluations, return.
        if eval_batch <= 0:
            return best_x, best_y

        # Track improvement for step adaptation.
        prev_best = best_y
        stagnation = 0

        # Determine number of iterations roughly.
        # We'll also cap each loop by remaining evals.
        while evals_used < B:
            remaining = B - evals_used
            k = min(eval_batch, remaining)
            if k <= 0:
                break

            # Sort current population by fitness (lower is better)
            order = np.argsort(Y)
            elites = X[order[:elite_k]]

            # Step size schedule: shrink as budget is consumed, but with adaptation.
            t = evals_used / B
            # Shrink factor from early exploration to late exploitation
            shrink = (1.0 - t) ** 1.5
            # If late in budget, strengthen exploitation around best
            late_boost = 0.5 + 0.5 * (t > 0.7)

            # Adapt sigma based on recent progress
            current_best = best_y
            improved = current_best < prev_best - 1e-12
            if improved:
                prev_best = current_best
                stagnation = 0
            else:
                stagnation += 1

            # Reduce shrink more slowly if stagnating
            adapt = 1.0 + 0.15 * min(5, stagnation)
            sigma = base_sigma * shrink * adapt * late_boost

            # Generate candidates around elites (Gaussian perturbations)
            # We pick elite centers probabilistically: higher fitness gets higher probability.
            # To keep deterministic under seeded numpy, use pure numpy randomness.
            elite_scores = np.max(Y) - Y[order[:elite_k]]  # larger = better
            # Avoid all-zero probability issues
            probs = elite_scores - elite_scores.min()
            if np.all(probs <= 0):
                p = np.ones(elite_k, dtype=float) / elite_k
            else:
                p = probs / np.sum(probs)

            # Draw centers
            center_idx = np.random.choice(elite_k, size=k, p=p)
            centers = elites[center_idx]  # shape (k, n)

            # Two-scale perturbations: one main gaussian, one occasional jitter early.
            Z = np.random.randn(k, n)
            X_new = centers + Z * sigma

            # Early uniform jitter to diversify if t is small
            if t < 0.35:
                jitter_scale = 0.15 * span * (1.0 - t)
                U = (np.random.rand(k, n) * 2.0 - 1.0) * jitter_scale
                X_new = X_new + U

            # Boundary handling: clip to [lb, ub]
            X_new = np.minimum(np.maximum(X_new, lb), ub)

            # Evaluate
            Y_new = np.empty(k, dtype=float)
            for i in range(k):
                Y_new[i] = eval_x(X_new[i])

            # Update global best
            j = int(np.argmin(Y_new))
            if Y_new[j] < best_y:
                best_y = float(Y_new[j])
                best_x = X_new[j].copy()

            # Elitist replacement into a maintained population:
            # combine and keep best pop points to continue guided search.
            X_comb = np.vstack((X, X_new))
            Y_comb = np.concatenate((Y, Y_new))
            ord = np.argsort(Y_comb)
            # Keep up to pop points (never empty)
            keep = min(pop, X_comb.shape[0])
            X = X_comb[ord[:keep]]
            Y = Y_comb[ord[:keep]]

            # Refresh elite_k if pop changed (unlikely)
            elite_k = int(np.clip(3 + d // 5, 3, min(10, keep)))
            elite_k = min(elite_k, keep)

            # If remaining is tiny, do a short targeted refinement around best.
            # This activates naturally as t grows due to shrink/late_boost,
            # but this makes the last part more decisive.
            if (B - evals_used) <= max(0, d // 2) and (B - evals_used) > 0:
                # We'll spend the rest by sampling tightly around best_x.
                rem = B - evals_used
                if rem > 0:
                    # Tight local sigma
                    local_sigma = np.where(span == 0, 0.0, 0.05 * span * (1.0 - t + 1e-9))
                    # Choose a few directions (Gaussian) centered at best.
                    Zl = np.random.randn(rem, n)
                    X_local = best_x + Zl * local_sigma
                    X_local = np.minimum(np.maximum(X_local, lb), ub)
                    for i in range(rem):
                        yi = eval_x(X_local[i])
                        if yi < best_y:
                            best_y = float(yi)
                            best_x = X_local[i].copy()
                break

        return best_x, best_y

    @staticmethod
    def _get_bounds(func, dim: int):
        # Bounds sources:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            # Support both .lb/.ub and tuple-like fallbacks
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            elif isinstance(b, (tuple, list)) and len(b) == 2:
                lb = np.asarray(b[0], dtype=float)
                ub = np.asarray(b[1], dtype=float)
            else:
                raise AttributeError("func.bounds must provide lb/ub or be a (lb, ub) pair.")
        else:
            raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub.")

        # Broadcast scalar bounds if provided
        if lb.ndim == 0:
            lb = np.full(dim, float(lb))
        if ub.ndim == 0:
            ub = np.full(dim, float(ub))

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)

        if lb.size != dim or ub.size != dim:
            raise ValueError(f"Bounds size mismatch: expected dim={dim}, got lb={lb.size}, ub={ub.size}.")
        if np.any(ub < lb):
            raise ValueError("Upper bound must be >= lower bound elementwise.")
        return lb, ub
