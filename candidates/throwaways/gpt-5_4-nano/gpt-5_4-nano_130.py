# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm using
# a randomized, budget-aware evolutionary strategy (ES) with coordinate-wise
# mutation adaptation. It maintains a small population, repeatedly samples candidates
# around the best-so-far solution, and shrinks/expands step sizes based on whether
# improvements occur. The implementation is robust across dimensions and only uses
# numpy + the standard library.
#
# Search state: The algorithm tracks an evaluation budget, number of evaluations used,
# best-so-far point best_x and value best_y, and per-coordinate mutation scales sigma.
# It also stores a small candidate population each iteration.
#
# Candidate generation: Each iteration creates offspring by taking the current best_x
# and adding Gaussian noise scaled by sigma. Additionally, it generates a few
# candidates via simple "crossover-like" averaging between best_x and a random
# population member to diversify directions.
#
# Selection and replacement: Offspring are evaluated (never exceeding the remaining budget).
# The next generation best is chosen as the best among evaluated offspring and the
# current best; the search then keeps best_x as the anchor for subsequent sampling.
#
# Adaptation: Sigma is adapted per coordinate: when an improvement occurs, sigma is
# slightly decreased to focus exploitation; when no improvement occurs, sigma is
# increased to encourage exploration. The adaptation uses simple multiplicative factors.
#
# Exploration mechanisms: Random Gaussian sampling around best_x, plus occasional
# averaged candidates that can move the search into new regions without relying only
# on pure perturbations.
#
# Exploitation mechanisms: The algorithm anchors mutations to the current best_x and
# reduces sigma upon improvements, making it increasingly local near good solutions.
#
# Boundary handling: After sampling, candidate points are clipped to the provided
# bounds (read from func.lower/upper or func.bounds.lb/ub). This guarantees feasibility.
#
# Budget strategy: The algorithm estimates an iteration count and offspring per iteration
# from the total budget and stops immediately when the budget would be exceeded.
# Every evaluation is accounted for; it never calls the objective after budget is used.
#
# Closest known influences: A lightweight (1+λ)/(μ+λ)-style evolution strategy with
# step-size control inspired by success-based adaptation, but simplified for
# robustness and compactness.
#
# Novelty or unusual aspects: Per-coordinate sigma adaptation combined with a small
# fraction of averaged-offspring candidates for added diversity while keeping the
# anchor at best_x. This helps in both separable and non-separable landscapes.
#
# Failure modes: If the objective is very noisy or has extremely tight/degenerate bounds,
# clipping may cause many candidates to collapse to boundaries, slowing progress.
# Also, with extremely small budgets, the algorithm may only sample a handful of points.
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
            # Degenerate case: no evaluations allowed. Return a point in the middle if possible.
            lb, ub = self._read_bounds(func)
            x = (lb + ub) / 2.0
            return x, float("inf")

        lb, ub = self._read_bounds(func)
        lb = np.asarray(lb, dtype=float).reshape(dim)
        ub = np.asarray(ub, dtype=float).reshape(dim)

        # Initial point: center of bounds.
        best_x = (lb + ub) / 2.0
        best_y = None

        evals = 0

        def eval_at(x):
            nonlocal evals, best_x, best_y
            if evals >= budget:
                return None
            y = float(func(x))
            evals += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = np.asarray(x, dtype=float).copy()
            return y

        # Evaluate initial solution if budget allows.
        eval_at(best_x)
        if evals >= budget:
            return best_x, best_y

        # Step-size initialization: fraction of box size, with floor to avoid stalling.
        box = ub - lb
        # If bounds are degenerate, avoid zero sigma.
        sigma_floor = 1e-12
        sigma = np.maximum(0.2 * box, sigma_floor)

        # Population/offspring sizes: keep lightweight.
        # Ensure at least 2 candidates per iteration (including possible improvements).
        # The harness sets RNG seed before runs, so randomness is reproducible.
        lambda_base = max(4, min(24, 2 + dim))
        # Aim for about 10-25 iterations but respect budget.
        max_iters = max(1, budget // max(1, lambda_base))
        # If budget is small, reduce offspring count so we don't overshoot.
        lam = min(lambda_base, max(2, (budget - evals) // 1))  # at least 2 if possible

        # Multiplicative adaptation factors.
        shrink = 0.82
        grow = 1.18

        # Diversification probability for averaged candidates.
        p_avg = 0.35

        # Main loop; stop when budget is exhausted.
        improved_prev = True
        it = 0
        while evals < budget:
            it += 1
            # Adjust lambda if nearing budget.
            remaining = budget - evals
            if remaining <= 0:
                break
            lam_eff = min(lam, remaining)

            # Generate offspring candidates around best_x.
            # We'll evaluate sequentially and stop if the budget ends mid-generation.
            # Candidate pool only needed for averaging diversity.
            offspring = []
            # A small pool of anchors for averaging. Include best_x and some random
            # points inside bounds to help escape flat basins.
            pool = [best_x]
            # Add a few random points for averaging (only if we can afford extra evals).
            # We do NOT evaluate these points; they are used only for mixing with
            # best_x before mutation.
            mix_count = 0
            if dim > 1 and remaining >= 4:
                mix_count = 2
            for _ in range(mix_count):
                r = np.random.rand(dim)
                pool.append(lb + r * (ub - lb))

            # Create offspring via Gaussian mutations.
            for k in range(lam_eff):
                if np.random.rand() < p_avg and len(pool) >= 2:
                    # Averaged candidate: best_x mixed with a random pool point.
                    a = pool[np.random.randint(len(pool))]
                    alpha = np.random.rand(dim)
                    center = alpha * best_x + (1.0 - alpha) * a
                else:
                    center = best_x

                # Gaussian step: per-coordinate sigma.
                # Use a diagonal covariance for compactness and speed.
                z = np.random.randn(dim)
                x = center + sigma * z

                # Boundary handling: clip to bounds.
                if np.any(x < lb) or np.any(x > ub):
                    x = np.minimum(ub, np.maximum(lb, x))

                offspring.append(x)

            # Evaluate offspring; track whether any improvement happened this iteration.
            best_y_before = best_y
            for x in offspring:
                if evals >= budget:
                    break
                eval_at(x)

            improved = (best_y is not None and best_y_before is not None and best_y < best_y_before - 1e-15)

            # Adapt sigma per coordinate using success signal.
            # If improved, shrink to exploit; otherwise grow to explore.
            if improved:
                # Shrink slightly but not too fast; keep minimal floor.
                sigma = np.maximum(sigma_floor, sigma * shrink)
                improved_prev = True
            else:
                sigma = np.minimum(np.maximum(sigma_floor, sigma * grow), np.maximum(box, sigma_floor))
                improved_prev = False

            # If the search seems stuck near boundaries, gently re-expand.
            # Heuristic: measure how often best_x lies on or near the boundary.
            if it % 5 == 0:
                near = (best_x - lb <= 1e-10) | (ub - best_x <= 1e-10)
                frac = float(np.mean(near))
                if frac > 0.6 and not improved:
                    sigma = np.minimum(np.maximum(sigma_floor, sigma * 1.15), np.maximum(box, sigma_floor))

        return best_x, best_y

    @staticmethod
    def _read_bounds(func):
        # Priority:
        # 1) func.lower / func.upper
        # 2) func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return np.asarray(func.lower, dtype=float), np.asarray(func.upper, dtype=float)
        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return np.asarray(b.lb, dtype=float), np.asarray(b.ub, dtype=float)
            # Also accept generic 'lower/upper' in bounds.
            if hasattr(b, "lower") and hasattr(b, "upper"):
                return np.asarray(b.lower, dtype=float), np.asarray(b.upper, dtype=float)
        raise AttributeError("Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub.")
