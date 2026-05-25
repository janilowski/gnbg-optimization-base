# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# based on a multi-start, cooperative coordinate search with occasional
# randomized “global” restarts. It maintains a single incumbent solution and
# repeatedly proposes new candidates using a mixture of local coordinate
# perturbations and global uniform sampling.
# Search state: The algorithm tracks the current best point x_best and its
# objective value y_best, plus an evaluation counter. It also tracks a step
# size (sigma) that controls the scale of local perturbations and is reduced
# when progress stalls.
# Candidate generation: Each iteration proposes:
#   1) a local candidate by selecting one coordinate at random and applying
#      a signed move with magnitude sigma (with a mirrored alternative),
#   2) optionally a small isotropic Gaussian perturbation around the best,
#   3) with some probability, a global candidate sampled uniformly from the
#      bounds (restart-like exploration).
# Selection and replacement: Proposed candidates are evaluated one by one
# (never exceeding the remaining budget). If a candidate improves the current
# best, it becomes the new incumbent.
# Adaptation: sigma is decreased gradually as iterations proceed, and when
# no improvement occurs for several steps, sigma is further reduced to refine
# around the incumbent.
# Exploration mechanisms: Random coordinate moves and occasional global
# uniform sampling provide exploration across the search space.
# Exploitation mechanisms: The algorithm heavily relies on perturbations around
# the best-so-far point with decreasing sigma to exploit local basins.
# Boundary handling: After generating a candidate, each coordinate is clipped
# to the valid bounds to ensure feasibility.
# Budget strategy: The number of evaluations is strictly limited by the given
# budget. The algorithm monitors remaining evaluations and stops immediately
# when the budget is exhausted.
# Closest known influences: The design is inspired by derivative-free coordinate
# search / pattern search, combined with annealed step sizing and restart-like
# global sampling.
# Novelty or unusual aspects: The implementation uses both mirrored local moves
# on a chosen coordinate and an adaptive sigma schedule with budget-aware
# termination, aiming for robustness across dimensions.
# Failure modes: If the objective landscape is highly irregular or the bounds
# are extremely tight, progress may stall and sigma may shrink too early;
# global sampling helps mitigate this but cannot guarantee success under all
# adversarial functions.
# ALGORITHM_ANALYSIS_NOTE_END

from typing import Callable, Tuple, Optional
import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func) -> Tuple[np.ndarray, float]:
        d = self.dim
        if d <= 0:
            raise ValueError("dim must be positive")

        # Read bounds from either func.lower/func.upper or func.bounds.lb/ub.
        lb, ub = None, None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
        elif hasattr(func, "bounds"):
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float).reshape(-1)
            ub = np.asarray(b.ub, dtype=float).reshape(-1)
        else:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub")

        if lb.size != d or ub.size != d:
            raise ValueError("Bounds dimension mismatch with dim")

        # Ensure valid ordering.
        lb, ub = np.minimum(lb, ub), np.maximum(lb, ub)
        span = ub - lb
        # Handle degenerate spans: avoid zero step sizes by using a tiny scale.
        span_nonzero = np.where(span > 0, span, 1.0)
        max_span = float(np.max(span_nonzero))

        rng = np.random  # harness sets global numpy seed

        evals = 0
        best_x: Optional[np.ndarray] = None
        best_y: Optional[float] = None

        def eval_at(x: np.ndarray) -> float:
            nonlocal evals, best_x, best_y
            # Enforce evaluation budget strictly.
            if evals >= self.budget:
                raise RuntimeError("Evaluation budget exceeded")
            y = float(func(x))
            evals += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Start from either the center or a random point (if center is degenerate).
        x0 = lb + 0.5 * (ub - lb)
        if np.all(span == 0):
            # Only one feasible point exists.
            best_x = x0.copy()
            best_y = eval_at(best_x)
            return best_x, best_y

        # Evaluate initial incumbent with some randomness.
        try:
            # Budget-aware initial sampling: evaluate a few starting points if budget allows.
            n_init = 1
            if self.budget >= 10:
                n_init = min(3, self.budget)
            for _ in range(n_init):
                if evals >= self.budget:
                    break
                if rng.random() < 0.5:
                    x = x0.copy()
                else:
                    x = lb + rng.random(d) * (ub - lb)
                x = np.clip(x, lb, ub)
                eval_at(x)
        except RuntimeError:
            # If budget is too small, we return what we have.
            if best_x is None:
                # Should not happen, but keep robust.
                best_x = x0.copy()
                best_y = float(func(best_x))
            return best_x, best_y

        if best_x is None:
            best_x = x0.copy()
            best_y = eval_at(best_x)

        # Initial step size: fraction of range.
        # Use a schedule that shrinks over time and also reacts to stagnation.
        t_total = max(1, self.budget - evals)
        sigma = 0.35 * max_span
        min_sigma = 1e-12 * max(1.0, max_span)

        no_improve = 0
        # Exploration probability decreases as budget runs out.
        base_explore = 0.35

        # Main loop: each step evaluates 1-3 candidates (budget-aware).
        while evals < self.budget:
            remaining = self.budget - evals
            # Compute an iteration-adaptive exploration rate.
            progress = evals / max(1, self.budget)
            p_explore = base_explore * (1.0 - progress)
            p_explore = float(np.clip(p_explore, 0.02, base_explore))

            # Occasionally do a global sample to escape local basins.
            do_global = (rng.random() < p_explore) and (remaining > 0)

            candidates = []

            if do_global:
                xg = lb + rng.random(d) * (ub - lb)
                candidates.append(xg)
            else:
                # Exploitation: coordinate perturbation around best.
                k = int(rng.randint(0, d))
                # Signed move with Gaussian magnitude scaled by sigma.
                # Use abs(Gauss)*sign to avoid bias toward small magnitudes.
                mag = abs(float(rng.randn())) * sigma
                sign = -1.0 if rng.random() < 0.5 else 1.0

                # Two mirrored candidates to better probe along chosen coordinate.
                dx = np.zeros(d, dtype=float)
                dx[k] = sign * mag
                c1 = best_x + dx
                c2 = best_x - dx

                # Optional isotropic refinement candidate.
                # Keep it lightweight; only add if we have budget.
                candidates.append(c1)
                if remaining >= 2:
                    candidates.append(c2)
                if remaining >= 3 and rng.random() < 0.5:
                    # Small Gaussian around best, scaled to sigma and clipped.
                    ga = rng.randn(d) * (0.15 * sigma)
                    candidates.append(best_x + ga)

            # Clip candidates to bounds.
            # Evaluate candidates sequentially, never exceeding remaining budget.
            improved = False
            for xc in candidates:
                if evals >= self.budget:
                    break
                xc = np.clip(np.asarray(xc, dtype=float), lb, ub)
                # If candidate is identical to best due to tight bounds, still evaluate;
                # but this could be redundant. We'll avoid exact duplicates by skipping
                # if it would be exactly the same (common with zero span).
                if np.array_equal(xc, best_x):
                    # Still allow evaluation if budget allows? Better to skip to save calls.
                    # However, some objective functions might be sensitive to exact inputs;
                    # identical vectors should yield identical values, so skipping is safe.
                    continue
                prev_y = best_y
                try:
                    eval_at(xc)
                except RuntimeError:
                    break
                if best_y is not None and prev_y is not None and best_y < prev_y:
                    improved = True

            # Adaptation based on progress.
            if improved:
                no_improve = 0
                # Slightly increase sigma if progress is steady early on; otherwise keep decreasing.
                # This helps avoid over-shrinking.
                sigma *= 0.95 + 0.05 * (rng.random())
            else:
                no_improve += 1
                sigma *= 0.92

            # Additional shrink when stagnating.
            if no_improve >= 8:
                sigma = max(min_sigma, sigma * 0.6)
                no_improve = 0

            # Budget-aware gentle decay.
            # Ensures eventual convergence/termination behavior.
            decay = 1.0 - (0.25 * (evals / max(1, self.budget)))
            sigma = max(min_sigma, sigma * max(0.8, decay))

            if sigma <= min_sigma and remaining <= 1:
                break

        # Final guard: best_x/best_y should be set if budget >= 1.
        if best_x is None:
            x0 = lb + 0.5 * (ub - lb)
            best_x = np.clip(x0, lb, ub)
            best_y = float(func(best_x))
        return best_x, float(best_y)
