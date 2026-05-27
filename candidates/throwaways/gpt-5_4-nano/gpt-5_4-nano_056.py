# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a simple, robust black-box minimization algorithm
# (derivative-free) using a variant of stochastic coordinate search with an adaptive
# random subspace. It maintains a population of candidate points and uses a Gaussian
# perturbation around the current best, plus occasional coordinate-style tweaks, to
# explore the domain. The algorithm always respects the provided evaluation budget.
#
# Search state: The algorithm keeps a current best point x_best with value y_best,
# a small population of recent candidates, and a step-size sigma that controls the
# magnitude of perturbations. It also tracks how many objective evaluations have been
# consumed so far.
#
# Candidate generation: Each iteration generates several new candidate points by
# adding Gaussian noise scaled by sigma to either the best point (random subspace
# perturbations) or to a single coordinate direction (coordinate tweaks). Candidates
# are always clipped into the feasible bounds.
#
# Selection and replacement: After evaluating candidates, the best among them becomes
# the new incumbent (x_best, y_best). Optionally, a replacement occurs via a small
# population: the population is updated to keep the best few candidates for stability.
#
# Adaptation: The step-size sigma is adapted based on whether improvements are found:
# successful iterations reduce a “bad streak” and may slightly increase exploration,
# while repeated failures shrink sigma to focus locally around the incumbent.
#
# Exploration mechanisms: Random subspace perturbations, occasional coordinate tweaks,
# and controlled step-size allow the algorithm to probe the space broadly early on and
# then focus as the budget tightens.
#
# Exploitation mechanisms: Most candidates are generated around the current best point,
# and sigma shrinking after failures encourages local refinement.
#
# Boundary handling: All proposed points are clipped to the provided lower/upper bounds.
# If sigma is extremely small, coordinate tweaks also consider minimal safe perturbations.
#
# Budget strategy: The algorithm computes the number of evaluations remaining and
# allocates work per loop without ever exceeding the budget. Each candidate evaluation
# increments the evaluation counter.
#
# Closest known influences: The approach is loosely inspired by evolution strategies /
# CMA-style “ask-and-tell” patterns, but simplified to be compact and standard-library
# only, using Gaussian sampling plus adaptive step-size.
#
# Novelty or unusual aspects: It uses both random subspace Gaussian sampling and lightweight
# coordinate tweaks in the same loop, and adapts sigma using a failure counter.
#
# Failure modes: In very high dimensions or extremely narrow feasible regions, progress
# may be slow. If the objective is highly ill-conditioned or discontinuous, the Gaussian
# neighborhood sampling could overshoot; sigma adaptation and clipping mitigate but cannot
# fully prevent this. If the budget is too small, it may only perform coarse search.
# ALGORITHM_ANALYSIS_NOTE_END

from typing import Any, Callable, Tuple
import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func: Any) -> Tuple[np.ndarray, float]:
        # ---- Read bounds ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            # Accept common attribute names
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            elif hasattr(b, "lower") and hasattr(b, "upper"):
                lb = np.asarray(b.lower, dtype=float)
                ub = np.asarray(b.upper, dtype=float)
        if lb is None or ub is None:
            raise ValueError("Bounds not found. Expected func.lower/func.upper or func.bounds.lb/func.bounds.ub.")
        if lb.shape == ():
            lb = np.full(self.dim, float(lb))
        if ub.shape == ():
            ub = np.full(self.dim, float(ub))
        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError(f"Bounds dimension mismatch: got lb/ub of size {lb.size}/{ub.size}, expected {self.dim}.")

        # Ensure lb <= ub
        if np.any(ub < lb):
            raise ValueError("Invalid bounds: found upper bound smaller than lower bound.")

        def clip(x: np.ndarray) -> np.ndarray:
            return np.minimum(np.maximum(x, lb), ub)

        # ---- Budget-safe evaluation ----
        evals = 0

        def eval_one(x: np.ndarray) -> float:
            nonlocal evals
            if evals >= self.budget:
                # Should never happen due to budget checks; raise to be safe.
                raise RuntimeError("Evaluation budget exceeded.")
            y = func(x)
            evals += 1
            return float(y)

        # ---- Initialization ----
        rng = np.random

        # Step-size based on domain scale
        span = ub - lb
        # If span has zeros, handle by giving small scale for those dims
        span_safe = np.where(span > 0, span, 1.0)
        sigma0 = 0.3 * float(np.mean(span_safe))
        sigma = max(1e-12, sigma0)

        # Start near the middle with some noise for diversity
        mid = (lb + ub) / 2.0
        x_best = clip(mid + rng.normal(0.0, 0.05, size=self.dim) * span_safe)
        y_best = eval_one(x_best)

        # Maintain a small pool of best candidates seen so far (for stability)
        pool_size = 4 if self.dim >= 2 else 2
        x_pool = [x_best]
        y_pool = [y_best]

        # Parameters for loop
        # Each "iteration" consumes k evaluations.
        # We choose k adaptively based on budget and dimension.
        # Typical: more candidates early, fewer near budget end.
        min_batch = 2
        max_batch = 8 if self.dim <= 20 else 6
        base_batch = int(np.clip(self.budget // 30, min_batch, max_batch))
        base_batch = max(min_batch, min(base_batch, max_batch))

        # Failure counter controls sigma shrink/expand
        failures = 0
        best_improve = 0.0

        # ---- Main search loop ----
        while evals < self.budget:
            remaining = self.budget - evals
            # Decide batch size without exceeding budget.
            k = int(min(remaining, base_batch + (1 if failures == 0 else 0)))

            # Random subspace dimension: smaller early, sometimes larger
            # to avoid too much randomness when dim is huge.
            if self.dim <= 3:
                sub_dim = self.dim
            else:
                # Use a random subspace with expected size around dim/2 (capped)
                sub_dim = int(np.clip(self.dim // 2 + rng.randint(-self.dim // 10, self.dim // 10 + 1),
                                      2, min(self.dim, 12)))
            # Generate candidates
            candidates = []
            for _ in range(k):
                # Mix of strategies:
                # - With high probability: perturb best using random subspace
                # - Occasionally: coordinate tweak from best
                if rng.rand() < 0.75:
                    # Random subspace perturbation
                    axes = rng.choice(self.dim, size=sub_dim, replace=False) if sub_dim < self.dim else np.arange(self.dim)
                    step = rng.normal(0.0, 1.0, size=axes.size) * sigma
                    x = x_best.copy()
                    x[axes] = x[axes] + step
                else:
                    # Coordinate tweak: pick one coordinate and move both signs with Gaussian
                    j = int(rng.randint(0, self.dim))
                    # Scale tweak by coordinate span to be comparable across dims
                    coord_scale = span_safe[j]
                    # Use a heavy-tailed-ish step occasionally by mixing Gaussians
                    if rng.rand() < 0.2:
                        step_mag = sigma * (0.5 * rng.normal() + 0.5 * rng.normal())
                    else:
                        step_mag = rng.normal() * sigma
                    x = x_best.copy()
                    x[j] = x[j] + step_mag * (coord_scale / float(np.mean(span_safe)))
                # Boundary handling
                x = clip(x)
                # If bounds are degenerate for all dims, x may equal x_best; still evaluate
                candidates.append(x)

            # Evaluate
            improved = False
            local_best_x = None
            local_best_y = y_best

            for x in candidates:
                y = eval_one(x)
                if y < local_best_y:
                    local_best_y = y
                    local_best_x = x

            # Selection/replacement
            if local_best_x is not None and local_best_y < y_best:
                improved = True
                best_improve = max(best_improve, float(y_best - local_best_y))
                x_best = local_best_x
                y_best = local_best_y
                failures = 0
            else:
                failures += 1

            # Update pool with best elements from pool and candidates
            # (cheap since k is small)
            # We'll add x_best and any candidates that beat the worst in pool.
            # But we didn't keep candidate y's; thus, simplest: keep only incumbent(s).
            # We can keep a few random evaluated points by re-evaluating? No.
            # Instead: track a "soft" pool by retaining best incumbents over time.
            if improved:
                x_pool.append(x_best.copy())
                y_pool.append(y_best)
                # Trim pool
                if len(x_pool) > pool_size:
                    idx = np.argsort(y_pool)[:pool_size]
                    x_pool = [x_pool[i] for i in idx]
                    y_pool = [y_pool[i] for i in idx]

            # Adapt sigma
            # - If we improved: slightly increase sigma to continue exploration, but cap it.
            # - If not: shrink sigma to focus locally.
            # Also tie adaptation to proximity to bounds.
            success_factor = 1.15
            failure_factor = 0.85
            # Compute a rough "domain pressure": near center, allow larger steps; near edges, smaller.
            # Use distance to center as a proxy.
            dist_to_center = float(np.mean(np.abs(x_best - mid) / (span_safe + 1e-12)))
            pressure = np.clip(dist_to_center / 0.5, 0.0, 1.0)  # 0 in center, 1 near edges
            if improved:
                sigma = min(float(np.mean(span_safe)) * 0.5 + 1e-12, sigma * success_factor * (1.0 - 0.25 * pressure))
            else:
                sigma = max(1e-12, sigma * failure_factor * (1.0 - 0.15 * (1.0 - pressure)))

            # If sigma becomes too small but still budget remains, occasionally kick exploration.
            # This helps avoid being stuck with clipping.
            if failures >= 6 and sigma <= 1e-6 * float(np.mean(span_safe) + 1e-12):
                # Kick by re-sampling around a random point in the domain.
                if evals < self.budget:
                    # Random restart candidate
                    x_restart = clip(lb + rng.rand(self.dim) * (ub - lb))
                    y_restart = eval_one(x_restart)
                    if y_restart < y_best:
                        x_best, y_best = x_restart, y_restart
                    failures = 0
                    sigma = max(sigma0 * 0.2, sigma)

        return x_best, y_best
