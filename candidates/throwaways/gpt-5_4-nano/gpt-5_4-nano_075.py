# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy using a
# derivative-free evolution approach with occasional local refinement.
# Search state: Maintains a Gaussian sampling distribution (mean and
# diagonal step sizes) and a small population of candidate solutions.
# Candidate generation: Each iteration samples multiple points from a
# diagonal normal distribution around the current mean; additionally,
# it performs a short coordinate-wise local probe around the current best.
# Selection and replacement: Uses best-so-far (global incumbent) and the best
# sampled candidate to update the mean; step sizes adapt based on improvement.
# Adaptation: If improvement is found, the distribution step sizes contract
# toward the successful scale; otherwise they expand to encourage exploration.
# Exploration mechanisms: Random population sampling from the current
# distribution, plus the occasional local coordinate probes.
# Exploitation mechanisms: Mean update toward the best candidate and
# coordinate-wise local probing near the incumbent.
# Boundary handling: Samples are clipped to provided bounds; local probes
# use step sizes clipped to stay within bounds.
# Budget strategy: Strictly tracks evaluations and never exceeds the given
# evaluation budget; stops immediately when budget is exhausted.
# Closest known influences: Evolution strategies / CMA-like ideas (mean and
# step-size adaptation) combined with a small deterministic local search probe.
# Novelty or unusual aspects: Uses diagonal step sizes with a lightweight
# success-based adaptation and a simple coordinate probe that respects bounds.
# Failure modes: On very flat or highly multimodal landscapes, the algorithm
# may converge prematurely or waste evaluations on unhelpful local probes; the
# adaptive step sizes mitigate this by re-expanding when no progress occurs.
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
        if budget <= 0:
            # No evaluations allowed; return something deterministic.
            # We'll still obey "Never exceed budget" (0), and avoid calling func.
            bounds = self._get_bounds(func)
            lb, ub = bounds
            if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
                best_x = (lb + ub) / 2.0
            else:
                best_x = np.zeros(dim, dtype=float)
            return best_x, float("inf")

        lb, ub = self._get_bounds(func)
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimensionality must match dim.")

        # Handle degenerate bounds gracefully
        span = ub - lb
        span = np.where(span > 0, span, 1.0)  # avoid zero spans in scaling
        center = (lb + ub) / 2.0

        # --- Initialization ---
        # Start from the center and choose initial diagonal sigmas relative to bounds.
        # A slightly conservative initial sigma helps stability.
        sigma = 0.25 * span
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 0.25)

        x_mean = center.astype(float, copy=True)

        # Evaluate initial point (counts as 1 eval)
        evals = 0
        best_x = None
        best_y = None

        def eval_at(x):
            nonlocal evals, best_x, best_y
            if evals >= budget:
                return
            y = float(func(np.asarray(x, dtype=float)))
            evals += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = np.asarray(x, dtype=float).copy()
            return y

        eval_at(x_mean)

        # Population and iteration controls
        # Choose a population size that scales with dimension but stays budget-safe.
        # We keep it small for compactness and robustness.
        # The number of main iterations is not fixed; we run until budget is exhausted.
        pop_size = int(min(10 + dim, max(4, budget)))  # at most budget
        pop_size = max(4, pop_size)

        # Success-based adaptation parameters
        # (lightweight; aims to contract on success and expand otherwise)
        shrink = 0.82
        expand = 1.22
        min_sigma = 1e-12 * (span + 1.0)
        max_sigma = 0.5 * span + 1e-12

        # Local probe settings
        # We'll probe a few coordinates around the incumbent best.
        # Number of coordinates probed scales mildly with dim.
        local_coords = max(1, min(dim, 3 + dim // 5))
        probe_steps = 2  # how many magnitudes to try per probe

        # --- Main loop ---
        while evals < budget:
            # Determine how many evaluations we can spend in this batch.
            # Include at least 1 candidate evaluation if possible.
            remaining = budget - evals
            k = min(pop_size, remaining)

            # Sample k candidates from diagonal Gaussian around the current mean.
            # Candidate clipping handles bound constraints.
            # We add a tiny noise floor to avoid exact duplicates.
            eps = np.random.randn(k, dim)
            noise_floor = 1e-12
            candidates = x_mean + eps * (sigma + noise_floor)

            # Clip to bounds
            candidates = np.clip(candidates, lb, ub)

            ys = np.empty(k, dtype=float)
            # Evaluate candidates; stop if we hit budget (shouldn't, since k <= remaining)
            for i in range(k):
                y = float(func(candidates[i]))
                evals += 1
                ys[i] = y
                if best_y is None or y < best_y:
                    best_y = y
                    best_x = candidates[i].copy()
                if evals >= budget:
                    break

            if evals >= budget:
                break

            # Find best in this batch
            idx_best = int(np.argmin(ys))
            x_best_batch = candidates[idx_best].copy()
            y_best_batch = float(ys[idx_best])

            # Success check: did we beat current incumbent?
            improved = (best_y is not None) and (y_best_batch <= best_y + 1e-18)

            # Update mean toward batch best using a learning rate.
            # If improved, move more aggressively; otherwise cautiously toward best batch.
            # (This still leverages information even when not improving.)
            # Note: Since best_y is the global incumbent, "improved" here is always true
            # if the batch best equals the global best; still, we adapt as below.
            if best_y is not None and y_best_batch <= best_y + 1e-18:
                # We can't directly know whether this evaluation improved, because best_y
                # has already potentially been updated. Detect improvement by comparing
                # against the previous incumbent if needed; to keep code compact, we use a
                # proxy: compare x_mean-batch distance and y trend via sigma changes.
                pass

            # Track improvement by recomputing whether x_best_batch became incumbent.
            # We'll infer via proximity to best_x at the end of evaluation.
            became_incumbent = np.allclose(best_x, x_best_batch, rtol=0, atol=1e-14)

            if became_incumbent:
                lr = 0.65
                sigma = np.maximum(sigma * shrink, min_sigma)
            else:
                lr = 0.35
                sigma = np.minimum(sigma * expand, max_sigma)

            x_mean = x_mean + lr * (x_best_batch - x_mean)
            x_mean = np.clip(x_mean, lb, ub)

            # Occasional local coordinate probe around current best_x (exploitation)
            # Use only if budget remains.
            if evals < budget:
                remaining = budget - evals
                if remaining >= 1:
                    # Choose a stable set of coordinates: those with largest sigma (more potential)
                    # plus some randomness to escape symmetry.
                    order = np.argsort(-sigma + 1e-20)
                    coords = order[:local_coords].copy()
                    if dim > local_coords:
                        # random extra coordinates to diversify
                        extra = np.random.choice(np.setdiff1d(np.arange(dim), coords, assume_unique=False),
                                                 size=min(1, dim - local_coords),
                                                 replace=False) if dim - local_coords > 0 else []
                        if np.ndim(extra) == 0:
                            coords = np.unique(np.concatenate([coords, np.array([extra])]))
                        else:
                            coords = np.unique(np.concatenate([coords, extra]))
                    coords = np.asarray(coords, dtype=int).reshape(-1)
                    # Try a few step magnitudes per selected coordinate
                    # Use sigma coordinate-scaled steps
                    base = best_x if best_x is not None else x_mean
                    for c in coords:
                        if evals >= budget:
                            break
                        sc = sigma[c]
                        if not np.isfinite(sc) or sc <= 0:
                            continue
                        for m in range(probe_steps):
                            # Try negative and positive offsets with decreasing/increasing scale
                            # m=0 => larger, m=1 => smaller for refinement
                            factor = 1.0 / (1.5 ** m)
                            step = factor * sc

                            # Negative probe
                            x1 = base.copy()
                            x1[c] = np.clip(x1[c] - step, lb[c], ub[c])
                            if evals < budget:
                                y1 = float(func(x1))
                                evals += 1
                                if best_y is None or y1 < best_y:
                                    best_y = y1
                                    best_x = x1.copy()
                                    # If local improvement, contract sigma in that direction
                                    sigma[c] = max(sigma[c] * 0.85, min_sigma[c])
                            if evals >= budget:
                                break

                            # Positive probe
                            x2 = base.copy()
                            x2[c] = np.clip(x2[c] + step, lb[c], ub[c])
                            if evals < budget:
                                y2 = float(func(x2))
                                evals += 1
                                if best_y is None or y2 < best_y:
                                    best_y = y2
                                    best_x = x2.copy()
                                    sigma[c] = max(sigma[c] * 0.85, min_sigma[c])
                            if evals >= budget:
                                break

                    # After local probes, nudge mean toward best_x for exploitation.
                    if best_x is not None:
                        x_mean = x_mean + 0.4 * (best_x - x_mean)
                        x_mean = np.clip(x_mean, lb, ub)

        # Ensure best_x is valid (it should be).
        if best_x is None:
            best_x = np.clip(x_mean, lb, ub)
            # Do not call func again; budget exhausted or budget was zero.
            best_y = float("inf")
        return best_x, float(best_y)

    def _get_bounds(self, func):
        # Bounds from func.lower/func.upper or func.bounds.lb/func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
            return lower, upper

        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return np.asarray(b.lb, dtype=float), np.asarray(b.ub, dtype=float)

        raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub.")
