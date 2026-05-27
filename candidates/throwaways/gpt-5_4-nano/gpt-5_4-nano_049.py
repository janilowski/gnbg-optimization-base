# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm combining
# randomized sampling, coordinate-aligned local improvement, and a small
# population-based evolution strategy. It maintains and updates a search
# region while strictly respecting the evaluation budget.
# Search state: Tracks current best point (best_x, best_y) and a scalar
# step-size (sigma) controlling how far new candidates are sampled. Also keeps
# a list of recent successful improvements to adapt sigma.
# Candidate generation: 
# - Global phase: draws random points uniformly within bounds.
# - Local phase: generates candidates by perturbing the best point using
#   isotropic Gaussian noise and axis-aligned steps. The perturbation scales
#   with sigma and adapts over time.
# - Evolution sub-phase: samples offspring around the current best with
#   Gaussian noise and selects the best offspring.
# Selection and replacement: Always retains the best-so-far point. When an
# improvement is found, sigma is reduced slightly; otherwise it is increased
# slightly to encourage exploration.
# Adaptation: Sigma is updated using a simple success/failure heuristic based on
# whether any candidate improved the best_y during the last chunk of evaluations.
# Exploration mechanisms: Uniform random sampling and larger sigma expansions
# when improvements stall.
# Exploitation mechanisms: Perturbations around best_x with decreasing sigma and
# coordinate-aligned moves.
# Boundary handling: Any candidate coordinate is clipped to [lb, ub] after
# generation. This ensures feasibility without extra evaluations.
# Budget strategy: Uses a hard remaining-evaluations counter. Each objective
# call decreases the counter; no evaluation is performed once the budget is
# exhausted.
# Closest known influences: Loosely resembles CMA-ES-lite / (μ+λ) ES with
# success-based step-size adaptation and occasional uniform restarts, but kept
# deliberately simple and dimension-robust.
# Novelty or unusual aspects: Uses both isotropic and axis-aligned candidate
# perturbations in one step, and adapts sigma based on a chunk-level success
# indicator to remain compact yet responsive.
# Failure modes: With extremely small budgets or very narrow feasible regions,
# improvements may be missed; clipping can cause reduced diversity. The algorithm
# still returns the best point evaluated within budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # --- Read bounds from func ---
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Bounds not found. Expected func.lower/func.upper or func.bounds.lb/func.bounds.ub.")

        if lb.shape[0] != dim or ub.shape[0] != dim:
            lb = np.reshape(lb, (dim,))
            ub = np.reshape(ub, (dim,))

        # Ensure valid ranges (robustness against degenerate bounds)
        lb = np.minimum(lb, ub)
        ub = np.maximum(lb, ub)

        # Handle the extreme corner case of zero dimension
        if dim == 0:
            # Best_x is empty; evaluate once if allowed.
            if budget <= 0:
                return np.zeros(0, dtype=float), float("inf")
            x0 = np.zeros(0, dtype=float)
            y0 = func(x0)
            return x0, float(y0)

        # --- Evaluation budget management ---
        remaining = budget

        def eval_at(x):
            nonlocal remaining
            if remaining <= 0:
                # Should never happen; guard for robustness.
                return float("inf")
            remaining -= 1
            return float(func(np.asarray(x, dtype=float)))

        # --- Helper: clip to bounds ---
        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        # Initial sigma based on bounds scale
        span = ub - lb
        # If all bounds collapse, span==0 => sigma=0
        base_span = float(np.max(span)) if np.max(span) > 0 else 1.0
        sigma0 = 0.3 * base_span / max(1.0, np.sqrt(dim))
        sigma0 = max(sigma0, 1e-12)

        # --- Initialize with a best-so-far point ---
        # Try to evaluate the center first (common good default).
        x_best = clip((lb + ub) / 2.0)
        y_best = eval_at(x_best)

        # If budget only allows one eval
        if remaining <= 0:
            return x_best, y_best

        # --- Determine phase lengths (budget-aware) ---
        # Keep it simple: global random sampling fraction + local refinement.
        # The algorithm remains meaningful even for small budgets.
        b = budget
        # Global fraction: ~25%, but at least some points if possible.
        n_global = max(0, min(remaining, int(round(0.25 * b))))
        # We'll allocate remaining evaluations across global + iterative local chunks.
        # Each local chunk uses a small batch of evaluations.
        # Batch size grows moderately with dim.
        batch = max(3, min(12, 2 + int(np.log2(dim + 1)) * 2))
        local_iters = 0

        # --- Global exploration: uniform random points ---
        if n_global > 0:
            # sample n_global points uniformly in box
            for _ in range(n_global):
                if remaining <= 0:
                    break
                r = np.random.rand(dim)
                x = lb + r * (ub - lb)
                y = eval_at(x)
                if y < y_best:
                    y_best = y
                    x_best = x

        # --- Main refinement loop ---
        sigma = sigma0
        recent_success = 0
        # Keep iterating until budget runs out.
        while remaining > 0:
            local_iters += 1

            # If sigma becomes extremely small, still try axis moves on best.
            sigma_eff = float(sigma)
            if sigma_eff <= 1e-15:
                sigma_eff = 1e-15

            # Decide number of candidates in this chunk
            k = min(batch, remaining)  # ensure we don't exceed budget

            candidates = []
            # Candidate 1: best itself (useful baseline; sometimes budget is tiny)
            candidates.append(x_best.copy())
            # Remaining: build perturbations
            for i in range(k - 1):
                if i % 3 == 0:
                    # Isotropic Gaussian around best
                    z = np.random.randn(dim)
                    x = x_best + sigma_eff * z
                elif i % 3 == 1:
                    # Axis-aligned perturbation: pick coordinate and move
                    j = np.random.randint(0, dim)
                    step = sigma_eff * np.random.randn()
                    x = x_best.copy()
                    x[j] = x[j] + step
                else:
                    # Mixed: small Gaussian + occasional uniform direction
                    z = np.random.randn(dim)
                    x = x_best + (0.7 * sigma_eff) * z
                    # Add a tiny uniform jitter to maintain diversity
                    if dim >= 2:
                        rdir = np.random.rand(dim) - 0.5
                        x = x + 0.1 * sigma_eff * rdir

                x = clip(x)
                candidates.append(x)

            # Evaluate candidates and select best in this chunk
            improved = False
            y_chunk_best = y_best
            x_chunk_best = x_best

            for x in candidates:
                if remaining <= 0:
                    break
                y = eval_at(x)
                if y < y_chunk_best:
                    y_chunk_best = y
                    x_chunk_best = x
                    improved = True

            if y_chunk_best < y_best:
                y_best = y_chunk_best
                x_best = x_chunk_best
                recent_success += 1
            else:
                recent_success = max(0, recent_success - 1)

            # Success-based step-size adaptation:
            # - On improvement: contract
            # - Otherwise: expand (bounded by box size)
            if improved:
                sigma *= 0.82 + 0.02 * (1.0 / (1.0 + local_iters))
            else:
                sigma *= 1.18 + 0.01 * (1.0 / (1.0 + local_iters))

            # Bound sigma so it doesn't explode or vanish too hard
            max_sigma = max(1e-12, float(np.max(ub - lb)))
            if max_sigma <= 0:
                max_sigma = 1.0
            sigma = float(np.clip(sigma, 1e-15, 0.6 * max_sigma + 1e-15))

            # Optional micro-restart when stagnating:
            # If multiple chunks pass without improvement, sample a couple of points.
            # (This uses remaining budget safely via eval_at.)
            if recent_success == 0 and remaining > 0:
                # Do a very small restart attempt to keep evaluation count tight.
                if local_iters % 3 == 0:
                    # Only evaluate up to 2 points if budget permits.
                    for _ in range(2):
                        if remaining <= 0:
                            break
                        r = np.random.rand(dim)
                        x = lb + r * (ub - lb)
                        y = eval_at(x)
                        if y < y_best:
                            y_best = y
                            x_best = x
                            recent_success = 1
                            break

        return x_best, y_best
