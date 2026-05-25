# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimization algorithm
# (a bounded stochastic search). It maintains a current best point and a Gaussian
# search distribution around it, shrinking the distribution as improvements occur.
# Search state: Tracks current best solution x_best and its objective value y_best,
# along with a step-size sigma and an evaluation counter to respect the budget.
# Candidate generation: In each iteration, samples a small batch of candidates by
# adding zero-mean Gaussian noise to x_best. Each candidate is clipped back into
# the provided bounds. The batch size adapts to dimension to balance exploration
# and exploitation.
# Selection and replacement: Evaluates all candidates in the batch (without
# exceeding the remaining budget), selects the best among them, and replaces
# x_best if an improvement is found.
# Adaptation: If improvement occurs, sigma is slightly reduced (more focused search);
# otherwise, sigma is increased or kept larger to encourage escape from local
# minima, similar to a simple success-based adaptation.
# Exploration mechanisms: Random sampling with a nonzero sigma and occasional
# larger steps when no progress is observed.
# Exploitation mechanisms: Sigma shrinks after improvements, concentrating search
# around the best-so-far solution.
# Boundary handling: Candidates are clipped to the feasible box using bounds
# read from func.lower/func.upper or func.bounds.lb/ub.
# Budget strategy: Uses a strict evaluation counter; in each call, performs an
# initial evaluation at a reasonable starting point (midpoint), then runs as many
# full batches as possible with the remaining budget.
# Closest known influences: Lightweight variants of evolution strategies / CMA-style
# ideas (best-so-far with Gaussian sampling and step-size adaptation), simplified
# for robustness and compactness.
# Novelty or unusual aspects: Uses deterministic initialization at the midpoint
# plus a small set of random restarts derived from sigma scaling; includes adaptive
# batch sizing and strict budget accounting.
# Failure modes: If the objective is very noisy or the optimum lies near
# boundaries with steep gradients, clipping may reduce efficiency; with extremely
# small budgets, the method may rely primarily on its initial evaluations.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim

        lb, ub = self._read_bounds(func, dim)
        # Ensure valid bounds shape and ordering
        lb = np.asarray(lb, dtype=float).reshape(dim)
        ub = np.asarray(ub, dtype=float).reshape(dim)
        if np.any(~np.isfinite(lb)) or np.any(~np.isfinite(ub)):
            raise ValueError("Bounds must be finite.")
        if np.any(ub < lb):
            raise ValueError("Upper bounds must be >= lower bounds.")

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        # Start from midpoint (robust deterministic baseline).
        x_best = clip((lb + ub) * 0.5)
        evals_used = 0
        y_best = self._eval_min(func, x_best)
        evals_used += 1

        # If budget is too small, return immediately.
        if evals_used >= self.budget:
            return x_best, y_best

        # Step-size initialization: use box scale, but avoid being too small.
        box_span = ub - lb
        # If some dims have zero span, keep sigma at least small but not harmful.
        scale = np.where(box_span > 0, box_span, 1.0)
        sigma = 0.3 * np.mean(scale)
        sigma = float(max(sigma, 1e-12))

        # Adaptive batch size: larger in higher dims, capped for budget.
        # Typical values: min(16, 4+dim//2) but ensure at least 1.
        base_bs = 4 + dim // 2
        batch_size = int(min(16, max(1, base_bs)))

        # Small random "spread" initialization around midpoint (budget-aware).
        # This helps when the midpoint is far from optimum.
        # We evaluate only if budget allows.
        remaining = self.budget - evals_used
        if remaining > 0:
            k0 = min(batch_size, remaining)
            # Spread candidates with slightly larger initial sigma
            sigma0 = min(1.0, 1.5) * sigma
            for _ in range(k0 - 1):
                pass  # just to keep code structured; actual loop below

            for i in range(k0):
                # Use different noise scales to diversify early search.
                # (deterministic randomness comes from global numpy seed set by harness)
                noise = np.random.randn(dim)
                x_cand = clip(x_best + sigma0 * noise)
                y_cand = self._eval_min(func, x_cand)
                evals_used += 1
                if y_cand < y_best:
                    y_best = y_cand
                    x_best = x_cand
                if evals_used >= self.budget:
                    return x_best, y_best

        # Main optimization loop
        # Simple success-based adaptation of sigma.
        # We run batches until budget is exhausted.
        # sigma is bounded to avoid stagnation or exploding too much.
        sigma_min = 1e-12
        sigma_max = 0.5 * float(np.max(scale)) + 1e-6

        while evals_used < self.budget:
            remaining = self.budget - evals_used
            k = min(batch_size, remaining)
            if k <= 0:
                break

            # Sample candidates around current best
            # (Gaussian steps; we clip into bounds).
            # We evaluate sequentially (standard-library only).
            best_in_batch_x = None
            best_in_batch_y = y_best
            improved = False

            # Candidate generation and selection: evaluate all k candidates,
            # keep the best; then optionally update global best.
            # We bias slightly towards smaller steps by varying sigma per candidate.
            for j in range(k):
                # Occasionally increase step length to encourage escape.
                # More likely when we fail to improve in earlier batches.
                if not improved and (j == 0):
                    local_sigma = sigma * 1.25
                else:
                    # Slightly vary sigma to diversify
                    local_sigma = sigma * (0.85 + 0.3 * np.random.rand())
                noise = np.random.randn(dim)
                x_cand = clip(x_best + local_sigma * noise)
                y_cand = self._eval_min(func, x_cand)
                evals_used += 1

                if y_cand < best_in_batch_y:
                    best_in_batch_y = y_cand
                    best_in_batch_x = x_cand
                    improved = True

                if evals_used >= self.budget:
                    break

            # Replacement
            if improved and best_in_batch_x is not None:
                x_best = best_in_batch_x
                y_best = best_in_batch_y
                # Exploit: shrink sigma after success.
                sigma *= 0.82
            else:
                # Exploration/escape: enlarge sigma (but bounded).
                sigma *= 1.12

            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # Optional: if sigma becomes extremely small, re-inject slight diversity
            # by increasing sigma modestly (within bounds) to avoid stagnation.
            if sigma <= sigma_min * 10:
                sigma = min(sigma_max, sigma * 5.0)

        return x_best, y_best

    def _eval_min(self, func, x):
        # Objective is minimization; we assume func(x) returns scalar.
        y = func(x)
        # Convert numpy scalar to python float for stable comparisons.
        if isinstance(y, (np.generic,)):
            y = float(y)
        return float(y)

    def _read_bounds(self, func, dim):
        # Supports:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
            return lb, ub

        if hasattr(func, "bounds") and func.bounds is not None:
            b = func.bounds
            # try lb/ub first, then lower/upper fallbacks
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return b.lb, b.ub
            if hasattr(b, "lower") and hasattr(b, "upper"):
                return b.lower, b.upper

        raise AttributeError(
            "Could not read bounds from func. Expected func.lower/func.upper or func.bounds.lb/func.bounds.ub."
        )
