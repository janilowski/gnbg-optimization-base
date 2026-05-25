# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm based on a
# restartable Evolution Strategy (ES) with covariance-free step adaptation.
# It searches in a bounded continuous space by generating a small batch of
# Gaussian mutations around the current best solution, selecting the best
# offspring, and adapting mutation strength using simple success rules.
# Search state: Maintains current best point x_best and its objective value
# y_best, plus a current global mutation scale sigma and a per-restart
# best-so-far. Tracks the remaining evaluation budget to ensure no more than
# the provided limit are used.
# Candidate generation: Each iteration samples lambda offspring as
# x = clip(x_best + sigma * N(0,1), bounds). The offspring are evaluated
# and the best one replaces x_best (elitist selection).
# Selection and replacement: Uses (μ+1)-style behavior: the single best among
# the offspring plus the current best is selected as the new incumbent.
# Adaptation: Uses a 1/5-success style rule: if the best offspring improves
# on the incumbent, sigma is increased; otherwise sigma is decreased.
# Exploration mechanisms: Restarts with a reinitialized mean near the
# incumbent when stagnation is detected or when sigma becomes too small.
# Exploitation mechanisms: As improvements occur, sigma shrinks/centers on
# the best point, focusing sampling around promising regions.
# Boundary handling: Candidate points are clipped to the provided bounds
# to enforce feasibility without extra evaluations.
# Budget strategy: Divides the total budget into a fixed number of iterations
# with a small batch size; each iteration consumes exactly lambda evaluations.
# It also includes a final partial batch if remaining evaluations are fewer
# than lambda, ensuring strict budget compliance.
# Closest known influences: Inspired by classic evolution strategies,
# elitist (μ+1) selection, and simple success-rate step adaptation.
# Novelty or unusual aspects: Uses covariance-free global sigma adaptation
# with bounded clipping and lightweight restart/stagnation logic for
# robustness across dimensions without extra parameters.
# Failure modes: In extremely rugged/noisy objectives, clipping and simplistic
# adaptation may lead to premature shrinking or slow progress. In very tight
# bounds, random restarts may be ineffective.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # --- Bounds handling ---
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub.")

        if lb.shape == ():  # scalar bounds
            lb = np.full(self.dim, float(lb))
        if ub.shape == ():
            ub = np.full(self.dim, float(ub))

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError("Bounds dimension mismatch with dim.")

        if np.any(ub <= lb):
            raise ValueError("Invalid bounds: each upper bound must be > lower bound.")

        # --- Objective wrapper (counts evaluations) ---
        remaining = self.budget
        evals_used = 0

        def eval_f(x):
            nonlocal remaining, evals_used
            if remaining <= 0:
                raise RuntimeError("Evaluation budget exceeded.")
            remaining -= 1
            evals_used += 1
            return float(func(np.asarray(x, dtype=float)))

        # --- Initialization ---
        # Start at a random point inside bounds; harness controls randomness via numpy seed.
        x_best = lb + (ub - lb) * np.random.rand(self.dim)
        y_best = eval_f(x_best)

        # Initial sigma: proportional to box size, but not too small.
        box = ub - lb
        sigma = 0.3 * float(np.mean(box))
        if sigma <= 0:
            sigma = 1.0

        # Batch size: small enough to allow multiple steps, scalable with dimension.
        # Typical ES uses ~4-10. We cap to preserve budget usage.
        lam = int(max(4, min(16, 2 + 2 * (self.dim >= 10) + self.dim // 20)))
        lam = max(2, lam)

        # Determine number of full iterations; leftover handled later.
        # Each full iteration uses lam evals. We already used 1 eval for x_best.
        remaining_after_seed = remaining
        full_iters = remaining_after_seed // lam
        # If budget is tiny, full_iters may be 0; handle via final partial batch loop.

        # Success rule counters
        consecutive_fail = 0
        stagnation_limit = int(max(5, min(25, 2 + self.dim // 2)))
        min_sigma = 1e-12 * float(np.mean(box)) if np.mean(box) > 0 else 1e-12

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        # --- Main loop ---
        it = 0
        while remaining > 0 and (it < full_iters or remaining < lam):
            it += 1
            # Decide how many offspring we can evaluate this iteration.
            k = lam if remaining >= lam else remaining
            if k <= 0:
                break

            # Candidate generation: Gaussian perturbations around incumbent.
            # Using vectorized generation for speed and cleanliness.
            noise = np.random.randn(k, self.dim)
            X = x_best[None, :] + sigma * noise
            X = np.clip(X, lb[None, :], ub[None, :])

            # Evaluate and select best offspring.
            best_idx = 0
            best_val = None
            for i in range(k):
                yi = eval_f(X[i])
                if best_val is None or yi < best_val:
                    best_val = yi
                    best_idx = i

            x_candidate = X[best_idx]
            y_candidate = float(best_val)

            # Selection and replacement
            if y_candidate < y_best:
                x_best = x_candidate
                y_best = y_candidate
                consecutive_fail = 0
                # If improving, increase step slightly to explore more.
                sigma = sigma * 1.2
            else:
                consecutive_fail += 1
                # If not improving, decrease step to exploit locally.
                sigma = sigma / 1.2

            # Adaptation bounds on sigma
            if sigma < min_sigma:
                sigma = min_sigma
            if sigma > 0.5 * float(np.max(box)):
                sigma = 0.5 * float(np.max(box))

            # Exploration via lightweight restart if stagnating
            if consecutive_fail >= stagnation_limit:
                consecutive_fail = 0
                # Reinitialize mean near incumbent using a fraction of box size.
                # This keeps some exploitation while escaping plateaus.
                radius = 0.5 * float(np.mean(box)) * (0.5 + 0.5 * np.random.rand())
                x_best = clip(x_best + radius * np.random.randn(self.dim))
                y_best = eval_f(x_best)
                sigma = 0.3 * float(np.mean(box))

            # Stop if exactly budget consumed
            if remaining <= 0:
                break

        # Safety: ensure we respected budget (eval_f would have thrown otherwise)
        return np.asarray(x_best, dtype=float), float(y_best)
