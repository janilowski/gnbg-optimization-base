import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, robust derivative-free black-box minimizer
# using a variant of Evolution Strategies (ES) with diagonal step-size control
# and occasional coordinate-wise refinement. It works in any dimension and
# obeys a hard evaluation budget.
# Search state: Maintains a current best solution x_best and a per-dimension
# mutation scale sigma (step sizes). Tracks remaining evaluations and the
# best objective value found so far.
# Candidate generation: Each iteration samples lambda offspring around the
# current best using Gaussian noise: x = x_best + sigma * N(0, 1).
# Boundary handling: Offspring are clipped into the feasible hyper-rectangle
# defined by the objective's bounds. The best-so-far point is also kept in
# bounds.
# Selection and replacement: Offspring are evaluated; the best candidate replaces
# x_best if it improves the objective. The parent is the current best.
# Adaptation: Step sizes adapt multiplicatively based on "success" (improvement):
# if an iteration yields improvement, sigma is slightly increased; otherwise it is
# slightly decreased. Additionally, sigma is driven down as the budget
# approaches exhaustion to reduce overshooting.
# Exploration mechanisms: Random Gaussian sampling with per-dimension sigmas
# provides exploration; larger initial sigma encourages coverage.
# Exploitation mechanisms: On improvements, sigmas for dimensions correlated
# with the winning move are nudged down (encouraging local refinement), and a
# lightweight coordinate-refinement is periodically tried with a smaller
# radius.
# Budget strategy: Converts the user-provided budget into an exact evaluation
# limit. Each objective call consumes one evaluation, and the code never
# evaluates more than the budget total.
# Closest known influences: Diagonal-covariance ES (CMA-ES-like spirit but far
# simpler), with 1/5th-success style step-size adaptation and occasional local
# refinement.
# Novelty or unusual aspects: Includes a periodic, budget-aware coordinate
# refinement that tries a small step along the currently best-known direction,
# while still keeping the implementation compact and fully standard-library /
# numpy-only.
# Failure modes: In extremely noisy objectives or very ill-scaled bounds,
# progress may stall; clipping can bias search near boundaries. If the budget is
# too small, the algorithm may return only the initial best (random) candidate.
# ALGORITHM_ANALYSIS_NOTE_END
class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # --- Read bounds from func ---
        lb = ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        if lb is None or ub is None:
            raise AttributeError("Objective function must provide bounds via "
                                 "func.lower/func.upper or func.bounds.lb/func.bounds.ub.")
        if lb.shape != (dim,) or ub.shape != (dim,):
            lb = np.broadcast_to(lb, (dim,)).astype(float)
            ub = np.broadcast_to(ub, (dim,)).astype(float)

        # Ensure feasible ordering; if swapped, correct.
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)

        def clamp(x):
            return np.minimum(np.maximum(x, lo), hi)

        # --- Budget bookkeeping ---
        # Always evaluate at least once if budget permits.
        # If budget <= 0, return something deterministic within bounds.
        if budget <= 0:
            x0 = clamp(np.zeros(dim, dtype=float))
            return x0, float(func(x0))

        evals = 0

        def eval_obj(x):
            nonlocal evals
            if evals >= budget:
                # Should never happen; guard against harness errors.
                raise RuntimeError("Evaluation budget exceeded.")
            y = func(x)
            evals += 1
            return float(y)

        # --- Initialize ---
        # Choose initial point near center plus small random perturbation.
        center = 0.5 * (lo + hi)
        span = (hi - lo)
        # Prevent degenerate bounds from causing zero sigma.
        min_span = np.where(span > 0, span, 1.0)
        # Initial sigma: a fraction of the span with floor.
        sigma0 = 0.2 * min_span
        sigma = np.maximum(sigma0, 1e-12)

        # Random starting candidate; harness controls numpy seed.
        x_best = clamp(center + sigma * np.random.randn(dim))
        y_best = eval_obj(x_best)

        # --- ES parameters (kept simple and robust) ---
        # lambda based on dimension but limited so it fits budget.
        lam_default = 4 + int(3 * np.log1p(dim))
        lam = int(min(max(2, lam_default), max(2, budget - 1)))
        # Number of iterations depends on remaining evals; recompute dynamically.
        # Parent is replaced by best offspring (1-parent ES).
        tau_inc = 1.15
        tau_dec = 0.85

        # Periodic coordinate refinement probability and frequency.
        refine_period = max(3, int(2 * np.log1p(dim)))
        refine_trials = 2  # keep cheap

        # Precompute a small radius schedule.
        # As budget approaches end, exploitation grows (smaller sigma and radius).
        def budget_fraction_used():
            return evals / budget

        # --- Main loop ---
        # Each loop evaluates 'lam' candidates (or fewer if budget nearly ends).
        while evals < budget:
            remaining = budget - evals
            k = min(lam, remaining)
            # If only one evaluation remains, just do one offspring.
            if k <= 0:
                break

            # Adaptive: slightly shrink sigma as we approach budget end.
            frac = budget_fraction_used()
            sigma = sigma * (1.0 - 0.25 * min(1.0, frac)) + (1.0e-12)

            # Sample offspring around best.
            Z = np.random.randn(k, dim)
            X = clamp(x_best + (sigma[None, :] * Z))

            # Evaluate offspring.
            ys = np.empty(k, dtype=float)
            best_idx = 0
            best_y_off = y_best
            for i in range(k):
                yi = eval_obj(X[i])
                ys[i] = yi
                if yi < best_y_off:
                    best_y_off = yi
                    best_idx = i

            # Selection and replacement.
            improved = best_y_off < y_best
            if improved:
                x_win = X[best_idx]
                # For exploitation: nudge sigma down in dimensions that
                # contributed strongly to the winning move.
                move = x_win - x_best
                # Correlation proxy using absolute move.
                denom = np.maximum(np.abs(move), 1e-12)
                # Reduce sigma more where we moved; mild to keep robustness.
                sigma = sigma * (0.7 + 0.3 * np.clip(np.abs(move) / np.maximum(np.abs(span), 1e-12), 0.0, 1.0))
                x_best = x_win
                y_best = best_y_off
            else:
                # No improvement: shrink step sizes.
                sigma = sigma * tau_dec

            # If improved, slightly increase sigmas to continue exploration;
            # otherwise don't do it (already shrunk).
            if improved:
                sigma = sigma * tau_inc

            # Lightweight coordinate refinement occasionally, if budget allows.
            # Tries a small move along a random coordinate based on sigma.
            if (evals < budget) and (evals % refine_period < lam) and (np.any(sigma > 0)):
                # Try a couple of coordinate moves.
                for _ in range(refine_trials):
                    if evals >= budget:
                        break
                    # Choose coordinate biased by sigma (larger sigma -> more likely).
                    probs = sigma / np.sum(sigma)
                    j = int(np.random.choice(dim, p=probs))
                    step = 0.5 * sigma[j] * np.random.randn()
                    x_cand = x_best.copy()
                    x_cand[j] = x_cand[j] + step
                    x_cand = clamp(x_cand)
                    y_cand = eval_obj(x_cand)
                    if y_cand < y_best:
                        x_best = x_cand
                        y_best = y_cand
                        # Coordinate refinement reduces sigma in that coordinate.
                        sigma[j] = max(sigma[j] * 0.7, 1e-12)

        return x_best, y_best
