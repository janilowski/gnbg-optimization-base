# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# using an evolution-strategy style loop with a simple, robust population
# approach. It maintains a population of candidate points sampled from a
# Gaussian distribution around the current best solution, then uses selection
# pressure to update both the best point and the sampling scale.
# Search state: The algorithm tracks the current best solution (best_x)
# and its objective value (best_y), plus the sampling step-size (sigma)
# and population size (mu/lam derived from the problem dimension).
# Candidate generation: At each iteration, it samples offspring by drawing
# z ~ N(0, I) and setting x = clip(best_x + sigma * z, bounds). It also
# evaluates a small set of "refinement" points around the best to help
# exploitation.
# Selection and replacement: Offspring are evaluated, and the best among
# them is selected to update best_x/best_y. The algorithm uses the
# quantile-best subset to estimate how good improvements are, enabling
# adaptive step-size control.
# Adaptation: If improvements are observed, sigma is reduced (to exploit);
# if not, sigma is increased (to explore). The adaptation is based on
# comparing the median of top candidates to the current best.
# Exploration mechanisms: Occasional larger sampling via increased sigma
# (on stagnation) and diversity from Gaussian sampling around best_x.
# Exploitation mechanisms: Always centers sampling on best_x and uses
# additional local perturbations with a smaller radius.
# Boundary handling: All candidate points are clipped to the provided
# search bounds.
# Budget strategy: The algorithm strictly respects the evaluation budget by
# tracking remaining evaluations and only generating/evaluating up to that
# limit. It uses a dynamic number of iterations inferred from remaining budget.
# Closest known influences: Inspired by basic (1+λ)-ES / (μ,λ)-ES patterns
# and CMA-lite step-size adaptation, but kept intentionally simple and
# dimension-robust.
# Novelty or unusual aspects: The refinement sampling and sigma update are
# designed to be safe under tight budgets and to remain stable even for
# small dimensions.
# Failure modes: With very tight budgets or extremely flat/noisy objectives,
# the algorithm may not find meaningful improvements; sigma adaptation helps
# but cannot guarantee progress. If bounds are degenerate (lb==ub), it
# will effectively evaluate the fixed point.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")

    def __call__(self, func):
        # ---- Extract bounds (prefer func.lower/upper, else func.bounds.lb/ub) ----
        lb = ub = None

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        if lb is None or ub is None:
            raise AttributeError(
                "Objective must provide bounds via (lower, upper) or func.bounds.lb/ub."
            )

        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            lb = np.reshape(lb, (self.dim,))
            ub = np.reshape(ub, (self.dim,))

        # Ensure proper ordering and finite values
        lb = np.minimum(lb, ub)
        ub = np.maximum(lb, ub)

        # If degenerate bounds, the search space collapses to one point.
        if np.all(lb == ub):
            x = lb.copy()
            y = func(x)
            return x, y

        # ---- Helper: clip into bounds ----
        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        # ---- Budget bookkeeping ----
        evals_max = self.budget
        evals_used = 0

        def evaluate(x):
            nonlocal evals_used
            if evals_used >= evals_max:
                # Should never happen if logic is correct; keep safe.
                raise RuntimeError("Evaluation budget exceeded.")
            y = func(x)
            evals_used += 1
            return y

        # ---- Initialize best solution ----
        # Start from a reasonable point: midpoint with small random jitter.
        mid = (lb + ub) / 2.0
        span = (ub - lb)
        # Avoid zero span affecting sigma.
        span_nonzero = np.where(span == 0, 1.0, span)

        # Initial sigma: fraction of typical range
        sigma = 0.25 * np.max(span_nonzero)

        rng = np.random

        # Evaluate a small initial set to get a better starting point early.
        # Keep within budget.
        init_count = min(5, evals_max)
        best_x = None
        best_y = None
        for i in range(init_count):
            if i == 0:
                x = clip(mid)
            else:
                x = clip(mid + (rng.randn(self.dim) * 0.1 * np.max(span_nonzero)))
            y = evaluate(x)
            if best_y is None or y < best_y:
                best_x, best_y = x, y

        # If budget exhausted early
        if evals_used >= evals_max:
            return best_x, best_y

        # ---- Choose population sizes (compact, adaptive to dim & budget) ----
        # λ grows gently with dim, but never exceeds remaining evaluations.
        # Use mu as a small fraction of λ for improvement statistics.
        # These choices aim to keep the number of iterations small when budget is tight.
        remaining = evals_max - evals_used
        lam = int(np.clip(4 + self.dim, 4, 18))  # cap to stay compact
        lam = min(lam, remaining)
        mu = max(1, lam // 3)

        # ---- Main loop ----
        # Each iteration consumes `take` evaluations <= lam.
        # We decide a dynamic number of iterations based on remaining budget.
        while evals_used < evals_max:
            remaining = evals_max - evals_used
            take = min(lam, remaining)
            # number of refinement points: small, only when budget allows
            refine = 0 if take < 2 else min(2, take // 4)

            # --- Candidate generation: Gaussian around best_x ---
            # Generate all offspring; evaluate in one batch loop (no extra memory needed).
            # We'll keep their (x, y) temporarily to select best and top-mu.
            candidates_x = []
            candidates_y = []

            # Scale sigma down a bit as optimization progresses to stabilize.
            # This is mild; adaptation below will dominate.
            progress = evals_used / evals_max
            sigma_iter = sigma * (1.0 - 0.15 * progress)

            for _ in range(take - refine):
                z = rng.randn(self.dim)
                x = clip(best_x + sigma_iter * z)
                y = evaluate(x)
                candidates_x.append(x)
                candidates_y.append(y)

            # --- Local refinement: smaller perturbations near best_x ---
            for _ in range(refine):
                z = rng.randn(self.dim)
                x = clip(best_x + 0.3 * sigma_iter * z)
                y = evaluate(x)
                candidates_x.append(x)
                candidates_y.append(y)

            candidates_y = np.asarray(candidates_y, dtype=float)

            # Update best with current generation's best
            gen_best_idx = int(np.argmin(candidates_y))
            gen_best_x = candidates_x[gen_best_idx]
            gen_best_y = float(candidates_y[gen_best_idx])
            if gen_best_y < best_y:
                best_y = gen_best_y
                best_x = gen_best_x

            # --- Selection statistics for adaptation ---
            # Consider top mu candidates (lowest values) and compare their
            # median to current best_y.
            # If top candidates improve sufficiently, reduce sigma; else increase.
            sorted_idx = np.argsort(candidates_y)
            top_idx = sorted_idx[:mu] if mu <= len(sorted_idx) else sorted_idx
            top_vals = candidates_y[top_idx]

            # Improvement ratio: how much better the typical top is than best_y.
            # If best_y is already very small, ratio may underflow; use absolute.
            # We use a scale based on span.
            scale = float(np.max(span_nonzero))
            eps = 1e-12

            # Define "improved" as top median being lower than best_y by a
            # relative amount, but clipped to remain safe.
            top_median = float(np.median(top_vals))
            rel_improve = (best_y - top_median) / (abs(best_y) + eps)

            # sigma adaptation policy:
            # - If there is noticeable improvement in this generation, shrink sigma.
            # - If no meaningful improvement, enlarge sigma to explore.
            # - Additionally, if the generation best is far from top median,
            #   the landscape might be noisy; keep sigma from shrinking too much.
            gen_best_val = gen_best_y
            spread = float(np.percentile(top_vals, 90) - np.percentile(top_vals, 10) + eps)

            # Thresholds tuned for robustness
            if rel_improve > 0.02:
                sigma *= 0.85
            elif rel_improve > 0.005:
                sigma *= 0.95
            else:
                sigma *= 1.15

            # Keep sigma within reasonable bounds based on search span.
            sigma_min = 1e-12 * scale
            sigma_max = 0.5 * scale
            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # If we can't improve, and sigma becomes tiny, try a final exploratory jump
            # (only if budget remains).
            if evals_used < evals_max and sigma <= sigma_min * 10 and np.isfinite(best_y):
                # Random restart around best with broader sigma just once per near-stagnation.
                # This consumes no extra evaluations here, so we rely on next loop's samples.
                sigma = max(sigma_max * 0.3, sigma * 3.0)

            # Early exit if budget is done (loop condition covers it, but keep explicit)
            if evals_used >= evals_max:
                break

        return np.asarray(best_x, dtype=float), float(best_y)
