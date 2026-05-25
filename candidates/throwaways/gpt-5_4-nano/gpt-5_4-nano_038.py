# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy for continuous problems.
# It combines (1) a small population of candidate points, (2) rank-based selection, and
# (3) adaptive Gaussian search around the best-so-far point. The algorithm is robust
# across dimensions and uses only NumPy and the Python standard library.
# Search state: Maintains a population of points X and their objective values Y, plus the
# incumbent best_x/best_y found so far. Also maintains an adaptive step size sigma.
# Candidate generation: Each iteration samples new candidates by adding Gaussian noise
# to a “base” point (the current best), with the noise scaled by sigma. In addition,
# it occasionally uses differential-style perturbations from the population to diversify.
# Selection and replacement: Evaluates all candidates without exceeding the overall budget,
# then keeps the best individuals for the next generation using partial selection.
# Adaptation: Updates sigma based on the observed improvement rate: it decreases when
# improvement is rare and increases slightly when progress is good.
# Exploration mechanisms: Gaussian perturbations with a sigma schedule plus occasional
# population-difference moves encourage exploration beyond local neighborhoods.
# Exploitation mechanisms: Strong focus on sampling around the incumbent best point
# with shrinking sigma as the search progresses.
# Boundary handling: Samples are clipped to the provided bounds after mutation.
# Budget strategy: Uses a fixed generation size but truncates the final generation to
# never exceed the provided evaluation budget. Calls the objective exactly as many
# times as allowed.
# Closest known influences: Inspired by evolution strategies / (mu+lambda) ES with
# adaptive step size, combined with rank-based selection and occasional differential
# variation.
# Novelty or unusual aspects: Uses a simple “improvement-rate” controller for sigma
# that is budget-aware, plus a lightweight diversity move based on population differences.
# Failure modes: If the objective is extremely noisy or bounds are very tight, adaptive
# sigma may stagnate; clipping may lead to many repeated boundary points.
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
        # --- Bounds retrieval (as required) ---
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = func.bounds.lb
            ub = func.bounds.ub
        else:
            raise AttributeError("Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub")

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size == 1:
            lb = np.full(self.dim, float(lb), dtype=float)
        if ub.size == 1:
            ub = np.full(self.dim, float(ub), dtype=float)

        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError("Bounds must match dimension or be scalar")

        # Ensure lb <= ub (robustness)
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)

        rng = np.random

        # --- Evaluation helper (budget-aware) ---
        evals = 0

        def eval_one(x):
            nonlocal evals
            if evals >= self.budget:
                # This should not happen if we respect remaining budget carefully.
                # Still, be safe and avoid extra evaluations.
                return np.inf
            y = func(x)
            evals += 1
            return float(y)

        # --- Initialization: small population sampled uniformly in bounds ---
        dim = self.dim
        remaining = self.budget

        # Choose population size; keep compact but effective across dims.
        # For higher dims, increase slightly but cap to avoid too many evaluations.
        # Each generation will evaluate pop_size points (except first incumbent).
        pop_size = int(np.clip(4 + dim // 5, 6, 24))
        pop_size = min(pop_size, self.budget)  # can't exceed budget

        # Create initial population
        X = lo + (hi - lo) * rng.rand(pop_size, dim)
        Y = np.empty(pop_size, dtype=float)

        # Evaluate initial population respecting budget
        k = min(pop_size, self.budget)
        for i in range(k):
            Y[i] = eval_one(X[i])

        if k < pop_size:
            X = X[:k]
            Y = Y[:k]
            pop_size = k

        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])

        # Step size: start as fraction of box size (avoid zero)
        box = hi - lo
        box_scale = float(np.mean(box)) if np.all(box > 0) else float(np.mean(np.maximum(box, 1e-12)))
        sigma = 0.3 * box_scale / max(1.0, np.sqrt(dim))
        sigma = max(sigma, 1e-12)

        # Generation loop
        # We build new candidates around the best, with diversity moves occasionally.
        # We stop when budget exhausted.
        # Keep a rough iteration budget to adapt sigma each step.
        gen = 0
        improvements = 0
        prev_best = best_y

        while evals < self.budget:
            gen += 1
            remaining = self.budget - evals

            # Build offspring count for this generation, truncated to remaining budget.
            # Evaluate at most pop_size new candidates each generation.
            lam = min(pop_size, remaining)

            # Rank-based parents: take top half of current population as a pool
            order = np.argsort(Y)
            elite_n = max(1, len(order) // 2)
            elite_idx = order[:elite_n]
            elite = X[elite_idx]

            # Always sample around best for exploitation.
            base = best_x

            # Create candidates
            C = np.empty((lam, dim), dtype=float)

            # Diversity probability increases slightly in early generations
            diversity_p = 0.15 + 0.1 * np.exp(-gen / 10)

            for j in range(lam):
                if rng.rand() < diversity_p and len(elite) >= 2:
                    # Differential-style move: base + sigma * (a - b) + noise
                    a, b = elite[rng.randint(elite.shape[0])], elite[rng.randint(elite.shape[0])]
                    diff = a - b
                    # Add directional mutation with slightly larger variance
                    noise = rng.randn(dim) * (sigma * 0.7)
                    step = sigma * diff / np.sqrt(max(1e-12, np.mean(diff * diff)))
                    x = base + noise + step
                else:
                    # Pure Gaussian around best
                    x = base + rng.randn(dim) * sigma

                # Boundary handling: clip
                # (For minimization, clipping is usually acceptable for this benchmark.)
                x = np.minimum(np.maximum(x, lo), hi)
                C[j] = x

            # Evaluate candidates
            YC = np.empty(lam, dtype=float)
            for j in range(lam):
                YC[j] = eval_one(C[j])
                if YC[j] < best_y:
                    best_y = float(YC[j])
                    best_x = C[j].copy()

            # Update improvement statistics and sigma
            if best_y < prev_best - 1e-15:
                improvements += 1
                prev_best = best_y

            # Sigma adaptation: based on recent improvement frequency
            # (simple but effective, avoids complicated ES step-size estimation).
            # Over the whole run, improvements <= number of generations.
            # Use a decaying target improvement probability.
            gen_done = gen
            target = 0.12 * np.exp(-gen_done / 50) + 0.06  # small target
            rate = (improvements / gen_done) if gen_done > 0 else 0.0

            if rate < target * 0.8:
                sigma *= 0.85  # stagnation -> shrink
            elif rate > target * 1.2:
                sigma *= 1.08  # success -> expand slightly
            else:
                sigma *= 0.95  # neutral drift

            # Prevent sigma collapse too much (still allow convergence)
            min_sigma = 1e-12
            max_sigma = 0.8 * max(box_scale, 1e-12)
            sigma = float(np.clip(sigma, min_sigma, max_sigma))

            # Selection and replacement: (mu + lambda) style keeping best pop_size
            # Combine and select best individuals for next generation.
            X = np.vstack([X, C])
            Y = np.concatenate([Y, YC])

            order = np.argsort(Y)
            keep = min(pop_size, len(order))
            order = order[:keep]
            X = X[order]
            Y = Y[order]

        return best_x, best_y
