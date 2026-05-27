# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, budget-aware black-box minimizer using a mixture of
# randomized restarts, local Gaussian sampling, and a simple coordinate-wise contraction
# strategy. It maintains the best-so-far solution and refines a local search around it.
# Search state: Tracks best_x, best_y, current evaluation count, a per-dimension step
# scale (sigma), and a small archive of elite points to stabilize selection.
# Candidate generation: Each iteration samples a population of candidates by combining
# (1) local Gaussian perturbations around best_x, (2) occasional “directional”
# perturbations using differences between elites, and (3) uniform sampling for renewed
# exploration when progress stalls.
# Selection and replacement: Evaluates all candidates within the remaining budget, then
# selects the best candidate as the new best; optionally updates the elite archive.
# Adaptation: Step size sigma is increased when no improvement occurs (to escape) and
# decreased when improvements are found (to focus). A simple coordinate-wise contraction
# also tightens sigma after repeated non-improvement.
# Exploration mechanisms: Uniform random points in the domain are used at start and
# periodically if progress stalls.
# Exploitation mechanisms: Gaussian perturbations around best_x plus low-rank directional
# moves based on elite differences drive exploitation.
# Boundary handling: Candidates are clipped to the provided bounds (or bounds inferred
# from func.lower/upper). This keeps feasibility without extra function calls.
# Budget strategy: Uses an internal counter and never evaluates beyond the provided budget.
# Populations are sized to fit the remaining evaluations. The algorithm stops immediately
# when budget is exhausted.
# Closest known influences: Combines ideas reminiscent of evolution strategies (population
# sampling + step adaptation) and trust-region-like shrinking/expanding behavior.
# Novelty or unusual aspects: Uses a small elite-difference-based directional proposal to
# create non-axis-aligned moves while staying simple and budget-aware.
# Failure modes: If the objective is extremely noisy or deceptive, the simple adaptation
# may oscillate; clipping can also bias search near boundaries. With very tight budgets,
# performance may rely mostly on initial random sampling.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        n = self.dim
        if n <= 0 or self.budget <= 0:
            # No meaningful search possible; return empty/zeros.
            x0 = np.zeros(max(0, n), dtype=float)
            y0 = float(func(x0)) if self.budget > 0 else np.inf
            return x0, y0

        # ---- Read bounds from func ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = func.bounds.lb
            ub = func.bounds.ub

        if lb is None or ub is None:
            # Fallback: no bounds provided. Use a conservative default box.
            # (Clipping becomes no-op if bounds are infinite; we avoid non-finite by choosing [-1,1].)
            lb = -1.0
            ub = 1.0

        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        if lb.ndim == 0:
            lb = np.full(n, lb, dtype=float)
        if ub.ndim == 0:
            ub = np.full(n, ub, dtype=float)

        # Make sure shape matches.
        if lb.shape[0] != n or ub.shape[0] != n:
            raise ValueError("Bounds must be scalars or arrays of length dim.")

        # Handle any accidental reversed bounds.
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo
        # If span is zero for a dimension, keep it fixed.
        span = np.where(span > 0, span, 1.0)

        eval_count = 0

        def clamp(x):
            return np.minimum(hi, np.maximum(lo, x))

        def f(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                # Hard stop: do not exceed budget.
                return np.inf
            x = np.asarray(x, dtype=float)
            x = clamp(x)
            y = float(func(x))
            eval_count += 1
            return y

        # ---- Budget-aware population size ----
        # Keep small populations for robustness; fit remaining budget.
        max_pop = max(2, min(12, n * 2 + 2))
        # Initialize step scale based on the domain size.
        base_sigma = 0.25 * span
        # If spans are tiny, avoid sigma=0.
        base_sigma = np.where(base_sigma > 0, base_sigma, 0.1)

        # ---- Initialize: random sampling to find a decent starting point ----
        # Ensure at least one evaluation.
        init_k = min(self.budget, min(10, max(1, n + 1)))
        # Use a simple strategy: start with uniform points, optionally include center.
        candidates = []
        for _ in range(init_k):
            r = np.random.random(n)
            x = lo + r * (hi - lo)
            candidates.append(x)

        # Also try the center if it doesn't consume extra evaluations (we can include it by adding
        # it to candidates but respecting init_k).
        if init_k < self.budget:
            center = (lo + hi) / 2.0
            candidates[0] = center

        best_x = None
        best_y = np.inf
        elite = []  # list of (y, x) sorted ascending by y

        for i in range(len(candidates)):
            y = f(candidates[i])
            if y < best_y:
                best_y = y
                best_x = clamp(np.asarray(candidates[i], dtype=float))
            # Maintain small elite archive
            elite.append((y, clamp(np.asarray(candidates[i], dtype=float))))
        elite.sort(key=lambda t: t[0])
        elite = elite[:min(5, len(elite))]

        if best_x is None:
            best_x = (lo + hi) / 2.0

        # Track progress for adaptation.
        no_improve_iters = 0
        best_global = best_y

        # Main loop: each iteration evaluates a batch within remaining budget.
        # Use a conservative upper bound on iterations to reduce overhead.
        # Each loop consumes at least 1 evaluation (since pop size >= 1).
        while eval_count < self.budget:
            remaining = self.budget - eval_count
            pop = min(max_pop, remaining)

            improved = False
            new_points = []

            # Decide exploration vs exploitation.
            # If no improvement recently, increase exploration probability.
            # Also add occasional uniform points.
            stall = no_improve_iters >= 3
            explore_prob = 0.25 + (0.25 if stall else 0.0)  # between 0.25 and 0.5

            # Current sigma scales with domain.
            sigma = base_sigma * (0.5 ** (no_improve_iters / 3.0))
            if stall:
                sigma = sigma * 1.6

            # Directional proposal using elite differences.
            dir_vec = np.zeros(n, dtype=float)
            if len(elite) >= 2:
                # Pick two elite points and use their difference as a direction.
                # Using a small coefficient helps keep steps bounded.
                a = elite[0][1]
                b = elite[-1][1]
                dir_vec = b - a
                # Normalize if possible
                dn = np.linalg.norm(dir_vec)
                if dn > 1e-12:
                    dir_vec = dir_vec / dn

            for _ in range(pop):
                r = np.random.random()
                if r < explore_prob:
                    # Uniform exploration.
                    rx = np.random.random(n)
                    x = lo + rx * (hi - lo)
                else:
                    # Exploit around best with Gaussian noise.
                    # Use diagonal sigma with per-dim scaling.
                    z = np.random.normal(size=n)
                    x = best_x + z * sigma

                    # Occasionally add a directional move.
                    if len(elite) >= 2 and np.random.random() < 0.35:
                        # Step along elite direction with a random magnitude.
                        mag = np.random.normal(0.0, 1.0) * 0.5 * np.mean(span)
                        x = x + dir_vec * mag

                    x = clamp(x)

                new_points.append(x)

            # Evaluate batch and update best.
            ys = []
            for x in new_points:
                y = f(x)
                ys.append(y)
                if eval_count >= self.budget:
                    break

            # Find best in batch among evaluated points.
            # If budget was exhausted mid-batch, ys may be shorter. Determine evaluated count.
            # (eval_count is already updated inside f.)
            # Compute best among all results collected so far in this batch call.
            for x, y in zip(new_points[: len(ys)], ys):
                if y < best_y:
                    best_y = y
                    best_x = clamp(np.asarray(x, dtype=float))
                    improved = True
                    best_global = min(best_global, best_y)

            # Update elite archive
            for x, y in zip(new_points[: len(ys)], ys):
                elite.append((y, clamp(np.asarray(x, dtype=float))))
            elite.sort(key=lambda t: t[0])
            elite = elite[:min(7, len(elite))]

            if improved:
                no_improve_iters = 0
                # Contract base_sigma a bit after improvement.
                base_sigma = base_sigma * 0.85
            else:
                no_improve_iters += 1
                # Expand base_sigma slightly after consecutive non-improvement.
                base_sigma = base_sigma * 1.05

            # Extra safeguard: prevent sigma from becoming too large.
            # Cap sigma relative to the domain span.
            max_sigma = 0.8 * span
            base_sigma = np.minimum(base_sigma, np.where(max_sigma > 0, max_sigma, base_sigma))

            # If base_sigma becomes extremely small and we're not improving, force exploration by
            # resetting sigma moderately.
            if no_improve_iters >= 6:
                base_sigma = np.maximum(base_sigma, 0.15 * span)
                no_improve_iters = 6  # keep stall high to increase exploration

            # If we are exactly out of budget, exit cleanly.
            if eval_count >= self.budget:
                break

        return np.asarray(best_x, dtype=float), float(best_y)
