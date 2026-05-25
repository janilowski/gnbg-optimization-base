# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimization strategy using a
# mixture of global sampling and local coordinate searches. It maintains a population of
# candidate points, evaluates them with an elitist selection scheme, and periodically
# shrinks/expands a step size based on observed improvements.
#
# Search state: A current best solution (best_x, best_y), a small population of
# points with associated fitness, and a current per-dimension step scale (sigma)
# controlling how far new candidates are perturbed.
#
# Candidate generation: New candidates are created by (1) global exploration via
# uniform sampling inside the bounds, and (2) local exploitation by coordinate-wise
# perturbations of the current best (and some population members) with steps drawn
# from a normal distribution scaled by sigma and the current bound range.
#
# Selection and replacement: At each iteration, all candidates are evaluated, then only
# the best few are kept (elitism). The best overall is tracked across the whole run.
#
# Adaptation: sigma shrinks when improvements are found (finer local search) and grows
# slightly when progress stalls (to escape local minima). The adaptation is driven by
# a simple patience counter and the improvement rate.
#
# Exploration mechanisms: Uses a fraction of remaining budget for purely random
# points to re-inject diversity; also increases sigma after stalls.
#
# Exploitation mechanisms: Performs coordinate perturbations around the current best,
# followed by local refinement attempts by probing positive/negative directions.
#
# Boundary handling: Candidate points are clipped to the valid bounds after mutation,
# ensuring feasibility without additional evaluations.
#
# Budget strategy: The algorithm consumes evaluations conservatively: the number of
# evaluations per iteration is chosen so the total never exceeds the provided budget.
# It can early-stop if the budget is exhausted.
#
# Closest known influences: Inspired by common patterns in black-box optimizers:
# elitist evolutionary sampling, coordinate perturbation, and step-size adaptation
# (e.g., CMA-ES-like adaptation at a simplified level).
#
# Novelty or unusual aspects: Uses an explicit remaining-budget-aware iteration loop,
# combines coordinate probes and population-based mutations, and includes robust
# bounds extraction to work with different func/bounds layouts.
#
# Failure modes: If the objective is very noisy or deceptive, sigma adaptation may
# oscillate; clipping to bounds can cause candidates to pile up near edges; for
# extremely ill-conditioned problems, a purely coordinate-based probe may converge
# slower than more sophisticated rotation-aware methods.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        n_evals_max = self.budget
        if n_evals_max <= 0:
            # No budget: return something within bounds if possible; otherwise zeros.
            lb, ub = self._get_bounds(func)
            x0 = np.clip(np.zeros(dim), lb, ub)
            return x0, float(func(x0)) if n_evals_max > 0 else (x0, np.inf)

        lb, ub = self._get_bounds(func)
        lb = np.asarray(lb, dtype=float).reshape(dim)
        ub = np.asarray(ub, dtype=float).reshape(dim)
        span = ub - lb
        # Handle degenerate dimensions with zero span.
        span_safe = np.where(span > 0, span, 1.0)

        rng = np.random  # harness seeds numpy RNG

        evals = 0

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_one(x):
            nonlocal evals
            if evals >= n_evals_max:
                # Should not happen if budget management is correct.
                return np.inf
            x = clip(x)
            y = float(func(x))
            evals += 1
            return y

        # Choose population size based on dimension (small to keep evals under budget).
        # Ensure at least 2, at most 2*dim+1 (reasonable upper bound).
        pop = int(np.clip(6 + dim // 2, 2, max(2, 2 * dim + 1)))
        # Iterations will use batches; keep per-iter candidates <= pop.
        # We'll evaluate in batches of up to pop to amortize adaptation.
        batch = min(pop, n_evals_max)

        # Initialization: start with random points plus one center/edge-informed point.
        center = lb + 0.5 * span
        # Initial sigma: fraction of span, robust if span is 0 in some dims.
        sigma = 0.3 * span_safe

        # Generate initial population points.
        X = np.empty((batch, dim), dtype=float)
        # First point: center
        X[0] = clip(center)
        # Remaining: uniform within bounds
        if batch > 1:
            u = rng.uniform(size=(batch - 1, dim))
            X[1:] = lb + u * span_safe * (span_safe / span_safe)  # same shape
            # If span is zero in a dimension, lb + u*0 should remain lb; use original span.
            X[1:] = lb + u * span

        Y = np.empty(batch, dtype=float)
        best_idx = 0
        best_y = np.inf
        best_x = X[0].copy()

        for i in range(batch):
            Y[i] = eval_one(X[i])
            if Y[i] < best_y:
                best_y = Y[i]
                best_x = X[i].copy()
                best_idx = i

        # Elitist memory: keep a small set of best individuals.
        elite_size = max(2, min(6, pop // 2))
        # Sort by fitness and take elite
        order = np.argsort(Y)
        elites_X = X[order[:elite_size]].copy()
        elites_Y = Y[order[:elite_size]].copy()

        # Simple progress tracking for sigma adaptation.
        prev_best = best_y
        patience = 0
        max_patience = 4

        # Main loop: each iteration creates new candidates around elites and occasionally random.
        # Remaining budget aware: number of evaluations per iteration adapts.
        while evals < n_evals_max:
            remaining = n_evals_max - evals

            # Determine how many candidates to evaluate this iteration.
            # Keep it at most pop, and at most remaining.
            k = min(pop, remaining)
            candidates = np.empty((k, dim), dtype=float)

            # Exploration ratio decreases as we approach budget end.
            # Also increase exploration when stalled.
            tfrac = evals / max(1, n_evals_max)
            explore_ratio = 0.35 * (1.0 - tfrac)
            if patience >= 1:
                explore_ratio += 0.15
            explore_ratio = float(np.clip(explore_ratio, 0.15, 0.6))

            n_explore = int(np.round(k * explore_ratio))
            n_explore = int(np.clip(n_explore, 0, k))
            n_exploit = k - n_explore

            # 1) Global exploration: uniform points inside bounds.
            if n_explore > 0:
                u = rng.uniform(size=(n_explore, dim))
                candidates[:n_explore] = lb + u * span

            # 2) Exploitation: coordinate perturbations around elites/best.
            # Generate perturbations using random normal steps, scaled by sigma and bound span.
            # Coordinate probes: try both + and - along a coordinate with a probability.
            for j in range(n_explore, k):
                # Choose a reference point: either best or one elite
                if rng.rand() < 0.6:
                    x_ref = best_x
                else:
                    idx = rng.randint(0, elite_size)
                    x_ref = elites_X[idx]

                # Coordinate selection biased towards larger sigma components (uniform otherwise).
                coord = rng.randint(0, dim)

                # Base step vector: mostly single-coordinate move, with small multivariate noise.
                step = np.zeros(dim, dtype=float)

                # Main coordinate step (random sign and magnitude)
                # Magnitude drawn from N(0,1) then scaled; ensure some nonzero effect.
                z = rng.normal()
                mag = sigma[coord] * (0.5 + 0.5 * abs(z))
                step[coord] = mag * (1.0 if rng.rand() < 0.5 else -1.0)

                # Add small noise in other dims to avoid strictly axis-aligned behavior.
                # Scale noise by 0.1*sigma.
                noise = rng.normal(size=dim) * (0.10 * sigma)
                # Keep noise small compared to main step.
                if dim > 1:
                    noise[coord] = 0.0
                step += noise

                candidates[j] = x_ref + step

            # Evaluate candidates (budget-safe).
            Yc = np.empty(k, dtype=float)
            # Track best within this batch too (slightly reduces sorting cost).
            batch_best_y = np.inf
            batch_best_x = None

            for i in range(k):
                Yc[i] = eval_one(candidates[i])
                if Yc[i] < batch_best_y:
                    batch_best_y = Yc[i]
                    batch_best_x = candidates[i].copy()

                # Early stop if budget exactly reached.
                if evals >= n_evals_max and i < k - 1:
                    # If budget unexpectedly depleted during loop, trim.
                    Yc = Yc[: i + 1]
                    candidates = candidates[: i + 1]
                    k = i + 1
                    break

            # Update global best.
            if batch_best_y < best_y:
                best_y = batch_best_y
                best_x = batch_best_x.copy()

            # Update elites from combined set: old elites + current candidates (elitist).
            # This keeps memory small and stable.
            combined_X = np.vstack([elites_X, candidates[:k]])
            combined_Y = np.hstack([elites_Y, Yc[:k]])

            order = np.argsort(combined_Y)
            elite_take = min(elite_size, combined_X.shape[0])
            elites_X = combined_X[order[:elite_take]].copy()
            elites_Y = combined_Y[order[:elite_take]].copy()

            # Adapt sigma based on improvement.
            improvement = prev_best - best_y
            if improvement > 1e-12 * (abs(prev_best) + 1.0):
                # Improvement: shrink sigma to focus search.
                sigma *= 0.85
                prev_best = best_y
                patience = 0

                # Occasionally do a cheap local refinement probe around best:
                # probe one coordinate both directions if budget allows.
                if evals < n_evals_max and rng.rand() < 0.35:
                    coord = rng.randint(0, dim)
                    # Step size for probe
                    s = sigma[coord] * 0.6
                    x1 = best_x.copy()
                    x2 = best_x.copy()
                    x1[coord] += s
                    x2[coord] -= s
                    y1 = eval_one(x1)
                    if y1 < best_y:
                        best_y = y1
                        best_x = clip(x1).copy()
                        # Update elites best
                    if evals < n_evals_max:
                        y2 = eval_one(x2)
                        if y2 < best_y:
                            best_y = y2
                            best_x = clip(x2).copy()
                    # Refresh elites including best_x if it improves.
                    # (Simple: rebuild from existing + best_x.)
                    elites_X = np.vstack([elites_X, best_x[None, :]])
                    elites_Y = np.hstack([elites_Y, np.array([best_y])])
                    order = np.argsort(elites_Y)
                    elite_take = min(elite_size, elites_X.shape[0])
                    elites_X = elites_X[order[:elite_take]]
                    elites_Y = elites_Y[order[:elite_take]]
            else:
                patience += 1
                # If stalled, expand sigma to encourage escape.
                if patience >= max_patience:
                    sigma *= 1.15
                    # Also keep sigma bounded by span to avoid excessively large steps.
                    sigma = np.minimum(sigma, 0.9 * span_safe + 1e-12)
                    patience = 0

            # Keep sigma within reasonable range to avoid stagnation.
            sigma_floor = 1e-6 * span_safe
            sigma = np.maximum(sigma, sigma_floor)
            sigma_ceil = 1.0 * span_safe + 1e-12
            sigma = np.minimum(sigma, sigma_ceil)

            # If span is zero for some dims, set sigma to 0 there to prevent unnecessary moves.
            sigma = np.where(span > 0, sigma, 0.0)

        return best_x, best_y

    def _get_bounds(self, func):
        """
        Bounds can be provided in multiple standard formats:
        - func.lower / func.upper
        - func.bounds.lb / func.bounds.ub
        """
        # Case 1: direct attributes
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return np.asarray(func.lower, dtype=float), np.asarray(func.upper, dtype=float)

        # Case 2: func.bounds.lb / func.bounds.ub
        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return np.asarray(b.lb, dtype=float), np.asarray(b.ub, dtype=float)

        raise AttributeError(
            "Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
        )
