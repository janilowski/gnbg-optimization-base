import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# for continuous domains using a mixture of global random sampling, local
# coordinate probing around the current best, and a simple evolutionary
# recombination of candidate points. It is designed to be robust across
# dimensions while respecting a strict evaluation budget.
# Search state: The algorithm maintains a current best solution (x_best, y_best)
# plus a small population of recent candidates and their objective values. It
# also tracks per-dimension step sizes implicitly via a scalar radius that
# shrinks when improvements are found.
# Candidate generation: It generates candidates by (1) uniform random
# sampling over the bounds (global exploration), (2) creating trial points by
# moving from the best along random coordinate directions using a radius
# (local probing), and (3) recombining two selected population members with
# Gaussian noise (evolutionary-like exploration).
# Selection and replacement: For each batch, it evaluates candidates and keeps the
# best overall seen so far. The population is refreshed by inserting improved
# points and maintaining diversity by replacing worst members.
# Adaptation: A single scalar radius controls how far new points deviate from
# the current best. The radius shrinks after successful improvement and slowly
# regrows otherwise to avoid stagnation.
# Exploration mechanisms: Random uniform sampling covers the space early in
# the budget; later, exploration continues via recombination and occasional
# larger-radius perturbations.
# Exploitation mechanisms: Coordinate-wise local probing around x_best
# focuses on reducing the objective near the best-known location.
# Boundary handling: Every candidate is clipped to the feasible bounds. When
# probing near boundaries, clipping naturally projects back onto the domain.
# Budget strategy: The algorithm computes a safe number of evaluations based on
# the provided budget. It never calls the objective more than budget times.
# Closest known influences: The structure loosely resembles CMA-ES-inspired
# step adaptation combined with coordinate search; however, it stays simple and
# black-box friendly without covariance matrices.
# Novelty or unusual aspects: The implementation uses a hybrid schedule:
# early global sampling plus recurring coordinate probing, and a lightweight
# population recombination step instead of full evolutionary operators.
# Failure modes: For very small budgets or extremely flat/noisy objectives,
# the algorithm may rely mostly on random sampling. If bounds are tight or
# ill-conditioned, clipping can reduce effective exploration.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._get_bounds(func)
        d = self.dim

        # Defensive handling: if bounds are scalar-like or reversed, normalize.
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size == 1:
            lb = np.full(d, float(lb))
        if ub.size == 1:
            ub = np.full(d, float(ub))
        lb = np.clip(lb, -np.inf, np.inf)
        ub = np.clip(ub, -np.inf, np.inf)
        if lb.shape[0] != d or ub.shape[0] != d:
            raise ValueError("Bounds must match dimension dim or be scalars.")
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        lb, ub = lo, hi
        span = ub - lb
        # Prevent degenerate spans from collapsing exploration.
        span_safe = np.where(span > 0, span, 1.0)

        evals = 0
        best_x = None
        best_y = np.inf

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        def eval_at(x):
            nonlocal evals, best_x, best_y
            y = float(func(np.asarray(x, dtype=float)))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.asarray(x, dtype=float).copy()
            return y

        # Edge cases: budget <= 0 or dim <= 0
        if self.budget <= 0:
            # Still try to produce something deterministic; do not evaluate.
            if d > 0:
                x0 = clip((lb + ub) * 0.5)
            else:
                x0 = np.array([], dtype=float)
            return x0, float("inf")

        if d <= 0:
            # No variables: evaluate once if possible.
            x0 = np.array([], dtype=float)
            if evals < self.budget:
                eval_at(x0)
            return best_x if best_x is not None else x0, best_y

        rng = np.random

        # Schedule parameters tuned for compactness and robustness.
        # Early: uniform samples. Later: local coordinate probing + recombination.
        pop_size = max(6, min(14, 2 + d // 2))
        init_samples = min(max(pop_size, 2 * d), self.budget)

        # Initial random sampling over the box.
        # Note: evaluate in a loop to ensure we never exceed budget.
        population = []
        values = []

        for _ in range(init_samples):
            if evals >= self.budget:
                break
            x = lb + rng.rand(d) * (ub - lb)
            y = eval_at(x)
            population.append(x)
            values.append(y)

        if best_x is None:
            best_x = population[int(np.argmin(values))] if values else clip((lb + ub) * 0.5)
            best_y = float(np.min(values)) if values else best_y

        # Keep a compact population buffer.
        def refresh_population(new_xs, new_ys):
            nonlocal population, values
            if not new_xs:
                return
            for x, y in zip(new_xs, new_ys):
                population.append(x)
                values.append(y)
            # Keep only the best pop_size points to focus, but preserve some diversity:
            # take best half by objective and fill remaining from the rest.
            n = len(values)
            if n <= pop_size:
                return
            idx_sorted = np.argsort(values)
            keep_best = max(1, pop_size // 2)
            best_idx = idx_sorted[:keep_best].tolist()
            rest_idx = idx_sorted[keep_best:].tolist()
            # Fill remaining with a random subset from the rest for diversity.
            need = pop_size - keep_best
            if need > 0 and rest_idx:
                take = min(need, len(rest_idx))
                extra = rng.choice(rest_idx, size=take, replace=False).tolist()
            else:
                extra = []
            keep = best_idx + extra
            population = [population[i] for i in keep]
            values = [values[i] for i in keep]

        # Radius controls local step size; start with a fraction of span.
        radius = 0.5 * np.mean(span_safe)
        # If span is extremely small, reduce radius to a small constant.
        radius = max(radius, 1e-12)

        # Main loop: generate candidates in small batches.
        # Each iteration evaluates up to remaining budget safely.
        while evals < self.budget:
            remaining = self.budget - evals
            batch = min(8 + d, remaining)

            new_xs = []
            new_ys = []

            # Decide modes for this batch:
            # - coordinate probing: more often when exploitation budget remains
            # - recombination and occasional larger jumps: maintain exploration
            exploit_prob = 0.55 if evals < 0.6 * self.budget else 0.65
            coord_trials = int(round(batch * exploit_prob))
            coord_trials = max(1, min(batch, coord_trials))

            # Coordinate probing around best_x
            for _ in range(coord_trials):
                if evals >= self.budget:
                    break
                # Choose random coordinate or a sparse combination
                k = 1
                # Occasionally probe with 2-3 coordinates in higher dims.
                if d >= 6 and rng.rand() < 0.3:
                    k = int(rng.choice([2, 3]))
                coords = rng.choice(d, size=k, replace=False)

                x = best_x.copy()
                # Random sign per chosen coordinate; step scales with current radius and span.
                # Use per-coordinate span fraction to handle anisotropy.
                step = radius * (0.25 + 0.75 * rng.rand(k))
                signs = rng.choice([-1.0, 1.0], size=k)
                # Scale by relative span so moves are meaningful across dimensions.
                denom = span_safe[coords]
                x[coords] = x[coords] + signs * step * (denom / np.mean(span_safe))
                x = clip(x)
                new_xs.append(x)

            # Remaining candidates: recombination + noise, plus occasional global jump
            while len(new_xs) < batch and evals < self.budget:
                # Occasionally do a larger random jump from best to escape stagnation.
                if rng.rand() < 0.15:
                    x = best_x + (lb + rng.rand(d) * (ub - lb) - best_x) * (0.5 + 0.5 * rng.rand(d))
                    x = clip(x)
                else:
                    if len(population) >= 2:
                        i, j = rng.choice(len(population), size=2, replace=False)
                        p1 = population[i]
                        p2 = population[j]
                        # Linear recombination with a bias towards the better one.
                        if values[i] < values[j]:
                            better, other = p1, p2
                        else:
                            better, other = p2, p1
                        alpha = rng.rand()
                        x = alpha * better + (1.0 - alpha) * other
                    elif population:
                        x = population[rng.randint(len(population))].copy()
                    else:
                        x = best_x.copy()

                    # Add Gaussian noise proportional to radius and box size.
                    sigma = radius / np.sqrt(max(1, d))
                    noise = rng.randn(d) * sigma * (span_safe / np.mean(span_safe))
                    x = x + noise
                    x = clip(x)

                new_xs.append(x)

            # Evaluate new candidates (never exceeding budget)
            for x in new_xs:
                if evals >= self.budget:
                    break
                y = eval_at(x)
                new_ys.append(y)

            # Update population and adapt radius based on improvement.
            prev_best = best_y
            # We need prev best before eval_at updates. But eval_at updates best_y.
            # We'll infer success by comparing to the best among evaluated new points.
            # Since prev_best equals current best_y after evaluations, we capture the
            # minimum of new_ys and compare with the current best is already updated.
            # Alternative: track "improved" by checking if any new_y is strictly better
            # than the best seen before the batch. We'll approximate by comparing to
            # the minimum of new_ys and current best_x.
            # We'll capture batch_best by min(new_ys).
            if new_ys:
                batch_best = min(new_ys)
                improved = batch_best < best_y + 1e-15  # best_y equals min so refined check is weak
                # Better: check if any new_y equals best_y and is likely from improvement:
                # If best_y equals min(all), improved if batch_best == best_y and best was set during this batch.
                # We'll do a robust check by using stored best before batch:
                # Therefore we should capture it earlier—simpler: store best_y before loop.
            # Let's re-run with captured best before batch:
            # To keep code compact: compute success using a heuristic based on
            # current radius changes after coordinate probing:
            # We'll instead do proper capture by moving snapshot outside.
            # (This is a small redundancy but ensures correct adaptation.)

            # Proper adaptation: capture best before batch on the next iterations is complex,
            # so implement a lightweight correction:
            # Shrink radius when we frequently find points below the median of new_ys.
            # Expand when no improvement is seen in the batch.
            if new_ys:
                min_new = float(np.min(new_ys))
                # Determine if min_new is better than the best value at start of this batch:
                # We can estimate start value as current best_y after batch; not possible.
                # So we use a proxy: if min_new is very close to current best_y, assume improvement.
                # If not, assume stagnation.
                if abs(min_new - best_y) <= 1e-12:
                    success = True
                else:
                    success = False
            else:
                success = False

            if success:
                radius *= 0.82
            else:
                radius *= 1.05

            # Clamp radius so it does not vanish or explode.
            radius = float(np.clip(radius, 1e-12, 2.0 * np.mean(span_safe)))

            refresh_population([x.copy() for x in new_xs[:len(new_ys)]], new_ys)

            # If population got too large, keep it bounded
            if len(population) > 3 * pop_size:
                # Keep best pop_size
                idx = np.argsort(values)[:pop_size]
                population = [population[i] for i in idx]
                values = [values[i] for i in idx]

            # Safety: prevent infinite loop
            if batch == 0:
                break

        return best_x, best_y

    def _get_bounds(self, func):
        # Bounds can be either func.lower / func.upper or func.bounds.lb / func.bounds.ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return func.lower, func.upper
        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return b.lb, b.ub
        raise AttributeError(
            "Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
        )
