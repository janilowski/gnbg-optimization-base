import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box optimizer for minimization
# using a restarted evolution-strategy-like approach with coordinate-wise crossover,
# Gaussian sampling, and periodic restart/step-size adaptation.
# Search state: Maintains a current best point x_best, its objective value y_best,
# a step-size sigma controlling exploration radius, and an archive of previously seen
# candidates for diversity.
# Candidate generation: Each iteration samples an offspring set by perturbing
# x_best in multiple random directions. Offspring are produced by combining
# Gaussian mutations with occasional coordinate masking (crossover-like behavior).
# Additionally, the algorithm includes “reflection” and “mean” candidates to help
# escape flat regions.
# Selection and replacement: Among evaluated offspring, the algorithm selects the best
# (lowest objective) and replaces x_best if improvement is found.
# Adaptation: Uses a simple 1/5-success style heuristic: if a sufficient fraction
# of offspring improve, sigma is increased/decreased conservatively; otherwise sigma
# is reduced to focus search.
# Exploration mechanisms: Gaussian perturbations, random coordinate masks, and
# occasional restart when progress stalls (or sigma becomes too small).
# Exploitation mechanisms: Perturbations centered at current best and shrinking sigma
# after unsuccessful iterations.
# Boundary handling: Uses clipped candidate positions to satisfy provided bounds.
# This keeps feasibility without extra evaluations.
# Budget strategy: Strictly tracks and never exceeds the evaluation budget. The number
# of offspring per iteration is chosen based on remaining evaluations.
# Closest known influences: Inspired by simple evolution strategies and CMA-lite ideas
# (step-size adaptation and restart), but kept intentionally minimal and black-box.
# Novelty or unusual aspects: Uses a small reflection/center candidate set plus
# coordinate-masked mutation to increase robustness across dimensions.
# Failure modes: Can stagnate on highly non-smooth or deceptive functions, especially
# when the optimum lies near bounds; restarts and sigma adaptation mitigate but do
# not guarantee avoidance.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return a deterministic point at center of bounds if possible.
            lower, upper = self._get_bounds(func, dim)
            x0 = (lower + upper) / 2.0
            return x0, float("inf")

        lower, upper = self._get_bounds(func, dim)

        evals = 0

        # Helper: evaluate with strict budget accounting.
        def eval_one(x):
            nonlocal evals
            if evals >= budget:
                return None
            y = float(func(x))
            evals += 1
            return y

        # Initialize best by sampling a single starting point inside bounds.
        # The harness sets numpy's seed, so this is reproducible per run.
        x_best = lower + (upper - lower) * np.random.rand(dim)
        y_best = eval_one(x_best)
        if y_best is None:
            return x_best, float("inf")

        # Step size: fraction of typical range.
        span = np.maximum(upper - lower, 1e-12)
        sigma = 0.25 * float(np.mean(span))
        sigma = max(sigma, 1e-12)

        # Simple stagnation counters for restarts.
        best_y = y_best
        no_improve_iters = 0

        # A tiny archive for diversity (stores a few recent points).
        archive = [x_best.copy()]

        # Remaining-evaluation-driven loop.
        # Each outer loop evaluates k offspring + (sometimes) one extra candidate.
        while evals < budget:
            remaining = budget - evals

            # Choose offspring count so we don't exceed the budget.
            # Keep k small early and allow at most dim+2 (but never too large for remaining).
            k = min(8 + dim // 2, remaining)
            # Ensure at least 2 candidates per iteration when possible.
            k = max(2, k) if remaining >= 2 else remaining

            improved = False
            improvements = 0

            # Prepare offspring candidates.
            # Offspring are generated around x_best using Gaussian steps.
            # Add coordinate-masked crossover to increase variety.
            X = np.empty((k, dim), dtype=float)

            # Coordinate mask probability: lower when dim is large to keep steps meaningful.
            p_mask = min(0.5, 1.0 / np.sqrt(max(dim, 1))) if dim > 0 else 0.0
            # Reflection candidate probability.
            do_reflect = (np.random.rand() < 0.25)

            for i in range(k):
                z1 = np.random.randn(dim)
                z2 = np.random.randn(dim)

                # Coordinate-masked combination:
                # Use z1 most of the time, occasionally replace coordinates with z2.
                if p_mask > 0:
                    mask = (np.random.rand(dim) < p_mask)
                    z = np.where(mask, z2, z1)
                else:
                    z = z1

                # Mutation step.
                x = x_best + sigma * z

                # Occasional directional bias: half of the time, add a small mean shift
                # towards a randomly archived point to help exploitation around valleys.
                if archive and (np.random.rand() < 0.4):
                    a = archive[np.random.randint(len(archive))]
                    x = 0.7 * x + 0.3 * a

                # Boundary handling by clipping.
                x = np.clip(x, lower, upper)
                X[i] = x

            # Optionally add a reflection around a random archive point (counts toward budget).
            y_candidates = []
            xs = []

            # Evaluate offspring set.
            for i in range(k):
                if evals >= budget:
                    break
                y = eval_one(X[i])
                if y is None:
                    break
                xs.append(X[i])
                y_candidates.append(y)
                if y < y_best:
                    improved = True
                    improvements += 1

            # Add at most one extra evaluation for reflection, if we still have budget.
            if do_reflect and evals < budget:
                if archive:
                    a = archive[np.random.randint(len(archive))]
                else:
                    a = x_best
                x_ref = x_best + (x_best - a)  # reflect away from archived point
                x_ref = np.clip(x_ref, lower, upper)
                y_ref = eval_one(x_ref)
                if y_ref is not None:
                    xs.append(x_ref)
                    y_candidates.append(y_ref)
                    if y_ref < y_best:
                        improved = True
                        improvements += 1

            if not y_candidates:
                break

            # Selection: pick best among evaluated.
            idx_best_local = int(np.argmin(y_candidates))
            x_new = xs[idx_best_local]
            y_new = float(y_candidates[idx_best_local])

            # Replacement and sigma adaptation (simple success-rate heuristic).
            # If many offspring improved, slightly increase sigma; otherwise decrease.
            # This stabilizes and avoids premature collapse.
            success_rate = improvements / max(1, len(y_candidates))

            if y_new < y_best:
                x_best = x_new
                y_best = y_new
                best_y = y_best
                no_improve_iters = 0
            else:
                no_improve_iters += 1

            # Adapt sigma: shrink on low success, expand moderately on high success.
            # Clamp sigma to a reasonable range.
            mean_span = float(np.mean(span))
            min_sigma = 1e-12
            max_sigma = 0.75 * mean_span + 1e-12

            if success_rate >= 0.25:
                sigma *= 1.15
            elif success_rate <= 0.1:
                sigma *= 0.82
            else:
                sigma *= 0.92

            sigma = float(np.clip(sigma, min_sigma, max_sigma))

            # Archive maintenance: store improved or periodic points for diversity.
            if improved:
                archive.append(x_best.copy())
            else:
                # Keep some diversity even if no global improvement.
                if archive and (len(archive) < 6) and np.random.rand() < 0.35:
                    archive.append(xs[idx_best_local].copy())
            if len(archive) > 6:
                # Drop oldest.
                archive = archive[-6:]

            # Restart mechanism on stagnation: reinitialize around a new random point,
            # but keep current best as reference.
            # Uses remaining budget indirectly by not restarting too aggressively.
            if no_improve_iters >= 6 and evals < budget:
                # Hard restart: sample new center, reset sigma to fraction of span.
                x_center = lower + (upper - lower) * np.random.rand(dim)
                # Evaluate x_center only if budget allows; otherwise just shift x_best.
                if evals < budget:
                    y_center = eval_one(x_center)
                    if y_center is not None and y_center < y_best:
                        x_best = x_center
                        y_best = y_center
                        best_y = y_best
                else:
                    x_best = x_center

                archive = [x_best.copy()]
                sigma = 0.25 * float(np.mean(span))
                sigma = max(sigma, 1e-12)
                no_improve_iters = 0

        return x_best, float(y_best)

    @staticmethod
    def _get_bounds(func, dim):
        # Bounds can be provided via func.lower/func.upper or func.bounds.lb/ub.
        lower = None
        upper = None

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            # Support either attributes lb/ub or lower/upper inside bounds.
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lower = np.asarray(b.lb, dtype=float)
                upper = np.asarray(b.ub, dtype=float)
            elif hasattr(b, "lower") and hasattr(b, "upper"):
                lower = np.asarray(b.lower, dtype=float)
                upper = np.asarray(b.upper, dtype=float)

        if lower is None or upper is None:
            # Fallback: assume [-1, 1]^dim if not provided.
            lower = -np.ones(dim, dtype=float)
            upper = np.ones(dim, dtype=float)
        else:
            # Broadcast/validate shapes.
            lower = np.asarray(lower, dtype=float).reshape(-1)
            upper = np.asarray(upper, dtype=float).reshape(-1)
            if lower.size != dim or upper.size != dim:
                # Try broadcasting a scalar bound.
                if lower.size == 1:
                    lower = np.full(dim, float(lower[0]), dtype=float)
                if upper.size == 1:
                    upper = np.full(dim, float(upper[0]), dtype=float)
            lower = lower.reshape(dim)
            upper = upper.reshape(dim)

        # Ensure valid bounds order.
        swap = lower > upper
        if np.any(swap):
            lo = lower.copy()
            lower = upper
            upper = lo
        # Avoid degenerate bounds leading to zero span everywhere; still allow exact feasibility.
        return lower, upper
