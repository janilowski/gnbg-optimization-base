# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy combining
# random sampling, coordinate-wise local search, and occasional global
# restarts using a shrinking Gaussian step size.
# Search state: Maintains the best-so-far point and value, along with a
# current step scale (sigma) controlling how far candidates are perturbed.
# Candidate generation: Creates new points by (1) uniform random sampling,
# (2) Gaussian perturbations around the current best, and (3) coordinate
# perturbations (finite-difference-like exploration) for local refinement.
# Selection and replacement: Each candidate is evaluated; if it improves the
# best value, it replaces the incumbent. A separate "radius" of improvement
# tracking is used implicitly via adaptive sigma.
# Adaptation: Sigma shrinks when improvements occur, and grows slightly after
# stagnation (triggered by lack of improvements).
# Exploration mechanisms: Initial quasi-global uniform sampling, plus random
# restarts when progress stalls, help escape local minima.
# Exploitation mechanisms: Coordinate-wise directional probing around the best
# uses smaller step sizes to refine locally.
# Boundary handling: All candidate points are clipped to the provided bounds.
# Budget strategy: Uses a strict evaluation counter; never evaluates more
# than the provided budget.
# Closest known influences: Inspired by simple ES/CMA-like step adaptation and
# derivative-free coordinate search, but kept intentionally lightweight.
# Novelty or unusual aspects: Combines Gaussian sampling with adaptive
# coordinate probing in a single budgeted loop, with restarts governed by
# stagnation.
# Failure modes: If the objective is extremely noisy or highly deceptive,
# coordinate probing and step adaptation may waste evaluations; also, very tight
# bounds can reduce effective movement and slow convergence.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from func ----
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
            raise AttributeError(
                "Objective must provide bounds via func.lower/func.upper "
                "or func.bounds.lb/func.bounds.ub."
            )
        if lb.shape == ():  # scalar bounds
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        lb = np.asarray(lb, dtype=float).reshape(dim)
        ub = np.asarray(ub, dtype=float).reshape(dim)

        if np.any(ub <= lb):
            raise ValueError("Invalid bounds: require ub > lb elementwise.")

        span = ub - lb
        span_norm = float(np.linalg.norm(span))
        if span_norm <= 0:
            # Degenerate case: all points identical under clipping.
            x0 = np.clip(lb, lb, ub)
            return x0.copy(), float(func(x0))

        # ---- Budgeted evaluation helper ----
        evals = 0
        best_x = None
        best_y = None

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        def evaluate(x):
            nonlocal evals, best_x, best_y
            x = clip(np.asarray(x, dtype=float))
            y = float(func(x))
            evals += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # ---- Initial sampling ----
        # Use a mix: uniform sampling + a starting incumbent near the center.
        # Remaining budget drives local refinement and restarts.
        remaining = budget

        # Ensure we don't over-sample when budget is small.
        n_uniform = min(10 + 2 * dim, remaining)
        if n_uniform > 0:
            for _ in range(n_uniform):
                r = np.random.random(dim)
                x = lb + r * span
                evaluate(x)

        # Add a center point if budget allows (helps when incumbents are bad).
        if evals < budget:
            x = lb + 0.5 * span
            evaluate(x)

        if best_x is None:  # should not happen
            best_x = lb + 0.5 * span
            best_y = float(func(best_x))
            evals = 1

        # ---- Step size adaptation and restart control ----
        # Start sigma as a fraction of the bounds span.
        sigma = 0.25 * (span_norm / np.sqrt(dim))
        sigma = max(sigma, 1e-12)

        best_snapshot = best_y
        no_improve_steps = 0

        # Coordinate probing step (smaller than sigma).
        coord_scale = 1e-3 + 1e-2 / (1.0 + dim)
        delta = coord_scale * sigma

        # How often to attempt global restart
        # (based on stagnation and remaining budget).
        max_stagnation = max(10, int(0.08 * budget) + 1)

        # ---- Main loop ----
        # Each iteration consumes evaluations; keep it simple:
        # - Gaussian candidates around best for exploitation/global-ish.
        # - Coordinate probing on a few axes for local refinement.
        # - Occasional restart to re-seed search.
        while evals < budget:
            remaining = budget - evals

            # Decide exploration intensity.
            # When stagnating, explore more and occasionally restart.
            improving = best_y < best_snapshot - 1e-16
            if improving:
                best_snapshot = best_y
                no_improve_steps = 0
                # shrink sigma to focus locally after improvement
                sigma *= 0.85
                sigma = max(sigma, 1e-12)
                delta = coord_scale * sigma
            else:
                no_improve_steps += 1
                # grow sigma slightly during stagnation (broader search)
                sigma *= 1.03
                sigma = min(sigma, span_norm)
                delta = coord_scale * sigma

            # Restart condition
            if no_improve_steps >= max_stagnation and remaining > dim:
                # Reset around a random point, but keep best so far.
                # This is exploration to escape stagnation.
                # (Do not reset best_x/best_y.)
                no_improve_steps = 0
                sigma = max(sigma * 0.6, 1e-12)

                # Sample one new base point uniformly and evaluate it.
                r = np.random.random(dim)
                base = lb + r * span
                evaluate(base)

                # Continue with exploitation from new base:
                # try a couple Gaussian moves around best_x and base.
                n_g = min(3 + dim // 2, remaining)
                for _ in range(n_g):
                    # With some probability use base; otherwise best_x
                    center = base if np.random.rand() < 0.5 else best_x
                    noise = np.random.randn(dim)
                    x = center + sigma * noise
                    evaluate(x)

                continue

            # Exploitation: Gaussian candidates around current best
            # Use a batch size based on remaining budget.
            n_gauss = min(5 + dim, remaining)

            for _ in range(n_gauss):
                # Orthogonality isn't needed; random directions are enough.
                noise = np.random.randn(dim)
                # A small amount of uniform jitter helps avoid exact duplicates.
                jitter = (np.random.random(dim) - 0.5) * 1e-6 * span
                x = best_x + sigma * noise + jitter
                evaluate(x)
                if evals >= budget:
                    break

            if evals >= budget:
                break

            remaining = budget - evals
            if remaining <= 0:
                break

            # Coordinate probing: try moves along a few coordinates.
            # Choose coordinates randomly but bias toward larger spans.
            # This resembles derivative-free local search.
            if remaining > 2 * min(dim, 8):
                k = min(dim, 8 + dim // 4)
                idxs = np.random.choice(dim, size=k, replace=False)

                # Try +delta and -delta for each selected coordinate until budget runs out.
                for j in idxs:
                    if evals >= budget:
                        break
                    x_plus = best_x.copy()
                    x_minus = best_x.copy()

                    # Scale delta per-coordinate based on span to remain meaningful.
                    # Using span ratio keeps steps consistent across axes.
                    per = delta * (span[j] / (span_norm / np.sqrt(dim)))
                    per = float(per)
                    if per <= 0:
                        per = delta

                    x_plus[j] = x_plus[j] + per
                    x_minus[j] = x_minus[j] - per

                    evaluate(x_plus)
                    if evals >= budget:
                        break
                    evaluate(x_minus)

            # If we're close to the limit, do no extra work (loop condition handles it).

        # Ensure best_x exists
        if best_x is None:
            # Fallback (should not happen)
            best_x = lb + 0.5 * span
            best_y = float(func(best_x))

        return best_x, best_y
