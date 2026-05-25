# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm for continuous
# domains using a derivative-free population strategy. It mixes global random
# sampling with local coordinate-wise refinement while adapting step sizes
# based on improvements.
# Search state: Maintains a population of candidate points, their objective
# values, the best-so-far solution, and a scalar step size per iteration.
# Candidate generation: Generates candidates by combining (1) uniform random
# samples within bounds and (2) Gaussian perturbations around the best point
# with occasional coordinate-wise probes. Step size shrinks when progress
# occurs and grows slightly when improvements stall.
# Selection and replacement: Each iteration evaluates a batch, then selects the
# top fraction of candidates to seed the next iteration, while retaining the
# current best. Rejected points are replaced by newly generated samples.
# Adaptation: Uses a success counter and compares improvements versus the
# previous best to adapt the mutation scale (sigma). If improvements happen,
# sigma decreases; if not, sigma increases slightly.
# Exploration mechanisms: Random points and broad Gaussian moves around the best
# are used early; later, exploration probability decreases and local probing
# becomes more prominent.
# Exploitation mechanisms: Coordinate-wise one-dimensional moves around the best
# point with adaptive per-coordinate step fractions, plus Gaussian exploitation
# centered at the best.
# Boundary handling: Uses a projection ("clipping") strategy that keeps all
# candidate coordinates within [lb, ub] after perturbations.
# Budget strategy: Never exceeds the evaluation budget by tracking remaining
# evaluations and allocating a safe per-iteration batch size. Stops as soon as
# the budget is exhausted.
# Closest known influences: Inspired by (1) evolution strategies / CMA-like ideas
# but simplified, and (2) coordinate descent-style local probing combined with
# population selection.
# Novelty or unusual aspects: Uses a hybrid of population-based sampling and
# explicit coordinate probes around the incumbent best, with simple sigma
# adaptation controlled by improvement success frequency.
# Failure modes: If the objective is extremely noisy or highly deceptive,
# adaptation may stall or over-shrink sigma, reducing exploration. The algorithm
# mitigates this via occasional random sampling and modest sigma growth.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from func ----
        lb = None
        ub = None

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            # Support b.lb / b.ub
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            else:
                # Fallback if bounds stored differently (best-effort)
                if hasattr(b, "lower") and hasattr(b, "upper"):
                    lb = np.asarray(b.lower, dtype=float)
                    ub = np.asarray(b.upper, dtype=float)

        if lb is None or ub is None:
            raise AttributeError(
                "func must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub"
            )

        lb = np.broadcast_to(lb, (dim,)).astype(float)
        ub = np.broadcast_to(ub, (dim,)).astype(float)
        if np.any(ub <= lb):
            raise ValueError("Invalid bounds: require ub > lb for all dimensions.")

        span = ub - lb
        # Prevent degenerate spans
        span = np.maximum(span, 1e-12)

        def project(x):
            return np.minimum(np.maximum(x, lb), ub)

        # ---- Budget-managed evaluation wrapper ----
        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                # Do not exceed budget; return +inf so it won't be selected.
                return float("inf")
            y = func(x)
            evals += 1
            return float(y)

        # ---- Choose initial sampling and population sizes ----
        # Population size depends on budget and dimension; keep modest for speed.
        # Ensure at least 2 to support selection, but never exceed budget.
        pop = int(np.clip(2 * dim, 6, max(6, budget)))
        # If budget is tiny, fallback gracefully.
        pop = max(2, min(pop, budget)) if budget > 0 else 0

        # If budget is 0, return any feasible point with +inf (no eval performed).
        if budget <= 0:
            x0 = project(lb + 0.5 * span)
            return x0, float("inf")

        # ---- Initial population: mix corners and random points ----
        # Use deterministic corner heuristics to give coverage without extra randomness.
        # Then fill remaining with uniform samples.
        corners = []
        if dim <= 20:
            # Create a few corners based on dimension grouping
            # (still deterministic, but covers extremes)
            for sgn in (-1.0, 1.0):
                # Alternate sign pattern across dimensions
                pattern = (np.arange(dim) % 2) * sgn
                x = lb + (pattern > 0).astype(float) * span
                corners.append(x)
        if len(corners) < 2:
            # Add a center point
            corners.append(lb + 0.5 * span)

        X = []
        for c in corners:
            X.append(project(np.array(c, dtype=float)))
        while len(X) < pop:
            X.append(project(lb + np.random.random(dim) * span))
        X = np.asarray(X, dtype=float)

        # Evaluate initial population (may not consume full budget)
        ys = np.full((len(X),), float("inf"), dtype=float)
        for i in range(len(X)):
            if evals >= budget:
                break
            ys[i] = evaluate(X[i])

        best_idx = int(np.argmin(ys))
        best_x = X[best_idx].copy()
        best_y = float(ys[best_idx])

        # ---- State variables ----
        # Sigma is a mutation scale; start from a fraction of domain size.
        sigma = 0.35
        sigma_low = 1e-6
        sigma_high = 1.0

        # Success counter to adapt sigma
        success_in_a_row = 0
        prev_best_y = best_y

        # Exploration probability decays over time
        # (but remains >0 to avoid premature convergence).
        # We'll compute based on remaining evaluations.
        iteration = 0

        # Helper: evaluate a batch with remaining budget
        def eval_batch(points):
            nonlocal evals
            m = len(points)
            out = np.full((m,), float("inf"), dtype=float)
            for i in range(m):
                if evals >= budget:
                    break
                out[i] = evaluate(points[i])
            return out

        # ---- Main loop ----
        # Each iteration produces up to `batch_size` new candidates.
        # Ensure not to exceed budget.
        # The number of iterations is bounded by budget / batch_size.
        # We select batch_size adaptively based on remaining evaluations.
        while evals < budget:
            iteration += 1
            remaining = budget - evals
            if remaining <= 0:
                break

            # Batch size: keep it bounded by pop and remaining
            batch_size = int(np.clip(pop, 2, remaining))
            # If remaining is small, we still make progress.
            batch_size = max(2, min(batch_size, remaining))

            # Decide exploitation vs exploration
            # Early more exploration; late mostly exploitation.
            frac_done = evals / max(1, budget)
            explore_prob = float(np.clip(0.6 * (1.0 - frac_done) + 0.15, 0.15, 0.7))

            # Candidate generation
            # Strategy: generate around best_x with gaussian noise,
            # occasionally from uniform random (exploration),
            # plus occasional coordinate probes to refine.
            # We'll build a batch array and then evaluate.
            candidates = np.empty((batch_size, dim), dtype=float)

            # Number of purely random exploration points
            n_rand = int(round(batch_size * explore_prob))
            n_rand = max(0, min(n_rand, batch_size))
            n_gauss = batch_size - n_rand

            # Random exploration points
            if n_rand > 0:
                r = lb + np.random.random((n_rand, dim)) * span
                candidates[:n_rand] = project(r)

            # Gaussian exploitation points around best
            if n_gauss > 0:
                # Scale mutation by sigma and domain span
                # Add small per-dimension scaling to avoid stagnation.
                # Note: use independent Gaussian per coordinate.
                if dim > 0:
                    jitter = np.random.normal(size=(n_gauss, dim))
                else:
                    jitter = np.empty((n_gauss, dim))
                step = (sigma * span)[None, :] * jitter
                g = best_x[None, :] + step
                candidates[n_rand:] = project(g)

            # Occasionally do coordinate probes around best
            # Replace a few candidates with 1D moves.
            # This helps refine along axes.
            n_probe = int(max(0, min(batch_size, dim // 2)))
            if n_probe > 0 and (np.random.random() < (0.35 if frac_done > 0.5 else 0.55)):
                # Choose distinct coordinates
                idxs = np.random.choice(dim, size=min(dim, n_probe), replace=False)
                # For each chosen coordinate, set a candidate with +/- move
                for k, j in enumerate(idxs[:n_probe]):
                    # Choose sign and magnitude
                    sign = -1.0 if np.random.random() < 0.5 else 1.0
                    # Probe step shrinks with sigma and iteration progress
                    mag = (0.25 + 0.75 * np.random.random()) * sigma
                    x = best_x.copy()
                    x[j] = x[j] + sign * mag * span[j]
                    candidates[(n_rand + k) % batch_size] = project(x)

            # Evaluate candidates
            ys_new = eval_batch(candidates)

            # Combine with current best only (selection-replacement implicit by choosing new best)
            idx_best_new = int(np.argmin(ys_new))
            y_best_new = float(ys_new[idx_best_new])
            x_best_new = candidates[idx_best_new].copy()

            # Determine improvement
            improved = y_best_new < best_y
            if improved:
                best_y = y_best_new
                best_x = x_best_new
                success_in_a_row += 1
            else:
                success_in_a_row = 0

            # Adapt sigma: decrease on improvement, increase slightly otherwise.
            # Use relative improvement magnitude to modulate adaptation.
            if improved:
                # If improvement is large, shrink more aggressively.
                denom = abs(prev_best_y) + 1e-12
                rel = (prev_best_y - best_y) / denom
                shrink = 0.85 - 0.25 * np.clip(rel, 0.0, 1.0)
                sigma *= shrink
            else:
                # If no improvement, grow a bit to re-explore.
                # Also grow mildly as budget runs out to avoid premature convergence.
                grow = 1.07 + 0.15 * frac_done
                sigma *= grow

            # Clamp sigma
            sigma = float(np.clip(sigma, sigma_low, sigma_high))

            # Update for next adaptation step
            prev_best_y = best_y

            # Optional: if sigma is extremely small, force some exploration by random reset
            # (only if we still have remaining budget).
            if sigma <= sigma_low * 5 and evals < budget:
                # Reset sigma to encourage escaping local minima
                if np.random.random() < 0.3:
                    sigma = max(0.25, sigma_high * 0.6)

        # Ensure best_x is within bounds (should already be)
        best_x = project(np.asarray(best_x, dtype=float))
        return best_x, float(best_y)
