# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# using an evolution strategy (ES) with Gaussian sampling, elitist selection,
# and an archive of the best points. It maintains a search distribution
# (mean + isotropic step size) and iteratively refines it to minimize a
# provided objective function.
# Search state: The algorithm keeps a current mean vector (x_mean), a
# scalar step size (sigma), an evaluation counter (evals_used), and the
# best-so-far solution (best_x, best_y). It also maintains an archive of a
# few best points found in previous iterations for directional exploitation.
# Candidate generation: Each iteration draws a small batch of candidates:
# x_i = x_mean + sigma * z_i, where z_i are standard normal vectors.
# Additionally, it injects a couple of candidates guided by the best archived
# points to improve robustness on simple landscapes.
# Selection and replacement: After evaluating all candidates, the algorithm
# identifies the best candidate in the batch and replaces x_mean with a
# weighted recombination that biases toward the better candidates. It also
# updates best_x/best_y using the global minimum seen so far.
# Adaptation: sigma is adapted using a simple success rule: if the batch
# produces an improvement over the previous best, sigma is decreased
# moderately; otherwise, sigma is increased to encourage exploration.
# Exploration mechanisms: Gaussian sampling from the current distribution
# provides exploration; sigma increases after stagnation.
# Exploitation mechanisms: Weighted recombination toward the best candidates
# and a small “directional” sampling component around archived best points
# provide exploitation.
# Boundary handling: Candidates are clipped to the provided box bounds.
# Step size sigma is kept within reasonable numeric limits.
# Budget strategy: The algorithm computes the maximum number of iterations
# based on the evaluation budget and uses fixed batch sizes, ensuring the
# total number of objective evaluations never exceeds the budget.
# Closest known influences: The design loosely follows strategies from classic
# evolution strategies (e.g., CMA-ES-like principles but simplified to isotropic
# covariance) with elitist selection and success-based step-size adaptation.
# Novelty or unusual aspects: Uses a tiny archive and mixes isotropic Gaussian
# search with a lightweight directional proposal derived from stored elites.
# Failure modes: If the objective is extremely noisy or highly constrained
# such that clipping dominates, the algorithm may stagnate; isotropic sigma
# adaptation may be insufficient for strongly anisotropic problems.
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

        # --- Read bounds from func ---
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        if lb is None or ub is None:
            # Fall back to a generic box if bounds are missing.
            # Note: The interface request says to read bounds from func.
            # If not provided, we still keep behavior safe.
            lb = -5.0 * np.ones(dim, dtype=float)
            ub = 5.0 * np.ones(dim, dtype=float)

        lb = np.broadcast_to(lb, (dim,)).copy()
        ub = np.broadcast_to(ub, (dim,)).copy()

        # Prevent degenerate bounds
        span = ub - lb
        span = np.where(span > 0, span, 1.0)

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        # --- Evaluation wrapper to never exceed budget ---
        evals_used = 0
        best_x = None
        best_y = None

        def eval_at(x):
            nonlocal evals_used, best_x, best_y
            if evals_used >= budget:
                # Should not happen due to careful planning; return best known.
                return best_y
            y = float(func(np.asarray(x, dtype=float)))
            evals_used += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = np.asarray(x, dtype=float).copy()
            return y

        # --- Initialization ---
        rng = np.random  # harness seeds numpy globally

        # Start mean at a random point within bounds
        x_mean = lb + rng.random(dim) * (ub - lb)

        # Estimate a reasonable initial sigma from bounds
        sigma = 0.3 * float(np.max(span)) / max(1.0, (dim ** 0.5))
        sigma = max(sigma, 1e-12)

        # Archive a small set of elite points for directional proposals
        elite_x = []
        elite_y = []

        # Evaluate initial point
        eval_at(x_mean)

        # --- Iteration planning ---
        # Use a small batch size to keep overhead low and adaptation stable.
        # Ensure at least one evaluation remains beyond initialization if budget allows.
        batch = 6 if dim >= 5 else 4
        batch = max(2, min(batch, max(2, budget // 5 if budget >= 10 else batch)))

        # Remaining evaluations after initial evaluation
        remaining = budget - evals_used
        if remaining <= 0:
            return best_x, best_y

        # Number of full batches we can run (may have a partial last batch)
        iters = remaining // batch
        if iters <= 0:
            # Do a single candidate batch partially
            iters = 1
            batch = min(batch, remaining)

        # Store previous best for success rule
        prev_best = best_y

        # Directional proposal strength
        dir_scale = 0.15

        for _ in range(iters):
            if evals_used >= budget:
                break

            # Adjust last batch to respect budget
            remaining = budget - evals_used
            k = min(batch, remaining)
            if k <= 0:
                break

            candidates = np.empty((k, dim), dtype=float)

            # Primary isotropic Gaussian sampling around current mean
            # (k - nb_dir for direction-guided candidates)
            nb_dir = 0
            if elite_x and k >= 3:
                nb_dir = 2
            nb_dir = min(nb_dir, k)

            nb_gauss = k - nb_dir

            # Gaussian candidates
            if nb_gauss > 0:
                Z = rng.standard_normal((nb_gauss, dim))
                candidates[:nb_gauss] = x_mean + sigma * Z

            # Directional candidates from elites
            if nb_dir > 0:
                # Pick an elite (best) and move slightly toward it from current mean.
                # Also include a small exploration orthogonal-ish component by adding noise.
                # We keep it simple: best elite + noise.
                e_idx = int(np.argmin(elite_y)) if elite_y else 0
                e = np.asarray(elite_x[e_idx], dtype=float)
                D = (e - x_mean)
                # Normalize direction to avoid huge steps near convergence
                dnorm = float(np.linalg.norm(D))
                if dnorm > 0:
                    D = D / dnorm
                else:
                    D = rng.standard_normal(dim)
                    D /= max(1e-12, float(np.linalg.norm(D)))
                for j in range(nb_dir):
                    noise = rng.standard_normal(dim)
                    # Blend direction with noise; ensure it doesn't vanish
                    proposal = x_mean + (dir_scale * sigma) * D + (0.25 * sigma) * noise
                    candidates[nb_gauss + j] = proposal

            # Clip candidates to bounds
            for i in range(k):
                candidates[i] = clip(candidates[i])

            # Evaluate batch
            ys = np.empty(k, dtype=float)
            for i in range(k):
                ys[i] = eval_at(candidates[i])

            # Update elite archive with top points from this batch
            # Keep up to 6 elites (small memory, stable).
            # We add points even if best is already known to help directional proposals.
            order = np.argsort(ys)
            for i in order[: min(3, k)]:
                elite_x.append(np.asarray(candidates[i], dtype=float).copy())
                elite_y.append(float(ys[i]))

            # Prune elites to fixed size
            if len(elite_x) > 6:
                o = np.argsort(elite_y)
                elite_x = [elite_x[i] for i in o[:6]]
                elite_y = [elite_y[i] for i in o[:6]]

            # Selection and recombination:
            # Choose top m and compute weighted mean. This biases exploitation
            # but remains stable with an isotropic distribution.
            m = min(3, k)
            top_idx = np.argsort(ys)[:m]
            top_y = ys[top_idx]

            # Convert to positive weights with rank-based scaling
            # (avoid instability from huge objective ranges).
            # Higher rank => better.
            ranks = np.arange(m, 0, -1, dtype=float)  # m..1
            # If there is exact improvement, slightly sharpen weights
            if best_y is not None and best_y < prev_best:
                ranks *= 1.15
            w = ranks / np.sum(ranks)

            x_new = np.zeros(dim, dtype=float)
            for wi, idx in zip(w, top_idx):
                x_new += wi * candidates[idx]

            x_mean = clip(x_new)

            # Success-based step size adaptation (simple and robust):
            # If we improved the global best, reduce sigma; otherwise increase.
            if best_y < prev_best - 1e-15:
                sigma *= 0.82
            else:
                sigma *= 1.08

            # Keep sigma within reasonable bounds based on span
            max_sigma = 0.8 * float(np.max(span))
            min_sigma = 1e-12
            sigma = float(np.clip(sigma, min_sigma, max_sigma))

            prev_best = best_y

        return best_x, best_y
