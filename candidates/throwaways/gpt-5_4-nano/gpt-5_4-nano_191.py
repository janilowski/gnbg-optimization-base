# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer using a
# combination of adaptive coordinate directions and local random search,
# inspired by coordinate descent with restart-like behavior.
# Search state: Maintains a current best point x_best, its value y_best, and
# a step-size (sigma) controlling how far candidate points are sampled. Also
# tracks improvement count and a direction-sign memory to slightly bias future
# searches.
# Candidate generation: At each iteration, generates a small set of candidates:
# (1) one-point mutations around the current best using isotropic Gaussian noise;
# (2) coordinate-wise perturbations (one dimension at a time) using a learned sign
# bias; and (3) optional “reflection” candidates when progress stalls.
# Selection and replacement: Evaluates all candidates, selects the best (minimum) and
# replaces the current best if improved; otherwise it may reduce sigma and update
# direction memory based on unsuccessful attempts.
# Adaptation: Step-size sigma adapts multiplicatively: it increases slightly on
# successful improvement and decreases on stagnation. Direction sign memory is
# updated using the relative performance of positive vs negative perturbations.
# Exploration mechanisms: Random Gaussian mutations and periodic coordinate sampling
# provide global-ish exploration. Reflection candidates add diversity when stuck.
# Exploitation mechanisms: Coordinate-wise moves and re-sampling around the current
# best with smaller sigma focus local improvement.
# Boundary handling: Samples are clipped to the provided box bounds; if a candidate
# hits a bound, it is still evaluated and may influence step-size adaptation.
# Budget strategy: Uses an explicit evaluation counter and never exceeds the provided
# evaluation budget. The number of iterations/candidates per iteration is chosen to
# respect the budget exactly.
# Closest known influences: Coordinate descent / evolution strategies (ES) style
# mutation-selection, blended with adaptive step-size control.
# Novelty or unusual aspects: Combines coordinate-wise sign bias with isotropic ES-like
# steps in a single compact loop, without requiring gradients or extra function calls.
# Failure modes: In very high dimensions with complex non-separable landscapes, the
# small candidate set per iteration may miss better regions; stagnation can lead to
# premature sigma shrinking if bounds are tight or the objective is noisy.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return something deterministic.
            x0 = np.zeros(d, dtype=float)
            return x0, float("inf")

        # --- Bounds extraction (robust to different func APIs) ---
        lb = ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lower") and hasattr(func.bounds, "upper"):
            lb = np.asarray(func.bounds.lower, dtype=float)
            ub = np.asarray(func.bounds.upper, dtype=float)

        if lb is None or ub is None:
            raise AttributeError(
                "func must expose bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub"
            )
        if lb.shape != (d,) or ub.shape != (d,):
            lb = np.broadcast_to(lb, (d,)).astype(float, copy=False)
            ub = np.broadcast_to(ub, (d,)).astype(float, copy=False)

        # Ensure valid bounds ordering.
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo
        # If some spans are 0, those dimensions are fixed.
        span_safe = np.where(span > 0, span, 1.0)

        # --- Evaluation budget bookkeeping ---
        n_eval = 0

        def eval_at(x):
            nonlocal n_eval
            if n_eval >= budget:
                # Do not exceed budget; return a very bad value.
                return float("inf")
            x = np.asarray(x, dtype=float)
            x = np.clip(x, lo, hi)
            y = float(func(x))
            n_eval += 1
            return y

        # --- Initial point ---
        # Use the center with a small random jitter to avoid flat symmetry.
        rng = np.random
        x_best = lo + 0.5 * span_safe * 0.0  # center by construction for fixed dims
        # More robust center:
        x_best = (lo + hi) * 0.5
        # Add jitter only where span > 0
        jitter = 0.05 * span_safe * rng.uniform(-1.0, 1.0, size=d)
        x_best = np.clip(x_best + jitter * (span > 0), lo, hi)
        y_best = eval_at(x_best)

        # If budget allows, also sample one random point.
        if n_eval < budget:
            x_rand = lo + rng.uniform(0.0, 1.0, size=d) * span_safe
            x_rand = np.clip(x_rand, lo, hi)
            y_rand = eval_at(x_rand)
            if y_rand < y_best:
                x_best, y_best = x_rand, y_rand

        # Adaptive step size based on span.
        # Use a fraction of average non-zero span as initial sigma.
        nonzero = span > 0
        if np.any(nonzero):
            base = float(np.median(span[nonzero]))
        else:
            base = 1.0
        sigma = 0.25 * base
        sigma_min = 1e-12 * base
        sigma_max = 2.0 * base

        # Direction sign memory for coordinate perturbations:
        # +1 means favor positive moves, -1 means favor negative moves, 0 unknown.
        dir_sign = rng.choice([-1.0, 1.0], size=d)
        # How many consecutive steps without improvement.
        no_improve = 0

        # Candidate generation strategy:
        # We choose a small set per iteration so we can respect budget.
        # Each loop can evaluate up to:
        # - k_mut isotropic Gaussian candidates
        # - k_coord coordinate candidates (paired +/- around sign memory sometimes)
        # - plus optional reflection(s)
        # We'll compute remaining budget and adjust.
        # For compactness, define default sizes and scale with dimension.
        k_mut = 2 if d <= 10 else 3
        k_coord = 2 if d <= 10 else 3
        k_reflect = 1

        # Total iterations target (bounded by budget).
        # Each iteration can use about k_mut + k_coord (and maybe k_reflect) evals.
        # We'll run until budget is consumed.
        # Ensure at least one iteration.
        avg_per_iter = max(1, k_mut + k_coord + k_reflect)
        max_iters = max(1, (budget - n_eval) // avg_per_iter + 1)

        def propose_gaussian(center, sigma_local, n):
            # Isotropic Gaussian around center.
            return center + rng.normal(0.0, sigma_local, size=(n, d))

        for _ in range(max_iters):
            if n_eval >= budget:
                break

            remaining = budget - n_eval
            if remaining <= 0:
                break

            # Adjust candidates to not exceed budget.
            # Evaluate cap per loop.
            cap = remaining

            candidates = []

            # 1) Isotropic ES-like mutations
            n_mut = min(k_mut, cap)
            if n_mut > 0:
                C = propose_gaussian(x_best, sigma, n_mut)
                candidates.extend(C)

            # 2) Coordinate perturbations with sign bias
            # Select a small subset of coordinates, prefer those with nonzero span.
            if cap - len(candidates) > 0:
                n_coord = min(k_coord, cap - len(candidates))
                # Weighted coordinate selection by span to avoid fixed dims.
                weights = np.where(span > 0, span / (np.sum(span) + 1e-30), 0.0)
                idx = np.arange(d)
                if np.sum(weights) > 0:
                    chosen = rng.choice(idx, size=n_coord, replace=False if n_coord < d else True, p=weights)
                else:
                    chosen = rng.choice(idx, size=n_coord, replace=False if n_coord < d else True)

                for j in chosen:
                    if span[j] <= 0:
                        # Fixed dimension; only return center.
                        candidates.append(x_best.copy())
                        continue
                    # Use biased step along coordinate; scale with coordinate span.
                    step = sigma * (span[j] / (base + 1e-30)) ** 0.5
                    step = np.clip(step, -span_safe[j] * 0.5, span_safe[j] * 0.5)
                    x1 = x_best.copy()
                    s = dir_sign[j]
                    x1[j] = x1[j] + s * step
                    candidates.append(x1)

            # 3) Reflection candidate for diversity when stagnating
            # Reflection: x_ref = x_best + (x_best - x_last) when available.
            # Since we don't store x_last robustly, we do a simpler symmetric probe:
            # reflect around center by negating a random direction.
            if cap - len(candidates) > 0 and no_improve >= 2:
                n_ref = min(k_reflect, cap - len(candidates))
                for _r in range(n_ref):
                    # Random direction with some sparsity.
                    mask = rng.rand(d) < (0.2 if d > 5 else 0.5)
                    if not np.any(mask):
                        mask[rng.randint(0, d)] = True
                    direction = np.zeros(d, dtype=float)
                    direction[mask] = rng.choice([-1.0, 1.0], size=int(np.sum(mask)))
                    # Reflection-like move: symmetric about current best
                    x_ref = x_best - direction * sigma * 0.5
                    candidates.append(x_ref)

            # If too many, trim.
            if len(candidates) > cap:
                candidates = candidates[:cap]

            # Evaluate candidates and pick best.
            y_candidates = []
            best_local_x = x_best
            best_local_y = y_best

            for x_c in candidates:
                y_c = eval_at(x_c)
                y_candidates.append(y_c)
                if y_c < best_local_y:
                    best_local_y = y_c
                    best_local_x = np.asarray(x_c, dtype=float)

            # Adaptation & direction memory updates.
            if best_local_y < y_best:
                # Successful improvement
                x_best, y_best = best_local_x, best_local_y
                no_improve = 0
                sigma = min(sigma_max, sigma * 1.15)

                # Update dir_sign based on which side improved for a few coordinates.
                # We do a small heuristic: compare performance of +/- for up to 3 coords.
                # This does consume extra evaluations; respect remaining budget by using 0..cap.
                # Use only if budget remains and we didn't use full cap.
                remaining_after = budget - n_eval
                if remaining_after > 0 and d > 0:
                    # Choose coords where span is nonzero, if possible.
                    coord_candidates = np.where(span > 0)[0]
                    if coord_candidates.size > 0:
                        n_test = min(3, coord_candidates.size, remaining_after)
                        chosen = rng.choice(coord_candidates, size=n_test, replace=False)
                        # Evaluate +/- around current best along chosen coords.
                        for j in chosen:
                            # Skip if extremely tiny span.
                            if span[j] <= 0:
                                continue
                            step = sigma * (span[j] / (base + 1e-30)) ** 0.5
                            x_pos = x_best.copy()
                            x_neg = x_best.copy()
                            sgn_pos = 1.0
                            sgn_neg = -1.0
                            x_pos[j] = x_pos[j] + sgn_pos * step
                            x_neg[j] = x_neg[j] + sgn_neg * step
                            # Evaluate, but each costs budget.
                            y_pos = eval_at(x_pos)
                            if n_eval >= budget:
                                break
                            y_neg = eval_at(x_neg)
                            if n_eval >= budget:
                                break
                            dir_sign[j] = 1.0 if y_pos <= y_neg else -1.0
            else:
                # No improvement
                no_improve += 1
                # Decrease sigma to refine local region.
                sigma = max(sigma_min, sigma * (0.7 if no_improve >= 2 else 0.85))

                # Mild random exploration reset if sigma gets too small.
                if sigma <= sigma_min * 2 and no_improve >= 3:
                    sigma = min(sigma_max, 0.5 * base)
                    no_improve = 0
                    # Randomize direction signs to avoid bias loops.
                    dir_sign = rng.choice([-1.0, 1.0], size=d)

        # Final safety clip.
        x_best = np.clip(np.asarray(x_best, dtype=float), lo, hi)
        return x_best, float(y_best)
