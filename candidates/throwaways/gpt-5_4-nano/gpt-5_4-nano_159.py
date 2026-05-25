# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm using a
# coordinate-wise adaptive random search with periodic local refinements.
# Search state: Maintains the current best solution x_best and its value,
# along with per-dimension step sizes and an "age" counter.
# Candidate generation: Proposes new candidates by adding Gaussian noise
# scaled by per-dimension step sizes, plus occasional orthogonal
# direction sweeps (coordinate moves). Uses reflection at bounds.
# Selection and replacement: Evaluates candidates one-by-one, accepting the
# first strict improvement and otherwise keeping the best found so far; step
# sizes shrink when improvements are found, and expand when no improvement
# occurs over several trials.
# Adaptation: Per-dimension step sizes adapt using a success/failure rule:
# successes reduce step sizes, failures increase them within safe limits.
# Exploration mechanisms: Random Gaussian sampling and occasional coordinate
# probing help explore broadly.
# Exploitation mechanisms: Smaller step sizes plus coordinate refinements
# focus search near the best point.
# Boundary handling: Uses reflection for any out-of-bounds values to keep
# candidates within the feasible hyper-rectangle.
# Budget strategy: Uses an explicit evaluation counter and never calls the
# objective more than the provided budget.
# Closest known influences: Inspired by evolutionary strategies / coordinate
# search hybrids, but implemented in a minimal form without external deps.
# Novelty or unusual aspects: Uses per-dimension adaptive Gaussian scales and
# mixes two proposal types (global random + local coordinate sweep) to be robust
# across dimensions.
# Failure modes: If the objective is extremely noisy, strict improvements may
# be rare; the algorithm still tracks best-so-far and increases step sizes to
# maintain exploration. In very high-dimensional, flat landscapes it may
# require many evaluations to see progress.
# ALGORITHM_ANALYSIS_NOTE_END

from typing import Any, Callable, Tuple

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        # --- Read bounds ---
        lb = ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        if lb is None or ub is None:
            raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub.")

        if lb.shape == () and self.dim != 1:
            lb = np.full(self.dim, float(lb))
        if ub.shape == () and self.dim != 1:
            ub = np.full(self.dim, float(ub))

        lb = lb.reshape(-1).astype(float)
        ub = ub.reshape(-1).astype(float)

        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError(f"Bounds size mismatch: expected dim={self.dim}, got lb={lb.size}, ub={ub.size}.")

        # Ensure finite and consistent bounds
        # (If infinities exist, fall back to a generic scaling neighborhood.)
        finite_mask = np.isfinite(lb) & np.isfinite(ub) & (ub > lb)
        if not np.all(finite_mask):
            # Create a safe finite box by clamping inf/invalid with a default radius.
            # This avoids crashing; it may reduce effectiveness but keeps robustness.
            width = np.where(finite_mask, ub - lb, 1.0)
            # Choose lb/ub as lb if finite else -0.5*width; ub similarly.
            lb = np.where(np.isfinite(lb), lb, -0.5 * width)
            ub = np.where(np.isfinite(ub), ub, 0.5 * width)
            # If still not valid, enforce width
            ub = np.where(ub > lb, ub, lb + np.maximum(width, 1.0))

        width = ub - lb
        # --- Budget control ---
        max_evals = max(1, self.budget)
        evals = 0

        def eval_obj(x: np.ndarray) -> float:
            nonlocal evals
            if evals >= max_evals:
                # Should never happen if we guard correctly, but keep it safe.
                return float("inf")
            y = float(func(np.asarray(x, dtype=float)))
            evals += 1
            return y

        def reflect_bounds(x: np.ndarray) -> np.ndarray:
            # Reflect values into [lb, ub] using a modulus-like reflection.
            # Works for any real input.
            # For each dimension: map x to interval by repeated reflection.
            # If width is zero, just clamp.
            x = np.asarray(x, dtype=float)
            w = ub - lb
            out = x.copy()
            zero_w = w <= 0
            if np.any(zero_w):
                out[zero_w] = lb[zero_w]
                w = np.where(zero_w, 1.0, w)

            # Shift to [0, w]
            z = (out - lb) / w
            # Bring to [0, 1] with reflection:
            # For periodic extension with period 2: frac -> reflect if > 1
            frac = z - np.floor(z)
            refl = np.where(frac <= 0.5, 2.0 * frac, 2.0 * (1.0 - frac))
            out = lb + refl * w
            return out

        # --- Initialization ---
        # Start at a random point within the bounds.
        # (Harness controls randomness via numpy seed.)
        rng = np.random
        x_best = lb + rng.random(self.dim) * (ub - lb)
        y_best = eval_obj(x_best)

        # Per-dimension step sizes: start proportional to width.
        # Use a moderate fraction to balance exploration vs exploitation.
        step = 0.3 * width
        step = np.maximum(step, 1e-12)

        # Global parameters
        # Success/failure thresholds to adapt step sizes.
        success_streak = 0
        fail_streak = 0
        # Checkpoint for coordinate refinement frequency.
        coord_period = max(2, int(np.ceil(0.05 * max_evals)))  # about 5% of budget

        # Choose number of trials per loop; keep it small for readability.
        # We'll stop once we hit budget.
        trial_id = 0
        while evals < max_evals:
            trial_id += 1

            # Decide proposal type: mostly global random, sometimes coordinate sweep.
            do_coord = (coord_period > 0 and (trial_id % coord_period == 0))

            improved = False
            x_candidate = None
            y_candidate = None

            if not do_coord:
                # --- Candidate generation: global adaptive Gaussian ---
                # Draw noise; scale by step sizes.
                noise = rng.standard_normal(self.dim)
                # Add slight correlated tendency towards best (helps exploitation).
                # But since we propose around x_best, the correlation is implicit.
                x_candidate = reflect_bounds(x_best + noise * step)
                y_candidate = eval_obj(x_candidate)
            else:
                # --- Candidate generation: local coordinate sweep ---
                # Pick a dimension biased towards larger step sizes.
                idxs = np.argsort(-step)  # descending step
                # Try a few top coordinates within the remaining budget.
                # Each coordinate move counts as one evaluation (guarded).
                # We'll pick the best among those attempted.
                best_local_y = y_best
                best_local_x = x_best
                # How many coordinate attempts this round?
                k = min(3, self.dim)
                for j in range(k):
                    if evals >= max_evals:
                        break
                    d = int(idxs[j])
                    # Move +/- in the chosen coordinate with a step proportional move.
                    for sign in (1.0, -1.0):
                        if evals >= max_evals:
                            break
                        x_try = x_best.copy()
                        # Coordinate step: sometimes bigger to jump out.
                        scale = step[d] * (1.0 + 0.5 * rng.random())
                        x_try[d] = x_try[d] + sign * scale
                        x_try = reflect_bounds(x_try)
                        y_try = eval_obj(x_try)
                        if y_try < best_local_y:
                            best_local_y = y_try
                            best_local_x = x_try
                x_candidate = best_local_x
                y_candidate = best_local_y

            # --- Selection and replacement ---
            if y_candidate is not None and y_candidate < y_best:
                x_best = x_candidate
                y_best = y_candidate
                improved = True

            # --- Adaptation ---
            # Shrink step sizes on success; expand on failure.
            # Update per-dimension with mild randomness to avoid stagnation.
            if improved:
                success_streak += 1
                fail_streak = 0
                # Reduce steps (stronger reduction if consecutive successes).
                shrink = 0.85 ** (1.0 + 0.25 * min(3, success_streak))
                # Also reduce less on already-small dims.
                step = step * shrink
                # Keep a tiny minimum fraction of width
                step = np.maximum(step, 1e-6 * width)
            else:
                fail_streak += 1
                success_streak = 0
                # Expand steps after a few consecutive failures.
                expand = 1.07 ** (1.0 + 0.5 * min(5, fail_streak))
                step = step * expand
                # Cap by a fraction of the box to stay stable.
                step = np.minimum(step, 0.7 * width)

            # If width is extremely small, early exit to avoid numerical waste.
            if np.all(width <= 1e-12):
                break

        return x_best, y_best
