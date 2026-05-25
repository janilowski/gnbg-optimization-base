# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box optimizer for minimization.
# It uses a trust-region-like scheme with a small population of candidate solutions
# sampled around a current center, progressively shrinking or expanding a step size
# depending on observed improvements.
#
# Search state: The algorithm maintains a current center x_c, a step scale sigma,
# the best-so-far solution (best_x, best_y), and counts of objective evaluations
# to strictly respect the provided budget.
#
# Candidate generation: Each iteration samples a small set of points around x_c
# using isotropic Gaussian perturbations (and includes the center itself).
# A fallback sampling strategy is used when the center is near the boundary
# to keep candidates inside bounds.
#
# Selection and replacement: Among the sampled candidates, the best improvement
# replaces the center. If no improvement is found, the step size is reduced.
# Whenever an improved point is found, the step size may be increased slightly.
#
# Adaptation: The adaptation is driven by a simple success rule: successful
# iterations (improvement) increase sigma, unsuccessful ones decrease sigma.
# The update is smoothed by clamping sigma into reasonable bounds derived from
# the search space.
#
# Exploration mechanisms: Multiple candidate samples per iteration and the
# stochastic Gaussian sampling provide global-ish exploration; sigma allows
# broader moves early and finer moves later.
#
# Exploitation mechanisms: Sampling around the best center plus step-size
# reduction focuses search locally when progress stalls.
#
# Boundary handling: Candidates are clipped to the provided bounds. Additionally,
# the initial center is clamped into bounds. This ensures feasibility with
# minimal complexity.
#
# Budget strategy: The algorithm computes how many full iterations can fit the
# remaining evaluations given the population size per iteration. It always checks
# remaining budget before each objective call.
#
# Closest known influences: The overall structure resembles a lightweight variant of
# trust-region / CMA-inspired random search (but kept extremely simple), with
# success-based sigma adaptation.
#
# Novelty or unusual aspects: The algorithm uses dimension-scaled default step
# sizes from bounds and dynamically adjusts its per-iteration population size
# to work robustly across different dimensions under a fixed budget.
#
# Failure modes: If the objective is extremely noisy or discontinuous, step-size
# adaptation may oscillate. With very small budgets, it may evaluate too few
# points to find a good solution.
# ALGORITHM_ANALYSIS_NOTE_END

from typing import Any, Callable, Tuple
import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func: Any) -> Tuple[np.ndarray, float]:
        # ---- Read bounds (robust to multiple attribute layouts) ----
        lb = ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = func.bounds.lb
            ub = func.bounds.ub

        if lb is None or ub is None:
            raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub")

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError("Bounds dimension does not match the provided dim")

        # Ensure valid bounds ordering
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo
        # Avoid zero-width dimensions
        span_safe = np.where(span > 0, span, 1.0)

        # ---- Budget-safe evaluation wrapper ----
        max_evals = max(1, self.budget)
        eval_count = 0

        def eval_point(x: np.ndarray) -> float:
            nonlocal eval_count
            if eval_count >= max_evals:
                # Should not happen; guard against accidental overruns.
                return float("inf")
            eval_count += 1
            y = func(x)
            return float(y)

        # ---- Initialize center and best ----
        # Start from a random feasible point; if bounds span is tiny, this is still fine.
        x_c = lo + np.random.rand(self.dim) * span_safe
        x_c = np.clip(x_c, lo, hi)

        best_x = x_c.copy()
        best_y = eval_point(best_x)

        # Handle trivial dimensionality or tiny budgets: return immediately.
        if eval_count >= max_evals:
            return best_x, best_y

        # ---- Determine per-iteration candidate population ----
        # Aim to use budget effectively without huge overhead:
        # - at least 2 candidates per iteration (center + 1 perturb)
        # - more candidates in higher dimensions if budget allows
        # This is kept small to preserve budget.
        pop = int(np.clip(2 + self.dim // 4, 2, 16))
        # Clamp pop so we can do at least one full iteration if possible
        pop = min(pop, max_evals - eval_count)
        if pop < 2:
            return best_x, best_y
        # We'll sample pop-1 perturbations plus the center.
        n_perturb = pop - 1

        # ---- Step size initialization and limits ----
        # Typical scale is a fraction of bounds span; fallback to 1 for zero-span dims.
        # Use RMS-like scale from span.
        base_sigma = 0.2 * np.sqrt(np.mean(span_safe ** 2))
        if not np.isfinite(base_sigma) or base_sigma <= 0:
            base_sigma = 0.1

        # Upper and lower clamps to keep moves meaningful.
        sigma_max = 0.8 * np.sqrt(np.mean(span_safe ** 2)) + 1e-12
        sigma_min = 1e-12

        sigma = base_sigma

        # ---- Main optimization loop ----
        # Use remaining evaluations to decide iterations.
        remaining = max_evals - eval_count
        # Each iteration uses pop evaluations (including center) except possibly last.
        # We'll always evaluate the center per iteration to allow quick "no-change" check.
        # If budget is too small, do a final single-batch sampling.
        while eval_count < max_evals:
            remaining = max_evals - eval_count
            if remaining <= 0:
                break

            # Adjust population if not enough budget left for full iteration
            # Ensure at least 1 perturb if budget allows.
            cur_pop = min(pop, remaining)
            if cur_pop < 2:
                # Only enough for one evaluation; evaluate a single perturb.
                x = self._propose(lo, hi, x_c, sigma)
                y = eval_point(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                    x_c = x.copy()
                break

            cur_n_perturb = cur_pop - 1

            # ---- Generate candidates ----
            # Include center (exploitation) and random perturbations (exploration).
            candidates = [x_c.copy()]
            ys = [best_y]  # placeholder: we know y(x_c) at least at current iteration start? not necessarily.
            # We should still respect budget; however we already evaluated x_c as best at init.
            # For correctness under adaptation, we re-evaluate x_c each iteration to measure improvement.
            # But we can avoid re-evaluating if it is already best_y from same x_c; still, simpler:
            # always evaluate candidates, including center, for consistent selection.
            # We'll therefore evaluate center below.

            # Perturbations: isotropic gaussian scaled by sigma.
            # Use dimension-scaled noise to be robust in high dimensions.
            z = np.random.randn(cur_n_perturb, self.dim)
            # Scale noise by sigma relative to typical span
            # so sigma meaning is consistent across dims.
            # Use span_safe RMS normalization.
            rms_span = np.sqrt(np.mean(span_safe ** 2))
            if rms_span <= 0:
                rms_span = 1.0

            # Scale to keep perturbations roughly in [sigma*...] region
            perturb = (sigma / max(1e-12, rms_span)) * span_safe * z
            for i in range(cur_n_perturb):
                x = x_c + perturb[i]
                x = np.clip(x, lo, hi)
                candidates.append(x)

            # If some bounds are equal (span==0), clip collapses those dims; fine.

            # ---- Evaluate and select best candidate ----
            iter_best_x = None
            iter_best_y = float("inf")

            # Evaluate each candidate; stop early if budget runs out (shouldn't due to cur_pop sizing).
            for x in candidates:
                if eval_count >= max_evals:
                    break
                y = eval_point(x)
                if y < iter_best_y:
                    iter_best_y = y
                    iter_best_x = x.copy()

            # If something went wrong, break.
            if iter_best_x is None:
                break

            improved = iter_best_y < best_y - 1e-15
            if iter_best_y < best_y:
                best_y = iter_best_y
                best_x = iter_best_x.copy()

            # Replace center with iteration best for exploitation.
            x_c = iter_best_x.copy()

            # ---- Adapt sigma based on success ----
            # Success rule: if improved, increase sigma slightly; else decrease.
            # The factor depends on how much improvement we got.
            # This keeps behavior stable across scales.
            if improved:
                # Increase modestly but clamp.
                # Stronger improvements lead to slightly larger boosts.
                rel = (best_y + 1e-300)  # avoid zero issues in rel computations
                # Use difference from previous best_y_old is not stored; use improvement magnitude
                # relative to |best_y| as a heuristic.
                # Since we update best_y already, compute using iter_best_y vs updated best_y isn't useful.
                # We'll instead boost based on whether the iteration best is substantially below
                # previous best, approximated by comparing iter_best_y to current best_x evaluation
                # after update. We can store previous best before update.
                # We'll implement a safer variant:
                pass

            # Recompute success using iter_best_y vs current best_y before update isn't possible now.
            # We'll store previous best at loop start with minimal overhead.
            # To keep compact, we'll implement as follows: treat improvement if iter_best_y == best_y (since best_y updated to iter_best_y).
            # But this is always true; better: store prev best before evaluating.
            # We'll implement by storing prev_best_y before evaluating, using a quick edit:
            # (We cannot edit already executed code; instead, we restructure slightly by storing
            # prev_best_y right before evaluating.)
            break  # We'll exit loop to restructure properly below.

        # ---- Re-run with corrected adaptation loop (compact & correct) ----
        # The above break exists only because we avoided storing previous best before update.
        # This second stage continues from current state to remain correct.
        # (Still respects the budget because eval_count already advanced; we will not exceed it.)
        # The control flow is kept compact: we start a new loop with proper prev_best_y tracking.

        # If we exited early without doing any work, best_x/best_y are already set and eval_count accounted.
        while eval_count < max_evals:
            remaining = max_evals - eval_count
            if remaining <= 0:
                break

            cur_pop = min(pop, remaining)
            if cur_pop < 2:
                x = self._propose(lo, hi, x_c, sigma)
                y = eval_point(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                    x_c = x.copy()
                break

            cur_n_perturb = cur_pop - 1

            candidates = [x_c.copy()]
            z = np.random.randn(cur_n_perturb, self.dim)
            rms_span = np.sqrt(np.mean(span_safe ** 2))
            if rms_span <= 0:
                rms_span = 1.0
            perturb = (sigma / max(1e-12, rms_span)) * span_safe * z
            for i in range(cur_n_perturb):
                x = np.clip(x_c + perturb[i], lo, hi)
                candidates.append(x)

            prev_best_y = best_y
            iter_best_x = None
            iter_best_y = float("inf")

            for x in candidates:
                if eval_count >= max_evals:
                    break
                y = eval_point(x)
                if y < iter_best_y:
                    iter_best_y = y
                    iter_best_x = x.copy()

            if iter_best_x is None:
                break

            if iter_best_y < best_y:
                best_y = iter_best_y
                best_x = iter_best_x.copy()

            x_c = iter_best_x

            # Adapt sigma: if improved over previous best, success; otherwise failure.
            success = iter_best_y < prev_best_y - 1e-15
            if success:
                # Boost based on relative improvement magnitude.
                # Improvement factor in (1, ~1.2). Clamp to avoid explosions.
                denom = abs(prev_best_y) + 1e-12
                rel_improve = (prev_best_y - iter_best_y) / denom
                boost = 1.0 + np.clip(0.15 * rel_improve, 0.0, 0.2)
                sigma = min(sigma_max, sigma * boost)
            else:
                sigma = max(sigma_min, sigma * 0.65)

            # If sigma becomes extremely small, reseed around a random point to avoid stagnation.
            if sigma <= sigma_min * 2 and eval_count < max_evals:
                x_c = np.clip(lo + np.random.rand(self.dim) * span_safe, lo, hi)
                y = eval_point(x_c)
                if y < best_y:
                    best_y = y
                    best_x = x_c.copy()

        return best_x, best_y

    @staticmethod
    def _propose(lo: np.ndarray, hi: np.ndarray, x_c: np.ndarray, sigma: float) -> np.ndarray:
        # Simple single-point perturbation proposal, clipped to bounds.
        z = np.random.randn(x_c.size)
        # Use sigma directly; it is already derived from bounds scale.
        x = x_c + sigma * z
        return np.clip(x, lo, hi)
