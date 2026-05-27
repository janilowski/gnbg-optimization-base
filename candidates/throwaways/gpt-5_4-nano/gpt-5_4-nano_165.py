# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, budget-aware black-box minimization algorithm for
# continuous box-constrained problems. It uses an evolution-strategy-like loop with
# directional sampling, rank-based selection, and a step-size adaptation, while
# ensuring the total number of objective evaluations never exceeds the given budget.
# Search state: Maintains the current best solution x_best, its value y_best, and a
# global step-size sigma. Also tracks a small set of recent best points for mild
# stabilization.
# Candidate generation: Each iteration samples a batch of candidate points around the
# current best. Directions are drawn from a standard normal distribution and normalized,
# then scaled by sigma and optionally combined with "momentum-like" shift vectors from
# recent bests. Candidates are clipped to the provided bounds.
# Selection and replacement: Evaluates candidates, picks the best among them, and updates
# x_best/y_best if improved. Uses a simple success rule to decide whether sigma should
# increase or decrease.
# Adaptation: Step-size sigma adapts using a multiplicative factor based on whether
# any candidate improved upon the incumbent. This keeps behavior robust across dimensions.
# Exploration mechanisms: Larger sigma at the start encourages exploration; sampling a batch
# of candidates per iteration explores multiple directions simultaneously.
# Exploitation mechanisms: After improvements, sigma shrinks to focus sampling near the
# incumbent best point.
# Boundary handling: Uses clipping to enforce feasibility within the provided lower/upper
# bounds (read from func.lower/upper or func.bounds.lb/ub). If bounds are degenerate (lb==ub),
# motion along those dimensions is naturally suppressed.
# Budget strategy: The algorithm allocates evaluation budget across iterations by picking a
# batch size that fits the remaining budget. It stops early if no budget remains.
# Closest known influences: Inspired by classic evolution strategies / CMA-ES-style step-size
# adaptation, but kept intentionally simple (single global sigma and rank-1 style updates).
# Novelty or unusual aspects: Adds a small "recent-best shift" term to bias sampling toward
# previously found improvement directions, without using covariance matrices.
# Failure modes: If the objective is extremely noisy or deceptive, simple success-based
# sigma adaptation may oscillate. Hard constraints with narrow feasible regions may
# cause frequent clipping, reducing effective search progress.
# ALGORITHM_ANALYSIS_NOTE_END

from typing import Callable, Tuple, Any
import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; still return a valid shaped x.
            # Use center of bounds if possible; y is NaN since objective was not evaluated.
            lb, ub = self._get_bounds(func, dim)
            x0 = (lb + ub) / 2.0
            return x0.astype(float, copy=False), float("nan")

        rng = np.random

        lb, ub = self._get_bounds(func, dim)
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        # Initial point: center of bounds (robust without assuming function structure).
        x_best = ((lb + ub) / 2.0).astype(float, copy=False)
        x_best = np.clip(x_best, lb, ub)

        # Evaluate incumbent.
        evals = 0
        y_best = self._safe_eval(func, x_best)
        evals += 1

        # Step-size: about 1/3 of the typical box scale, with safeguards.
        box_span = np.maximum(ub - lb, 0.0)
        # Use median span as scale to reduce sensitivity to outlier dimensions.
        med_span = float(np.median(box_span)) if box_span.size else 1.0
        sigma = max(1e-12, 0.33 * med_span)
        # Ensure sigma is not astronomically large.
        sigma = min(sigma, 1e6 if med_span > 0 else 1.0)

        # Recent bests (for a mild directional bias).
        recent = [x_best.copy()]
        max_recent = 5

        # Iterative sampling with adaptive batch sizes.
        # Use a modest batch size that scales with dim but respects the remaining budget.
        # Evaluate best-only each batch (selection is implicit by comparing to y_best).
        while evals < budget:
            remaining = budget - evals
            # Batch size: ~4 per dimension capped reasonably; must be >=1.
            # Keep small enough to adapt frequently.
            batch = int(min(remaining, max(1, min(12, 2 + dim // 2))))
            # If very low remaining budget, just sample one candidate.
            if batch <= 0:
                break

            # Build a momentum-like shift from recent improvements.
            shift = np.zeros(dim, dtype=float)
            if len(recent) >= 2:
                # Difference between last two best points, scaled down.
                delta = recent[-1] - recent[-2]
                shift = 0.25 * delta

            # Candidate generation:
            # Sample directions; normalize to keep step magnitude consistent across dimensions.
            Z = rng.standard_normal(size=(batch, dim))
            norms = np.linalg.norm(Z, axis=1)
            # Avoid division by zero for pathological samples.
            norms = np.where(norms > 0, norms, 1.0)
            D = Z / norms[:, None]

            # Random scaling per candidate to encourage diversity.
            # Use log-normal-ish spread via exp of normal noise.
            scale = np.exp(0.25 * rng.standard_normal(size=(batch, 1)))
            steps = sigma * scale * D

            # Candidates around incumbent with optional shift bias.
            X = x_best[None, :] + steps + shift[None, :]

            # Boundary handling by clipping.
            X = np.clip(X, lb, ub)

            # Evaluate batch; count exact evaluations and stop if we somehow exceed budget.
            Y = np.empty(batch, dtype=float)
            actual = 0
            for i in range(batch):
                if evals >= budget:
                    break
                y = self._safe_eval(func, X[i])
                Y[i] = y
                evals += 1
                actual += 1

            if actual == 0:
                break

            # Selection: pick best candidate of the evaluated batch.
            idx = int(np.argmin(Y[:actual]))
            x_cand = X[idx].copy()
            y_cand = float(Y[idx])

            improved = y_cand < y_best

            if improved:
                x_best = x_cand
                y_best = y_cand
                recent.append(x_best.copy())
                if len(recent) > max_recent:
                    recent.pop(0)

                # Exploitation: shrink sigma after success.
                sigma *= 0.85
            else:
                # Exploration: slightly expand sigma on failure to escape stagnation.
                sigma *= 1.08

            # Keep sigma within reasonable limits based on box size.
            # If span is tiny, sigma should be tiny too.
            tiny_span = float(np.min(box_span)) if box_span.size else 0.0
            # Lower bound based on feasible scale.
            min_sigma = 1e-12 if tiny_span == 0.0 else 1e-6 * tiny_span
            # Upper bound based on feasible scale.
            max_sigma = 1.0 if box_span.size == 0 else 0.8 * float(np.max(box_span))
            sigma = float(np.clip(sigma, min_sigma, max_sigma if max_sigma > 0 else min_sigma))

        return x_best, float(y_best)

    @staticmethod
    def _safe_eval(func: Callable[[np.ndarray], float], x: np.ndarray) -> float:
        y = func(x)
        # Ensure scalar float.
        try:
            y = float(y)
        except Exception:
            y = float(np.asarray(y).item())
        return y

    @staticmethod
    def _get_bounds(func: Any, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        # Priority:
        # 1) func.lower/func.upper
        # 2) func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(getattr(func, "lower"), dtype=float).reshape(-1)
            ub = np.asarray(getattr(func, "upper"), dtype=float).reshape(-1)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(getattr(func.bounds, "lb"), dtype=float).reshape(-1)
            ub = np.asarray(getattr(func.bounds, "ub"), dtype=float).reshape(-1)
        else:
            # Fallback: assume a standard box [-5, 5] if bounds aren't provided.
            lb = np.full(dim, -5.0, dtype=float)
            ub = np.full(dim, 5.0, dtype=float)

        # If bounds are scalar, expand.
        if lb.size == 1:
            lb = np.full(dim, float(lb[0]), dtype=float)
        if ub.size == 1:
            ub = np.full(dim, float(ub[0]), dtype=float)

        if lb.size != dim or ub.size != dim:
            # Try to broadcast if possible; otherwise truncate/pad.
            try:
                lb = np.broadcast_to(lb, (dim,)).astype(float, copy=False)
                ub = np.broadcast_to(ub, (dim,)).astype(float, copy=False)
            except Exception:
                lb2 = np.zeros(dim, dtype=float)
                ub2 = np.zeros(dim, dtype=float)
                n = min(dim, lb.size)
                lb2[:n] = lb[:n]
                ub2[:n] = ub[:n]
                if lb.size < dim:
                    lb2[n:] = lb[-1] if lb.size else 0.0
                if ub.size < dim:
                    ub2[n:] = ub[-1] if ub.size else 0.0
                lb, ub = lb2, ub2

        # Ensure lb <= ub elementwise.
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        return lb2, ub2
