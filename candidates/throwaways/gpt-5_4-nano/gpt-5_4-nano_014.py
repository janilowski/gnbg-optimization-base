# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm inspired by
# population-based evolutionary search with coordinate-wise Gaussian mutation,
# elitist selection, and intermittent exploration. It works for any dimension,
# uses only objective evaluations (no gradients), and strictly respects the
# provided evaluation budget.
# Search state: Maintains a small population of candidate vectors along with
# their objective values. Keeps track of the incumbent best solution found so far.
# Candidate generation: Samples offspring by adding Gaussian noise whose scale
# adapts based on progress. Also uses occasional uniform samples for exploration
# to prevent stagnation. All offspring are clipped to the provided bounds.
# Selection and replacement: Uses elitist replacement by selecting the best
# solutions among parents and offspring. The best individual becomes the next
# incumbent and influences mutation step-size adaptation.
# Adaptation: Tracks improvement in the incumbent; if improvement stalls, it
# increases exploration (larger mutation scales and more uniform candidates).
# Exploitation mechanisms: Focuses sampling around the current best individual
# and uses shrinking mutation scales after successful improvements.
# Exploration mechanisms: Injects a few uniformly random candidates and applies
# a higher-variance mutation when progress stalls.
# Boundary handling: Candidate vectors are always clipped to the feasible
# box bounds derived from func.lower/upper or func.bounds.lb/ub.
# Budget strategy: Converts the overall evaluation budget into a fixed number
# of generations, evaluating exactly the remaining number of candidates each
# generation. Uses a conservative scheme to never exceed the budget.
# Closest known influences: Closely resembles simple evolutionary strategies
# (ES/CMA-lite style) with step-size adaptation and elitist selection, but
# implemented in a dimension-agnostic, lightweight way.
# Novelty or unusual aspects: Uses a dimension-scaled coordinate-wise mutation
# (per-dimension step sizes) with a stall-triggered “uniform injection” to
# enhance robustness on diverse black-box functions.
# Failure modes: If bounds are extremely tight, clipping can reduce diversity
# and lead to premature convergence. For highly deceptive landscapes with very
# small effective basin sizes, the algorithm may need more budget than provided.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = _read_bounds(func, self.dim)

        # Handle degenerate case early
        if self.budget <= 0:
            x0 = np.clip(np.zeros(self.dim, dtype=float), lb, ub)
            return x0, float(_safe_eval(func, x0))

        # Choose population/offspring sizes to fit budget.
        # We use a modest population for robustness across dimensions.
        # Ensure at least 1 offspring per generation.
        pop_size = max(4, min(12, int(np.ceil(np.sqrt(self.dim + 1)))))
        off_size = max(2, min(18, pop_size))  # offspring per generation
        # Number of generations determined by budget:
        # evaluations per gen: off_size, but first generation needs pop_size evals.
        # We'll adjust remaining evaluations exactly to not exceed budget.
        max_gens = max(1, (self.budget - pop_size) // off_size + 1)

        # Initial center: if bounds are symmetric or wide, use midpoint.
        center = (lb + ub) * 0.5

        # Initial population: mix midpoint and random points.
        # Evaluate up to budget.
        evals_used = 0
        n_init = min(pop_size, self.budget)

        # Pre-allocate arrays
        pop = np.empty((n_init, self.dim), dtype=float)
        # Always include center if feasible
        pop[0] = np.clip(center, lb, ub)
        for i in range(1, n_init):
            pop[i] = _sample_uniform(lb, ub)

        vals = np.array([_safe_eval(func, pop[i]) for i in range(n_init)], dtype=float)
        evals_used += n_init

        # Incumbent best
        best_idx = int(np.argmin(vals))
        best_x = pop[best_idx].copy()
        best_y = float(vals[best_idx])

        # Coordinate-wise step size initialized from bounds scale
        span = ub - lb
        # Avoid zero span (tight bounds): set minimum step relative to numerical scale
        span_scale = np.maximum(span, 1e-12)
        step = 0.3 * span_scale

        # Stagnation tracking
        best_y_prev = best_y
        stall = 0
        # How many stalls trigger increased exploration
        stall_limit = 3

        # Determine generations with an exact budget accounting.
        gen = 0
        while evals_used < self.budget and gen < max_gens:
            gen += 1
            remaining = self.budget - evals_used
            m = min(off_size, remaining)
            if m <= 0:
                break

            # Determine whether to explore (uniform injection) due to stall
            # Stall increases exploration magnitude and fraction of uniform samples.
            explore_frac = 0.15
            if stall >= 1:
                explore_frac = min(0.6, explore_frac + 0.15 * stall)
            n_uniform = int(round(m * explore_frac))
            n_uniform = min(m, max(0, n_uniform))
            n_mut = m - n_uniform

            # Generate offspring
            offspring = np.empty((m, self.dim), dtype=float)
            o = 0

            # Uniform exploration samples (helps escape stagnation)
            for _ in range(n_uniform):
                offspring[o] = _sample_uniform(lb, ub)
                o += 1

            # Gaussian mutation around current best
            # Coordinate-wise: per-dimension standard deviation
            # When exploring, use larger noise; when exploiting, shrink.
            if stall == 0:
                noise_mult = 0.9
            else:
                noise_mult = 1.0 + 0.25 * min(4, stall)

            if n_mut > 0:
                # Draw normal noise: N(0, 1) then scale by step
                noise = np.random.randn(n_mut, self.dim) * (step * noise_mult)
                cand = best_x[None, :] + noise
                cand = np.clip(cand, lb, ub)
                offspring[o:o + n_mut] = cand

            # Evaluate offspring
            off_vals = np.empty(m, dtype=float)
            for i in range(m):
                off_vals[i] = _safe_eval(func, offspring[i])
            evals_used += m

            # Combine parents + offspring for elitist selection
            # Keep population size fixed (or as much as available in first round).
            combined = np.vstack([pop, offspring])
            combined_vals = np.concatenate([vals, off_vals])

            # Select best pop_size individuals
            keep = min(pop_size, combined.shape[0])
            order = np.argsort(combined_vals, kind="stable")
            keep_idx = order[:keep]
            pop = combined[keep_idx]
            vals = combined_vals[keep_idx]

            # Update incumbent and adaptation
            best_idx = int(np.argmin(vals))
            cand_best_x = pop[best_idx].copy()
            cand_best_y = float(vals[best_idx])

            improved = cand_best_y < best_y - 1e-12 * (1.0 + abs(best_y))
            if improved:
                # Exploit: shrink step when improvement happens
                best_x = cand_best_x
                best_y = cand_best_y
                stall = 0
                # Slight shrink to focus search; ensure nonzero
                step = np.maximum(step * 0.85, 1e-12)
            else:
                stall += 1
                # Exploration: broaden step gradually on stall
                # Also slightly decorrelate via per-dimension scaling
                step = np.maximum(step * (1.02 + 0.05 * min(5, stall)), 1e-12)

            # Secondary adaptation based on relative progress
            rel_improve = (best_y_prev - best_y) / (abs(best_y_prev) + 1e-12)
            if rel_improve > 0:
                best_y_prev = best_y

        return best_x, best_y


def _read_bounds(func, dim: int):
    # Support either func.lower/func.upper or func.bounds.lb/ub.
    if hasattr(func, "lower") and hasattr(func, "upper"):
        lb = np.asarray(func.lower, dtype=float)
        ub = np.asarray(func.upper, dtype=float)
    elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
    else:
        raise AttributeError(
            "func must provide either func.lower/func.upper or func.bounds.lb/func.bounds.ub"
        )

    if lb.shape == ():
        lb = np.full(dim, float(lb))
    if ub.shape == ():
        ub = np.full(dim, float(ub))
    lb = np.broadcast_to(lb, (dim,)).astype(float, copy=False)
    ub = np.broadcast_to(ub, (dim,)).astype(float, copy=False)

    # Ensure lb <= ub elementwise (swap if necessary)
    lo = np.minimum(lb, ub)
    hi = np.maximum(lb, ub)

    # Clamp to finite values if possible; if inf appears, fallback to centered finite region
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        # If bounds are infinite, use a reasonable scale around zero.
        # (This is a rare edge case; many benchmarks provide finite bounds.)
        lo = np.where(np.isfinite(lo), lo, -1.0)
        hi = np.where(np.isfinite(hi), hi, 1.0)

    return lo, hi


def _sample_uniform(lb, ub):
    # Sample uniformly from bounds (box).
    r = np.random.rand(lb.shape[0])
    return lb + r * (ub - lb)


def _safe_eval(func, x):
    # Objective is minimization: return func(x) as float.
    # If func errors due to shape, try passing a list.
    try:
        y = func(x)
    except Exception:
        y = func(x.tolist())
    # Ensure float conversion
    return float(np.asarray(y))
