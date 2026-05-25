# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# that maintains a population of candidate solutions and uses a simple,
# budget-aware evolutionary strategy (ES) with coordinate-wise mutation.
# Search state: The algorithm keeps a population (mu candidates) and tracks
# their objective values, remembering the best-so-far point across all
# evaluations. It also maintains an adaptive mutation scale (sigma).
# Candidate generation: Each generation, offspring are created by perturbing
# selected parents using Gaussian noise scaled by sigma, plus a small
# probability of coordinate-wise “jumps” to encourage escape from local minima.
# Selection and replacement: The algorithm uses elitist (mu+lambda) selection:
# it evaluates offspring, merges them with the parent population, and keeps
# the best mu by objective value (minimization).
# Adaptation: sigma is adapted using a simple success heuristic: if the best
# offspring improves the current best, sigma is increased slightly; otherwise it
# is decreased. This helps balance exploration and exploitation.
# Exploration mechanisms: Occasional coordinate-wise jumps and larger sigma
# during improvements help explore new regions.
# Exploitation mechanisms: Elitism and selecting from the current best-ranked
# parents focus search around promising areas.
# Boundary handling: After mutation, candidates are clamped to the provided
# lower/upper bounds (read from func.lower/func.upper or func.bounds.lb/ub).
# Budget strategy: The algorithm never evaluates more than the provided budget.
# It stops when the remaining budget is insufficient for the next batch.
# Closest known influences: The overall structure is inspired by classic
# evolution strategies (ES) with sigma adaptation and elitist selection, but
# kept deliberately lightweight for a benchmark harness.
# Novelty or unusual aspects: The mutation includes both isotropic Gaussian
# steps and occasional coordinate-wise jumps, with a success-based sigma
# adaptation that is robust across dimensions.
# Failure modes: If the objective is very noisy or poorly scaled, sigma
# adaptation may oscillate; boundary clamping can also reduce effective
# movement in tight bounds.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return something deterministic within bounds if possible.
            lower, upper = self._read_bounds(func, dim)
            x0 = (lower + upper) / 2.0
            return x0, float("inf")

        lower, upper = self._read_bounds(func, dim)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        # Protect against degenerate bounds
        span = np.maximum(upper - lower, 0.0)
        span_pos = np.where(span > 0, span, 1.0)

        # Population sizes: keep small/fast but enough diversity.
        # These are function of dimension to remain robust.
        mu = max(4, min(12, dim + 2))
        lam = max(8, min(24, 2 * dim + 4))
        # Ensure we don't exceed budget when evaluating initial + one generation.
        # We count evaluations precisely via eval_count.
        max_evals = budget

        eval_count = 0

        rng = np.random  # harness sets global seed for numpy

        def clamp(x):
            return np.minimum(np.maximum(x, lower), upper)

        # Initialize population uniformly within bounds.
        # If bounds are degenerate, all candidates become the same.
        X = lower + rng.rand(mu, dim) * span_pos
        X = clamp(X)

        # Evaluate initial population
        Y = np.empty(mu, dtype=float)
        for i in range(mu):
            if eval_count >= max_evals:
                # Should not happen due to sizing, but be safe.
                best_idx = int(np.argmin(Y[:i])) if i > 0 else 0
                return X[best_idx].copy(), float(Y[best_idx])
            y = func(X[i])
            y = float(y)
            Y[i] = y
            eval_count += 1

        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])

        # Adaptive sigma: start as a fraction of average span.
        avg_span = float(np.mean(span_pos))
        sigma = 0.25 * avg_span / np.sqrt(dim) if dim > 0 else 0.25 * avg_span

        # Success heuristic bounds
        sigma_min = 1e-12
        sigma_max = max(1.0, avg_span) * 2.0

        # Coordinate-wise jump probability increases as sigma decreases.
        # This helps escape when exploitation becomes too strong.
        jump_base_p = 1.0 / max(10.0, float(dim))

        # Main loop: keep evolving until budget is exhausted.
        while eval_count < max_evals:
            # Determine how many offspring we can evaluate this round.
            remaining = max_evals - eval_count
            # We evaluate offspring in a batch; ensure we don't exceed budget.
            k = min(lam, remaining)
            if k <= 0:
                break

            # Select parents: pick from top-ranked individuals.
            order = np.argsort(Y)
            top_k = min(mu, max(2, mu // 2))
            parents = order[:top_k]

            # Create offspring
            X_off = np.empty((k, dim), dtype=float)
            for j in range(k):
                p_idx = int(rng.choice(parents))
                x_parent = X[p_idx]

                # Isotropic Gaussian step
                step = rng.randn(dim) * sigma

                # Occasional coordinate-wise jump to increase exploration.
                # Jump magnitude is proportional to span.
                p_jump = float(jump_base_p) * (sigma_max / max(sigma, sigma_min)) ** (-0.5)
                if p_jump < 1e-6:
                    p_jump = 0.0
                if p_jump > 1.0:
                    p_jump = 1.0

                if dim > 0 and p_jump > 0.0:
                    mask = rng.rand(dim) < p_jump
                    if np.any(mask):
                        # Random jump direction per chosen coordinate
                        # magnitude: fraction of span.
                        magn = (0.5 + rng.rand(np.sum(mask))) * (span_pos[mask] if np.any(mask) else 1.0)
                        direction = rng.choice([-1.0, 1.0], size=np.sum(mask))
                        step = step.copy()
                        step[mask] += direction * (0.25 * magn)

                x_new = x_parent + step
                X_off[j] = clamp(x_new)

            # Evaluate offspring
            Y_off = np.empty(k, dtype=float)
            for j in range(k):
                if eval_count >= max_evals:
                    k = j
                    Y_off = Y_off[:k]
                    X_off = X_off[:k]
                    break
                y = func(X_off[j])
                Y_off[j] = float(y)
                eval_count += 1

            if k == 0:
                break

            # Track success for sigma adaptation
            off_best_idx = int(np.argmin(Y_off))
            off_best_y = float(Y_off[off_best_idx])
            improved = off_best_y < best_y

            if improved:
                best_y = off_best_y
                best_x = X_off[off_best_idx].copy()
                sigma = max(sigma_min, sigma * 1.1)  # can explore more after success
            else:
                sigma = min(sigma_max, sigma * 0.85)  # shrink if no progress

            # Elitist replacement (mu + k) -> keep best mu
            # Merge
            X_all = np.vstack([X, X_off[:k]])
            Y_all = np.concatenate([Y, Y_off[:k]])

            order_all = np.argsort(Y_all)
            keep = order_all[:mu]
            X = X_all[keep].copy()
            Y = Y_all[keep].copy()

        return best_x, best_y

    @staticmethod
    def _read_bounds(func, dim):
        # Bounds are expected in one of two formats:
        # 1) func.lower / func.upper
        # 2) func.bounds.lb / func.bounds.ub
        lower = None
        upper = None

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = func.lower
            upper = func.upper
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lower = b.lb
                upper = b.ub

        if lower is None or upper is None:
            # Fallback: try common names, else use unbounded-ish defaults.
            # Benchmark harness should provide proper bounds; this is just robustness.
            lower = getattr(func, "lb", None)
            upper = getattr(func, "ub", None)
            if lower is None or upper is None:
                lower = -5.0 * np.ones(dim, dtype=float)
                upper = 5.0 * np.ones(dim, dtype=float)

        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)

        if lower.size == 1 and dim > 1:
            lower = np.full(dim, float(lower[0]))
        if upper.size == 1 and dim > 1:
            upper = np.full(dim, float(upper[0]))

        if lower.size != dim or upper.size != dim:
            # Try to coerce to correct length
            if lower.size < dim:
                lower = np.resize(lower, dim)
            else:
                lower = lower[:dim]
            if upper.size < dim:
                upper = np.resize(upper, dim)
            else:
                upper = upper[:dim]

        # Ensure proper ordering (lb <= ub)
        lb = np.minimum(lower, upper)
        ub = np.maximum(lower, upper)
        return lb, ub
