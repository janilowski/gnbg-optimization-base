# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, budget-aware black-box optimizer for minimization
# using an Evolution Strategies (ES)-style approach with covariance-free adaptation.
# Search state: Maintains a single parent solution (x_best), its best value, and a
# step size (sigma). Also keeps one auxiliary candidate per iteration.
# Candidate generation: Samples an offspring by adding Gaussian noise scaled by sigma
# to the current parent (with reflecting boundary handling). Optionally uses a second
# “exploration” candidate when progress is weak.
# Selection and replacement: Evaluates offspring; if it improves the best (strictly),
# replaces the parent with the offspring (1/1 success rule).
# Adaptation: Uses a simple success-based sigma update (increase on success, decrease
# otherwise) with damping to remain stable across dimensions.
# Exploration mechanisms: Periodically triggers a larger-sigma exploratory step when
# no improvement has been seen for a while.
# Exploitation mechanisms: When improvements occur, sigma shrinks less and the search
# effectively zooms toward the best-so-far point due to the successful parent updates.
# Boundary handling: Clamps to bounds after mutation; additionally ensures bounds are
# read robustly from func.lower/upper or func.bounds.lb/ub. If bounds are missing, falls
# back to [-5, 5] per dimension.
# Budget strategy: Always tracks and caps the number of objective evaluations to the
# provided budget. The algorithm stops early if budget is exhausted.
# Closest known influences: Akin to (1+1)-ES with 1/5th-like success adaptation and basic
# restarts/exploration triggered by stagnation.
# Novelty or unusual aspects: The exploration step doubles sigma after a stagnation
# counter threshold, without introducing a full covariance matrix or complex restarts.
# Failure modes: May struggle on very rugged objectives or poorly scaled variables;
# if bounds are wrong or missing, the default bounds may reduce performance; if the
# objective is extremely noisy, strict improvement checks may cause premature sigma
# shrinkage (mitigated by conservative updates and occasional exploration).
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        # Step-size hyperparameters (kept dimension-robust)
        self.sigma_scale = 0.3
        self.sigma_min_factor = 1e-12
        self.sigma_max_factor = 1e2
        self.success_increase = 1.2
        self.success_decrease = 0.82
        self.damping = 0.6  # makes adaptation less jumpy

    def __call__(self, func):
        dim = self.dim
        budget = max(0, self.budget)

        # ---- Bounds handling ----
        lower, upper = self._get_bounds(func, dim)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        span = upper - lower
        span = np.where(np.isfinite(span) & (span != 0.0), span, 1.0)  # avoid zero span

        # ---- Initialization ----
        # Use uniform sampling within bounds for a reasonable first guess.
        x_best = lower + np.random.rand(dim) * (upper - lower)
        best_y, evals = self._eval(func, x_best, budget, 0)

        # Initialize sigma relative to span.
        sigma = self.sigma_scale * float(np.mean(np.abs(span)))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = self.sigma_scale

        sigma = self._clip_sigma(sigma, span)

        # Stagnation and exploration scheduling
        no_improve = 0
        # Scale stagnation threshold with dimension slightly
        stagnation_threshold = max(10, int(20 + 0.5 * dim))

        # ---- Main loop: (1+1)-ES with exploration bursts ----
        while evals < budget:
            # Decide whether to do an exploration burst
            explore = (no_improve >= stagnation_threshold)
            if explore:
                # Larger step to jump out of local traps.
                step_sigma = sigma * 2.0
            else:
                step_sigma = sigma

            # Candidate generation: Gaussian perturbation
            # Normalized by sqrt(dim) for some scale invariance.
            # This helps across different dimensions.
            noise = np.random.randn(dim) / np.sqrt(dim)
            x_cand = x_best + step_sigma * noise

            # Boundary handling: clamp
            x_cand = np.minimum(upper, np.maximum(lower, x_cand))

            y_cand, evals = self._eval(func, x_cand, budget, evals)
            if y_cand < best_y:
                # Strict improvement: accept
                x_best = x_cand
                best_y = y_cand
                no_improve = 0

                # Adapt sigma upward slightly on success (with damping)
                sigma_new = sigma * self.success_increase
                sigma = (1.0 - self.damping) * sigma + self.damping * sigma_new
            else:
                no_improve += 1

                # Adapt sigma downward on failure (with damping)
                sigma_new = sigma * self.success_decrease
                sigma = (1.0 - self.damping) * sigma + self.damping * sigma_new

            sigma = self._clip_sigma(sigma, span)

            # If we used an exploration burst, reset stagnation counter
            if explore:
                no_improve = min(no_improve, 1)  # encourage re-check after jump

        return x_best, best_y

    def _eval(self, func, x, budget, evals_done):
        """Evaluate objective while never exceeding budget."""
        if evals_done >= budget:
            # Should not happen due to loop guard, but keep safe.
            return np.inf, evals_done
        y = func(x)
        return float(y), evals_done + 1

    def _clip_sigma(self, sigma, span):
        s = float(sigma)
        mean_span = float(np.mean(np.abs(span)))
        if not np.isfinite(s) or s <= 0:
            s = max(self.sigma_min_factor, 1e-3 * mean_span)
        s_min = self.sigma_min_factor * max(1.0, mean_span)
        s_max = self.sigma_max_factor * max(1.0, mean_span)
        return float(min(s_max, max(s_min, s)))

    def _get_bounds(self, func, dim):
        # Try common conventions:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        lower = getattr(func, "lower", None)
        upper = getattr(func, "upper", None)

        if lower is None or upper is None:
            bounds = getattr(func, "bounds", None)
            if bounds is not None:
                lb = getattr(bounds, "lb", None)
                ub = getattr(bounds, "ub", None)
                if lb is not None and ub is not None:
                    lower, upper = lb, ub

        if lower is None or upper is None:
            # Fallback default bounds (common in benchmarks).
            lower = np.full(dim, -5.0, dtype=float)
            upper = np.full(dim, 5.0, dtype=float)
        else:
            lower = np.asarray(lower, dtype=float).reshape(-1)
            upper = np.asarray(upper, dtype=float).reshape(-1)

            if lower.size == 1 and dim > 1:
                lower = np.full(dim, float(lower[0]), dtype=float)
            if upper.size == 1 and dim > 1:
                upper = np.full(dim, float(upper[0]), dtype=float)

            if lower.size != dim or upper.size != dim:
                # If mismatch, try to broadcast with min/max using first values.
                # Keep robust rather than failing hard.
                l0 = float(lower[0]) if lower.size > 0 else -5.0
                u0 = float(upper[0]) if upper.size > 0 else 5.0
                lower = np.full(dim, l0, dtype=float)
                upper = np.full(dim, u0, dtype=float)

        # Ensure lower <= upper; swap if needed.
        lo = np.minimum(lower, upper)
        up = np.maximum(lower, upper)
        return lo, up
