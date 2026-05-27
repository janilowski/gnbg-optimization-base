from __future__ import annotations

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A deterministic-probe local optimizer that estimates a useful basin, then spends the budget on coordinate line searches.
# Search state: It keeps the incumbent point/value, an evaluation counter, bounds, and shrinking per-coordinate search widths.
# Candidate generation: It evaluates center/axis/diagonal anchors, a quadratic axis estimate, golden-section coordinate points, and a few fallback mutations.
# Selection and replacement: Every evaluated point updates the global incumbent when it improves the objective; line-search intervals keep the better side.
# Adaptation: Coordinate windows shrink after each sweep, faster when a full sweep fails to improve the incumbent.
# Exploration mechanisms: Center, symmetric axis probes, diagonal probes, damped quadratic estimates, and late random perturbations cover several broad regions.
# Exploitation mechanisms: Repeated golden-section coordinate searches refine the best point quickly under small budgets.
# Boundary handling: All proposed points are clipped to the lower and upper bounds before evaluation.
# Budget strategy: A single guarded evaluator counts calls, and each multi-point operation checks the remaining budget before starting.
# Closest known influences: Coordinate descent, golden-section line search, quadratic interpolation from symmetric probes, pattern search.
# Novelty or unusual aspects: It converts one set of symmetric axis probes into a full-vector minimizer estimate before coordinate refinement.
# Failure modes: It is weak on strongly rotated or highly multimodal landscapes where coordinate-wise progress points at a poor local basin.
# ALGORITHM_ANALYSIS_NOTE_END

import math

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lower, upper = self._bounds(func)
        span = np.maximum(upper - lower, 1e-12)
        center = lower + 0.5 * span

        evals = 0
        best_x = center.copy()
        best_y = float("inf")

        def clip(x):
            return np.minimum(np.maximum(np.asarray(x, dtype=float), lower), upper)

        def remaining():
            return self.budget - evals

        def evaluate(x):
            nonlocal evals, best_x, best_y
            if evals >= self.budget:
                return None
            x = clip(x)
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        if self.budget <= 0:
            return best_x, best_y

        center_y = evaluate(center)
        if center_y is None:
            return best_x, best_y

        # Symmetric axis probes are expensive but very informative for the
        # short GNBG quick budget: on separable bowls they imply the minimizer.
        if remaining() >= 2 * self.dim + 1:
            step = 0.25 * span
            y_plus = np.empty(self.dim, dtype=float)
            y_minus = np.empty(self.dim, dtype=float)

            for j in range(self.dim):
                xp = center.copy()
                xm = center.copy()
                xp[j] += step[j]
                xm[j] -= step[j]
                y_plus[j] = evaluate(xp)
                y_minus[j] = evaluate(xm)

            with np.errstate(over="ignore", invalid="ignore"):
                estimate = center + (y_minus - y_plus) / (4.0 * step)
            estimate = clip(np.where(np.isfinite(estimate), estimate, center))
            evaluate(estimate)

            direction = estimate - center
            for scale in (0.25, 0.50, 0.75, 1.25):
                if remaining() <= 0:
                    break
                evaluate(center + scale * direction)

        # Whole-box diagonal anchors are a cheap hedge against shifted basins.
        for scale in (-0.45, -0.35, 0.35, 0.45):
            if remaining() <= 0:
                break
            evaluate(center + scale * span)

        # Coordinate golden-section search.  It is intentionally restarted from
        # the current incumbent after every coordinate because any coordinate
        # can move the best point.
        golden = (math.sqrt(5.0) - 1.0) / 2.0
        widths = span.copy()
        while remaining() > 0 and np.max(widths) > 1e-9 * np.max(span):
            before = best_y
            for j in np.random.permutation(self.dim):
                if remaining() <= 8:
                    break

                x0 = best_x.copy()
                a = max(lower[j], x0[j] - widths[j])
                b = min(upper[j], x0[j] + widths[j])
                if b <= a:
                    continue

                c = b - golden * (b - a)
                d = a + golden * (b - a)
                xc = x0.copy()
                xd = x0.copy()
                xc[j] = c
                xd[j] = d
                yc = evaluate(xc)
                yd = evaluate(xd)
                if yc is None or yd is None:
                    break

                for _ in range(6):
                    if remaining() <= 0:
                        break
                    if yc < yd:
                        b = d
                        d = c
                        yd = yc
                        c = b - golden * (b - a)
                        xt = best_x.copy()
                        xt[j] = c
                        yc = evaluate(xt)
                    else:
                        a = c
                        c = d
                        yc = yd
                        d = a + golden * (b - a)
                        xt = best_x.copy()
                        xt[j] = d
                        yd = evaluate(xt)
                    if yc is None or yd is None:
                        break

                if best_y <= 1e-10:
                    return best_x, float(best_y)

            widths *= 0.62 if best_y < before else 0.45

            # When line search stalls, spend a tiny amount on incumbent-centered
            # perturbations before shrinking into a possibly wrong coordinate box.
            if best_y >= before and remaining() > 0:
                sigma = 0.08 * widths
                for _ in range(min(4, remaining())):
                    evaluate(best_x + np.random.normal(0.0, 1.0, self.dim) * sigma)

        return best_x, float(best_y)

    def _bounds(self, func):
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        else:
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)

        if lower.size == 1 and self.dim > 1:
            lower = np.full(self.dim, float(lower.item()))
        if upper.size == 1 and self.dim > 1:
            upper = np.full(self.dim, float(upper.item()))
        return lower.reshape(-1), upper.reshape(-1)
