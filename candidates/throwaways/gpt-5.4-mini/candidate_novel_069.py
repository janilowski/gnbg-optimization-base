# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Novel bounded minimizer with backbone=archive_eda, proposal=gaussian_step, explore=uniform_jump, adapt=variance_control.
# Search state: Best incumbent, small archive, step vector, success/failure counters, restart timer.
# Candidate generation: Mixes structured proposal, archive differences, exploratory jumps, and local repair.
# Selection and replacement: Uses age_replace replacement across archive and incumbent.
# Adaptation: variance_control adjusts step size from short success history.
# Exploration mechanisms: uniform_jump and restart-triggered global refresh.
# Exploitation mechanisms: Greedy incumbent update plus optional small_pattern polish.
# Boundary handling: Clip all points to lower/upper bounds before evaluation.
# Budget strategy: Guard every objective call with remaining-budget check.
# Closest known influences: DE, hill climbing, PSO, pattern search, EDA.
# Novelty or unusual aspects: Unique operator signature chosen before file write.
# Failure modes: Can slow on highly rotated, ill-conditioned landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        self.seed = 7259

    def __call__(self, func):
        rng = np.random.default_rng(self.seed + 37 * self.budget + 11 * self.dim)

        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        if lb.shape == ():
            lb = np.full(self.dim, float(lb))
        if ub.shape == ():
            ub = np.full(self.dim, float(ub))
        span = np.maximum(ub - lb, 1e-12)

        evals = 0
        def clip(x):
            return np.clip(x, lb, ub)
        def evaluate(x):
            nonlocal evals
            if evals >= self.budget:
                return np.inf
            y = float(func(clip(x)))
            evals += 1
            return y

        n = max(4, min(24, 11 + self.dim // 8))
        X = clip(lb + rng.random((n, self.dim)) * span)
        Y = np.empty(n, dtype=float)
        for i in range(n):
            if evals >= self.budget:
                break
            Y[i] = evaluate(X[i])
        ib = int(np.argmin(Y))
        x_best = X[ib].copy()
        y_best = float(Y[ib])
        sigma = np.full(self.dim, 0.03) * span
        succ = 0
        fail = 0
        it = 0

        while evals < self.budget:
            it += 1
            phase = (it + 6) % 7
            if phase == 0:
                z = x_best + rng.normal(0.0, 1.0, self.dim) * sigma
            elif phase == 1:
                a, b = rng.integers(0, n, 2)
                z = x_best + 0.12 * (X[a] - X[b])
            elif phase == 2:
                top = np.argsort(Y)[: max(2, n // 4)]
                w = rng.random(top.size)
                w /= (w.sum() + 1e-12)
                z = np.sum(X[top] * w[:, None], axis=0)
            elif phase == 3:
                j = int(rng.integers(0, self.dim))
                z = x_best.copy()
                z[j] += rng.normal(0.0, 1.0) * sigma[j]
            elif phase == 4:
                z = 0.5 * (x_best + X[int(rng.integers(0, n))])
            elif phase == 5:
                z = lb + rng.random(self.dim) * span
            else:
                z = x_best + rng.normal(0.0, 0.25, self.dim) * sigma

            if 'gaussian_step' == 'cauchy_step' and it % 11 == 0:
                z = x_best + rng.standard_cauchy(self.dim) * 0.02 * span
            elif 'gaussian_step' == 'differential_step' and it % 9 == 0:
                a, b = rng.integers(0, n, 2)
                c = rng.integers(0, n)
                z = X[a] + 0.5 * (X[b] - X[c])
            elif 'gaussian_step' == 'weighted_centroid' and it % 13 == 0:
                k = min(5, n)
                top = np.argsort(Y)[:k]
                z = X[top].mean(axis=0)
            elif 'gaussian_step' == 'axis_probe' and it % 7 == 0:
                j = int(rng.integers(0, self.dim))
                z = x_best.copy(); z[j] += rng.normal(0.0, 1.0) * span[j]
            elif 'gaussian_step' == 'random_subspace' and it % 8 == 0:
                mask = rng.random(self.dim) < max(0.2, 3.0 / max(3, self.dim))
                z = x_best.copy(); z[mask] = lb[mask] + rng.random(mask.sum()) * span[mask]
            elif 'gaussian_step' == 'opposition_step' and it % 10 == 0:
                c = 0.5 * (lb + ub)
                z = c + (c - x_best)
            elif 'gaussian_step' == 'reflect_step' and it % 12 == 0:
                z = np.where(x_best > 0.5 * (lb + ub), lb + (x_best - lb), ub - (ub - x_best))
            elif 'gaussian_step' == 'spiral_step' and it % 14 == 0:
                d = rng.normal(0.0, 1.0, self.dim)
                d /= (np.linalg.norm(d) + 1e-12)
                z = x_best + (0.2 + 0.3 * rng.random()) * d * span

            z = clip(z)
            yz = evaluate(z)
            if yz < y_best:
                x_best = z.copy(); y_best = yz; succ += 1; fail = 0
            else:
                fail += 1

            if 'variance_control' == 'one_fifth':
                if succ + fail >= 8:
                    rate = succ / max(1, succ + fail)
                    sigma = np.minimum(0.6 * span, sigma * (1.12 if rate > 0.2 else 0.87))
                    succ = 0; fail = 0
            elif 'variance_control' == 'success_ratio':
                if succ >= 3:
                    sigma = np.minimum(0.6 * span, sigma * 1.05); succ = 0
                if fail >= 4:
                    sigma = np.maximum(1e-8 * span, sigma * 0.92); fail = 0
            elif 'variance_control' == 'failure_decay':
                sigma = np.maximum(1e-8 * span, sigma * (0.98 if fail else 1.01))
            elif 'variance_control' == 'time_decay':
                t = evals / max(1, self.budget)
                sigma = np.maximum(1e-8 * span, sigma * (0.999 - 0.15 * t / max(1.0, self.dim)))
            elif 'variance_control' == 'rank_scaled':
                sigma = np.clip(sigma * (1.0 + 0.02 * (np.std(Y) + 1e-12)), 1e-8 * span, 0.6 * span)
            elif 'variance_control' == 'variance_control':
                v = float(np.var(Y)) if n > 1 else 0.0
                sigma = np.clip(sigma * (1.0 + 0.03 * np.tanh(v)), 1e-8 * span, 0.6 * span)
            elif 'variance_control' == 'temperature_control':
                temp = max(0.05, 1.0 - evals / max(1, self.budget))
                sigma = np.maximum(1e-8 * span, sigma * (0.995 + 0.01 * temp))
            else:
                sigma = np.clip(sigma * (1.03 if yz < y_best else 0.97), 1e-8 * span, 0.6 * span)

            if 'age_replace' == 'strict_elitist':
                w = int(np.argmax(Y))
                if yz < Y[w]:
                    X[w] = z; Y[w] = yz
            elif 'age_replace' == 'replace_worst':
                w = int(np.argmax(Y)); X[w] = z; Y[w] = yz
            elif 'age_replace' == 'tournament_replace':
                a, b = rng.integers(0, n, 2)
                w = a if Y[a] > Y[b] else b
                if yz < Y[w]: X[w] = z; Y[w] = yz
            elif 'age_replace' == 'age_replace':
                w = it % n
                if yz <= Y[w] or rng.random() < 0.1: X[w] = z; Y[w] = yz
            elif 'age_replace' == 'archive_insert':
                w = int(np.argmax(Y))
                if yz < Y[w] or rng.random() < 0.05: X[w] = z; Y[w] = yz
            else:
                w = int(rng.integers(0, n)); X[w] = z; Y[w] = yz

            if 'small_pattern' == 'coord_refine' and it % 7 == 0 and evals < self.budget:
                j = int(rng.integers(0, self.dim))
                p = x_best.copy(); p[j] += 0.4 * sigma[j]
                yp = evaluate(p)
                if yp < y_best: x_best, y_best = clip(p), yp
            elif 'small_pattern' == 'line_refine' and it % 9 == 0 and evals + 2 <= self.budget:
                d = rng.normal(0.0, 1.0, self.dim)
                d /= (np.linalg.norm(d) + 1e-12)
                p1 = clip(x_best + 0.3 * sigma * d)
                p2 = clip(x_best - 0.3 * sigma * d)
                y1 = evaluate(p1); y2 = evaluate(p2)
                if y1 < y_best or y2 < y_best:
                    x_best, y_best = (p1, y1) if y1 <= y2 else (p2, y2)
            elif 'small_pattern' == 'pair_refine' and it % 11 == 0 and evals + 2 <= self.budget:
                j = int(rng.integers(0, self.dim))
                p1 = x_best.copy(); p2 = x_best.copy()
                p1[j] += 0.25 * sigma[j]; p2[j] -= 0.25 * sigma[j]
                y1 = evaluate(p1); y2 = evaluate(p2)
                if y1 < y_best or y2 < y_best:
                    x_best, y_best = (clip(p1), y1) if y1 <= y2 else (clip(p2), y2)
            elif 'small_pattern' == 'small_pattern' and it % 13 == 0 and evals + 2 <= self.budget:
                p1 = clip(x_best + rng.normal(0.0, 0.1, self.dim) * span)
                p2 = clip(x_best - rng.normal(0.0, 0.1, self.dim) * span)
                y1 = evaluate(p1); y2 = evaluate(p2)
                if y1 < y_best or y2 < y_best:
                    x_best, y_best = (p1, y1) if y1 <= y2 else (p2, y2)
            elif 'small_pattern' == 'orth_refine' and it % 15 == 0 and evals + 2 <= self.budget:
                d = rng.normal(0.0, 1.0, self.dim)
                d /= (np.linalg.norm(d) + 1e-12)
                p1 = clip(x_best + 0.35 * sigma * d)
                p2 = clip(x_best - 0.35 * sigma * d)
                y1 = evaluate(p1); y2 = evaluate(p2)
                if y1 < y_best or y2 < y_best:
                    x_best, y_best = (p1, y1) if y1 <= y2 else (p2, y2)

            do_restart = False
            if 'population_reset' == 'fixed_patience':
                do_restart = fail >= 29
            elif 'population_reset' == 'entropy_patience':
                do_restart = fail >= 14 and float(np.std(Y)) < 1e-9
            elif 'population_reset' == 'budget_quarter':
                do_restart = (evals > self.budget // 4) and (it % max(10, 29//2) == 0) and fail > 3
            elif 'population_reset' == 'stall_and_spread':
                do_restart = fail >= 29 and float(np.max(Y) - np.min(Y)) < 1e-6
            elif 'population_reset' == 'progress_timeout':
                do_restart = (it % max(12, 29//3) == 0) and succ == 0
            else:
                do_restart = (it % max(14, 29//2) == 0) and fail > 4

            if do_restart and evals < self.budget:
                xr = clip(lb + rng.random(self.dim) * span)
                yr = evaluate(xr)
                sigma = np.full(self.dim, 0.03) * span
                succ = 0; fail = 0
                w = int(np.argmax(Y)); X[w] = xr; Y[w] = yr
                if yr < y_best: x_best, y_best = xr.copy(), yr

            if 'uniform_jump' == 'latin_jump' and it % 17 == 0 and evals < self.budget:
                xr = clip(lb + (rng.random(self.dim) + np.linspace(0, 1, self.dim)) % 1.0 * span)
                yr = evaluate(xr)
                if yr < y_best: x_best, y_best = xr.copy(), yr
            elif 'uniform_jump' == 'opposition_jump' and it % 19 == 0 and evals < self.budget:
                c = 0.5 * (lb + ub)
                xr = clip(c + (c - x_best) + rng.normal(0.0, 0.02, self.dim) * span)
                yr = evaluate(xr)
                if yr < y_best: x_best, y_best = xr.copy(), yr
            elif 'uniform_jump' == 'cauchy_jump' and it % 23 == 0 and evals < self.budget:
                xr = clip(x_best + rng.standard_cauchy(self.dim) * 0.01 * span)
                yr = evaluate(xr)
                if yr < y_best: x_best, y_best = xr.copy(), yr
            elif 'uniform_jump' == 'axis_long_jump' and it % 29 == 0 and evals < self.budget:
                j = int(rng.integers(0, self.dim))
                xr = x_best.copy(); xr[j] = lb[j] + rng.random() * span[j]
                yr = evaluate(xr)
                if yr < y_best: x_best, y_best = xr.copy(), yr
            elif 'uniform_jump' == 'scramble_jump' and it % 31 == 0 and evals < self.budget:
                xr = x_best.copy(); rng.shuffle(xr)
                yr = evaluate(xr)
                if yr < y_best: x_best, y_best = xr.copy(), yr

        return x_best, float(y_best)
