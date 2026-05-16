# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Novel signature search variant using triangular_mix + latin_like_batch + temp_decay with age_replace replacement and entropy_patience restarts.
# Search state: Best point/value, small population archive, step vector, success/failure counters.
# Candidate generation: Rotates among local perturbation, directional differences, elite mixing, and scheduled exploration jumps.
# Selection and replacement: Applies age_replace policy over trial and archive points.
# Adaptation: Uses temp_decay rule to resize mutation step scales online.
# Exploration mechanisms: Uses latin_like_batch branch plus periodic random refreshes.
# Exploitation mechanisms: Greedy incumbent updates with short-horizon local refinement (none).
# Boundary handling: Clips every trial to bounds from func.lower/upper or func.bounds.lb/ub.
# Budget strategy: Single guarded evaluate wrapper; all loops check budget before objective call.
# Closest known influences: Hill climbing, DE, PSO micro-population, pattern search.
# Novelty or unusual aspects: Signature composition with distinct operator schedule and restart trigger.
# Failure modes: Can stall on deceptive high-conditioning landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        self.seed = 2508

    def __call__(self, func):
        rng = np.random.default_rng(self.seed + self.budget * 13 + self.dim * 17)

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

        def clip(x):
            return np.clip(x, lb, ub)

        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= self.budget:
                return np.inf
            y = float(func(clip(x)))
            evals += 1
            return y

        pop_n = max(4, min(24, 14 + self.dim // 10))
        X = clip(lb + rng.random((pop_n, self.dim)) * span)
        Y = np.empty(pop_n, dtype=float)
        for i in range(pop_n):
            if evals >= self.budget:
                break
            Y[i] = evaluate(X[i])

        b = int(np.argmin(Y))
        x_best = X[b].copy()
        y_best = float(Y[b])

        sigma = np.full(self.dim, 0.08, dtype=float) * span
        succ = 0
        fail = 0
        it = 0

        while evals < self.budget:
            it += 1
            phase = (it + 16) % 6

            if phase == 0:
                z = x_best + rng.normal(0.0, 1.0, self.dim) * sigma
            elif phase == 1:
                i1, i2 = rng.integers(0, pop_n, 2)
                z = x_best + 0.15 * (X[i1] - X[i2]) + rng.normal(0.0, 0.05, self.dim) * span
            elif phase == 2:
                elite = np.argsort(Y)[: max(2, pop_n // 4)]
                w = rng.random(elite.size)
                w = w / (w.sum() + 1e-12)
                z = np.sum(X[elite] * w[:, None], axis=0) + rng.normal(0.0, 0.03, self.dim) * span
            elif phase == 3:
                j = int(rng.integers(0, self.dim))
                z = x_best.copy()
                z[j] += rng.normal(0.0, 1.0) * sigma[j]
            elif phase == 4:
                z = x_best + rng.normal(0.0, 0.2, self.dim) * sigma
            else:
                z = lb + rng.random(self.dim) * span

            if 3 == 1 and (it % 13 == 0):
                z = x_best + rng.standard_cauchy(self.dim) * 0.02 * span
            elif 3 == 2 and (it % 11 == 0):
                center = 0.5 * (lb + ub)
                z = center + (center - x_best) + rng.normal(0.0, 0.02, self.dim) * span
            elif 3 == 3 and (it % 17 == 0):
                B = clip(lb + rng.random((4, self.dim)) * span)
                z = B.mean(axis=0)
            elif 3 == 4 and (it % 19 == 0):
                j = int(rng.integers(0, self.dim))
                z = x_best.copy()
                z[j] = lb[j] + rng.random() * span[j]

            z = clip(z)
            yz = evaluate(z)

            improved = yz < y_best
            if improved:
                x_best = z.copy()
                y_best = yz
                succ += 1
                fail = 0
            else:
                fail += 1

            if 3 == 0:
                if (succ + fail) >= 8:
                    rate = succ / max(1, succ + fail)
                    if rate > 0.22:
                        sigma = np.minimum(0.6 * span, sigma * 1.12)
                    else:
                        sigma = np.maximum(1e-8 * span, sigma * 0.86)
                    succ = 0
                    fail = 0
            elif 3 == 1:
                if succ >= 3:
                    sigma = np.minimum(0.6 * span, sigma * 1.08)
                    succ = 0
                if fail >= 5:
                    sigma = np.maximum(1e-8 * span, sigma * 0.9)
                    fail = 0
            elif 3 == 2:
                if not improved:
                    sigma = np.maximum(1e-8 * span, sigma * 0.95)
                else:
                    sigma = np.minimum(0.6 * span, sigma * 1.03)
            elif 3 == 3:
                t = evals / max(1, self.budget)
                sigma = np.maximum(1e-8 * span, sigma * (0.999 - 0.15 * t / max(1.0, self.dim)))
            else:
                rank = np.argsort(np.argsort(Y))
                spread = 1.0 + 0.3 * (rank.mean() / max(1, pop_n - 1))
                sigma = np.clip(sigma * spread, 1e-8 * span, 0.6 * span)

            if 3 == 0:
                w = int(np.argmax(Y))
                if yz < Y[w]:
                    X[w] = z
                    Y[w] = yz
            elif 3 == 1:
                w = int(np.argmax(Y))
                X[w] = z
                Y[w] = yz
            elif 3 == 2:
                a, b2 = rng.integers(0, pop_n, 2)
                t = a if Y[a] > Y[b2] else b2
                if yz < Y[t]:
                    X[t] = z
                    Y[t] = yz
            else:
                idx = it % pop_n
                if yz <= Y[idx] or rng.random() < 0.1:
                    X[idx] = z
                    Y[idx] = yz

            if 0 == 1 and (it % 7 == 0) and evals < self.budget:
                j = int(rng.integers(0, self.dim))
                p1 = x_best.copy(); p1[j] += 0.5 * sigma[j]
                y1 = evaluate(p1)
                if y1 < y_best:
                    x_best, y_best = clip(p1), y1
                elif evals < self.budget:
                    p2 = x_best.copy(); p2[j] -= 0.5 * sigma[j]
                    y2 = evaluate(p2)
                    if y2 < y_best:
                        x_best, y_best = clip(p2), y2
            elif 0 == 2 and (it % 9 == 0) and evals + 2 <= self.budget:
                d = rng.normal(0.0, 1.0, self.dim)
                d /= (np.linalg.norm(d) + 1e-12)
                p1 = clip(x_best + 0.4 * sigma * d)
                p2 = clip(x_best - 0.4 * sigma * d)
                y1 = evaluate(p1)
                y2 = evaluate(p2)
                if y1 < y_best or y2 < y_best:
                    if y1 <= y2:
                        x_best, y_best = p1, y1
                    else:
                        x_best, y_best = p2, y2
            elif 0 == 3 and (it % 12 == 0) and evals + 2 <= self.budget:
                d = rng.normal(0.0, 1.0, self.dim)
                d /= (np.linalg.norm(d) + 1e-12)
                a = clip(x_best + 0.3 * sigma * d)
                c = clip(x_best - 0.3 * sigma * d)
                ya = evaluate(a)
                yc = evaluate(c)
                if ya < y_best or yc < y_best:
                    if ya <= yc:
                        x_best, y_best = a, ya
                    else:
                        x_best, y_best = c, yc

            need_restart = False
            if 1 == 0:
                need_restart = fail >= 22
            elif 1 == 1:
                need_restart = fail >= 11 and float(np.std(Y)) < 1e-9
            elif 1 == 2:
                need_restart = (evals > self.budget // 4) and (it % max(10, 22//2) == 0) and fail > 3
            else:
                need_restart = (fail >= 22) or ((it % max(10, 22//2) == 0) and succ == 0)

            if need_restart and evals < self.budget:
                xr = clip(lb + rng.random(self.dim) * span)
                yr = evaluate(xr)
                fail = 0
                succ = 0
                sigma = np.full(self.dim, 0.08, dtype=float) * span
                w = int(np.argmax(Y))
                X[w] = xr
                Y[w] = yr
                if yr < y_best:
                    x_best, y_best = xr.copy(), yr

        return x_best, float(y_best)
