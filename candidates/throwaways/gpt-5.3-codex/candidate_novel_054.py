# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Signature-guided optimizer: spiral_local + levy_like_jump + oscillating; selection=age_replace; restart=budget_quarter; local=line_probe.
# Search state: Best incumbent, archive population, adaptive sigma, counters for success/failure.
# Candidate generation: Combines incumbent perturbation, archive differences, weighted elite averages, and exploration jumps.
# Selection and replacement: Uses age_replace rule to maintain compact archive.
# Adaptation: oscillating updates sigma to balance global/local moves.
# Exploration mechanisms: levy_like_jump schedule and periodic refresh.
# Exploitation mechanisms: Greedy incumbent update with optional line_probe step.
# Boundary handling: Clips all candidate vectors to problem bounds.
# Budget strategy: evaluate() guards objective calls; no call after budget exhausted.
# Closest known influences: DE, hill climbing, pattern search, restart metaheuristics.
# Novelty or unusual aspects: Unique operator signature enforced before file creation.
# Failure modes: Can underperform on strong rotation/non-separable valleys.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        self.seed = 5111

    def __call__(self, func):
        rng = np.random.default_rng(self.seed + 31 * self.budget + 7 * self.dim)
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

        n = max(4, min(26, 16 + self.dim // 9))
        X = clip(lb + rng.random((n, self.dim)) * span)
        Y = np.empty(n, dtype=float)
        for i in range(n):
            if evals >= self.budget:
                break
            Y[i] = evaluate(X[i])

        ib = int(np.argmin(Y))
        x_best = X[ib].copy()
        y_best = float(Y[ib])
        sigma = np.full(self.dim, 0.051000000000000004) * span
        succ = 0
        fail = 0
        it = 0

        while evals < self.budget:
            it += 1
            m = (it + 3) % 7
            if m == 0:
                z = x_best + rng.normal(0.0, 1.0, self.dim) * sigma
            elif m == 1:
                a, b = rng.integers(0, n, 2)
                z = X[a] + 0.38 * (X[a] - X[b])
            elif m == 2:
                top = np.argsort(Y)[: max(2, n // 4)]
                w = rng.random(top.size); w /= (w.sum() + 1e-12)
                z = np.sum(X[top] * w[:, None], axis=0) + rng.normal(0.0, 0.04, self.dim) * span
            elif m == 3:
                j = int(rng.integers(0, self.dim)); z = x_best.copy(); z[j] += rng.normal(0.0, 1.0) * sigma[j]
            elif m == 4:
                a, b, c = rng.integers(0, n, 3)
                z = x_best + 0.5 * (X[a] - X[b]) + 0.25 * (X[c] - x_best)
            elif m == 5:
                z = 0.5 * (x_best + X[int(rng.integers(0, n))]) + rng.normal(0.0, 0.03, self.dim) * span
            else:
                z = lb + rng.random(self.dim) * span

            if 1 == 1 and it % 13 == 0:
                z = x_best + rng.standard_cauchy(self.dim) * 0.015 * span
            elif 1 == 2 and it % 11 == 0:
                c = 0.5 * (lb + ub); z = c + (c - x_best)
            elif 1 == 3 and it % 17 == 0:
                B = clip(lb + rng.random((5, self.dim)) * span); z = B.mean(axis=0)
            elif 1 == 4 and it % 19 == 0:
                j = int(rng.integers(0, self.dim)); z = x_best.copy(); z[j] = lb[j] + rng.random() * span[j]
            elif 1 == 5 and it % 23 == 0:
                z = x_best.copy(); perm = rng.permutation(self.dim); z = z[perm]

            z = clip(z)
            yz = evaluate(z)
            imp = yz < y_best
            if imp:
                x_best = z.copy(); y_best = yz; succ += 1; fail = 0
            else:
                fail += 1

            if 5 == 0:
                if succ + fail >= 10:
                    rate = succ / max(1, succ + fail)
                    sigma = np.minimum(0.7 * span, sigma * (1.1 if rate > 0.2 else 0.87))
                    succ = 0; fail = 0
            elif 5 == 1:
                if imp: sigma = np.minimum(0.7 * span, sigma * 1.02)
                else: sigma = np.maximum(1e-8 * span, sigma * 0.97)
            elif 5 == 2:
                if fail >= 4: sigma = np.maximum(1e-8 * span, sigma * 0.9); fail = 0
            elif 5 == 3:
                t = evals / max(1, self.budget)
                sigma = np.maximum(1e-8 * span, sigma * (0.998 - 0.1 * t / max(1.0, self.dim)))
            elif 5 == 4:
                spread = float(np.std(Y)) + 1e-12
                sigma = np.clip(sigma * (1.0 + 0.02 * np.tanh(spread)), 1e-8 * span, 0.7 * span)
            else:
                sigma = np.clip(sigma * (1.01 if (it % 8 < 3) else 0.95), 1e-8 * span, 0.7 * span)

            if 3 == 0:
                w = int(np.argmax(Y))
                if yz < Y[w]: X[w], Y[w] = z, yz
            elif 3 == 1:
                w = int(np.argmax(Y)); X[w], Y[w] = z, yz
            elif 3 == 2:
                a, b = rng.integers(0, n, 2); t = a if Y[a] > Y[b] else b
                if yz < Y[t]: X[t], Y[t] = z, yz
            elif 3 == 3:
                k = it % n
                if yz <= Y[k] or rng.random() < 0.08: X[k], Y[k] = z, yz
            else:
                k = int(rng.integers(0, n)); X[k], Y[k] = z, yz

            if 4 == 1 and it % 7 == 0 and evals < self.budget:
                j = int(rng.integers(0, self.dim)); p = x_best.copy(); p[j] += 0.4 * sigma[j]
                yp = evaluate(p)
                if yp < y_best: x_best, y_best = clip(p), yp
            elif 4 == 2 and it % 9 == 0 and evals + 2 <= self.budget:
                d = rng.normal(0.0, 1.0, self.dim); d /= (np.linalg.norm(d) + 1e-12)
                p1 = clip(x_best + 0.35 * sigma * d); p2 = clip(x_best - 0.35 * sigma * d)
                y1 = evaluate(p1); y2 = evaluate(p2)
                if y1 < y_best or y2 < y_best:
                    x_best, y_best = (p1, y1) if y1 <= y2 else (p2, y2)
            elif 4 == 3 and it % 12 == 0 and evals + 2 <= self.budget:
                j = int(rng.integers(0, self.dim));
                p1 = x_best.copy(); p2 = x_best.copy();
                p1[j] += 0.3 * sigma[j]; p2[j] -= 0.3 * sigma[j]
                y1 = evaluate(p1); y2 = evaluate(p2)
                if y1 < y_best or y2 < y_best: x_best, y_best = (clip(p1), y1) if y1 <= y2 else (clip(p2), y2)
            elif 4 == 4 and it % 15 == 0 and evals + 1 <= self.budget:
                a = clip(0.7 * x_best + 0.3 * X[int(rng.integers(0, n))]); ya = evaluate(a)
                if ya < y_best: x_best, y_best = a, ya

            restart = False
            if 2 == 0: restart = fail >= 35
            elif 2 == 1: restart = fail >= 17 and float(np.std(Y)) < 1e-9
            elif 2 == 2: restart = (evals > self.budget // 4) and (it % max(10, 35//2) == 0) and fail > 3
            elif 2 == 3: restart = (fail >= 35) or ((it % max(9, 35//3) == 0) and succ == 0)
            else: restart = (it % max(12, 35//2) == 0) and (float(np.mean(Y)) > y_best)

            if restart and evals < self.budget:
                xr = clip(lb + rng.random(self.dim) * span); yr = evaluate(xr)
                fail = 0; succ = 0; sigma = np.full(self.dim, 0.051000000000000004) * span
                w = int(np.argmax(Y)); X[w], Y[w] = xr, yr
                if yr < y_best: x_best, y_best = xr.copy(), yr

        return x_best, float(y_best)
