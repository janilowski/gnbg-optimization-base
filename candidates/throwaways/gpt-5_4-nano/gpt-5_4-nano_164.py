import numpy as np


# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, robust black-box minimization algorithm inspired by
# Evolution Strategies (ES) with a CMA-lite adaptation of step size. It maintains a
# population of candidate solutions, samples around a current centroid, evaluates the
# objective in a strict evaluation-budget-aware loop, and updates both centroid and
# mutation scale using rank-based selection.
# Search state: Tracks current best solution (x_best, y_best), a centroid (mean),
# mutation step-size (sigma), and a remaining evaluation budget counter.
# Candidate generation: Each iteration samples lambda offspring as mean + sigma * N(0, I).
# Optional small directional perturbations around the best are mixed into offspring to
# encourage progress when improvement is found.
# Selection and replacement: Offspring are evaluated, sorted by objective value (minimization),
# and the best mu individuals update the centroid using recombination weights.
# Adaptation: Step size sigma is adapted using a success-rate heuristic and a mild
# progress check relative to the best observed improvements.
# Exploration mechanisms: Random Gaussian sampling and occasional best-directed perturbations.
# Exploitation mechanisms: Mean recombination from top-ranked individuals and decreasing
# sigma when improvements are common; best-direction mixing when stagnating.
# Boundary handling: Candidates are projected (clipped) into provided bounds each time before
# evaluation to ensure feasibility.
# Budget strategy: Uses only func evaluations counted explicitly; stops immediately when
# the budget would be exceeded. The total number of evaluations equals the provided budget.
# Closest known influences: Rank-based ES (plus evolution-path-like success adaptation, but simplified).
# Novelty or unusual aspects: Includes a small mixture with best-centered perturbations and
# uses a combined success/progress signal to scale sigma without CMA matrix learning.
# Failure modes: If the function is highly discontinuous or bounds are extremely tight,
# projection can reduce effective search diversity; sigma may shrink too aggressively on
# noisy objectives, but budget-limited exploration via best-mix mitigates complete stagnation.
# ALGORITHM_ANALYSIS_NOTE_END
class Algorithm:
    def __init__(self, budget, dim):
        if budget is None or budget <= 0:
            raise ValueError("budget must be a positive integer")
        if dim is None or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # Bounds extraction
        lower, upper = self._get_bounds(func)
        d = self.dim
        lo = np.asarray(lower, dtype=float).reshape(-1)
        hi = np.asarray(upper, dtype=float).reshape(-1)
        if lo.shape[0] != d or hi.shape[0] != d:
            raise ValueError("Bounds must match the provided dim")
        # Ensure lo <= hi
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

        def clip(x):
            return np.minimum(hi, np.maximum(lo, x))

        evals = 0
        remaining = self.budget

        # Initialization: random mean within bounds
        mean = lo + (hi - lo) * np.random.rand(d)
        mean = clip(mean)

        # Evaluate initial point (counts toward budget)
        y_best = float(func(mean))
        evals += 1
        remaining -= 1
        x_best = mean.copy()

        # Choose ES parameters based on dimension and budget
        # Typical ES uses lambda ~ 4 + 3*log(d), but clamp to budget.
        lam = int(max(4, 4 + 3 * np.log(max(2, d))))
        lam = min(lam, self.budget)  # not necessarily usable each iter
        # mu as a fraction of lambda (top performers)
        mu = max(2, lam // 2)

        # Recombination weights: log-based, normalized
        ranks = np.arange(mu)
        weights = np.log(mu + 0.5) - np.log(ranks + 1.0)
        weights = weights / np.sum(weights)

        # Initial sigma: fraction of the box size
        box = np.linalg.norm(hi - lo) / np.sqrt(d)
        sigma = 0.3 * box if box > 0 else 0.1

        # Success/progress adaptation knobs
        # Keep conservative to be robust across dims.
        c_inc = 1.20
        c_dec = 1 / 1.20
        sigma_min = 1e-12 * (box if box > 0 else 1.0)
        sigma_max = 1e2 * (box if box > 0 else 1.0)

        best_mix_prob = 0.35  # mix in best-centered perturbations when stagnating
        stagnation_counter = 0
        best_improvement = y_best

        # Main loop: use as many full iterations as budget allows.
        # Each iteration evaluates lambda offspring.
        # For final leftover evaluations, evaluate fewer offspring to match budget exactly.
        while remaining > 0:
            # Decide iteration size to not exceed budget.
            cur_lam = min(lam, remaining)
            if cur_lam < 1:
                break

            # Offspring sampling
            # Mix: some offspring are mean + sigma*N and some are x_best + sigma*N
            # This lightly biases search toward promising areas without eliminating exploration.
            if stagnation_counter <= 2:
                p_best = best_mix_prob * 0.5
            else:
                p_best = best_mix_prob

            offspring = np.empty((cur_lam, d), dtype=float)
            for i in range(cur_lam):
                if np.random.rand() < p_best:
                    base = x_best
                else:
                    base = mean
                offspring[i] = base + sigma * np.random.randn(d)
            offspring = clip(offspring)

            # Evaluate offspring
            ys = np.empty(cur_lam, dtype=float)
            for i in range(cur_lam):
                ys[i] = float(func(offspring[i]))
            evals += cur_lam
            remaining -= cur_lam

            # Track best
            idx_best_local = int(np.argmin(ys))
            if ys[idx_best_local] < y_best:
                y_best = float(ys[idx_best_local])
                x_best = offspring[idx_best_local].copy()

            # Sort by fitness (minimization)
            order = np.argsort(ys)
            sorted_X = offspring[order]
            sorted_y = ys[order]

            # Recombination update for centroid using top mu individuals
            cur_mu = min(mu, cur_lam)
            top_X = sorted_X[:cur_mu]

            # Weighted recombination: sum w_i * x_i
            # Use first cur_mu weights re-normalized
            w = weights[:cur_mu]
            w = w / np.sum(w)
            new_mean = np.sum(top_X * w[:, None], axis=0)

            # Step-size adaptation (success-based)
            # Define "success" if offspring ranks improve on current best by relative margin.
            # Also incorporate progress trend.
            median_y = float(np.median(ys))
            improvement = best_improvement - y_best
            best_improvement = y_best

            # Use relative improvement signal: if best got better often, increase sigma; else decrease.
            # "Success rate": fraction of offspring better than current y_best (strict).
            success_rate = float(np.mean(sorted_y < y_best + 0.0))  # equals 0 or more if ties
            # Alternative progress measure: compare best of offspring to previous best.
            # Since y_best may just have updated, approximate using sorted_y[0] vs y_best.
            best_offspring_y = float(sorted_y[0])
            # If best_offspring_y is close to y_best (including new), still allow gradual adaptation.
            close_factor = 0.0
            if cur_mu > 0:
                # closeness: smaller median relative to best indicates more "bunched improvement"
                if median_y != best_offspring_y:
                    close_factor = (median_y - best_offspring_y) / (abs(median_y) + 1e-12)

            # Heuristic: combine success_rate and close_factor and improvement magnitude.
            # improvement sign is always nonnegative here; small improvements may still matter.
            if success_rate > 0 or improvement > 0 or close_factor > 0.01:
                sigma = min(sigma_max, sigma * c_inc)
                stagnation_counter = max(0, stagnation_counter - 1)
                mean = new_mean
            else:
                sigma = max(sigma_min, sigma * c_dec)
                stagnation_counter += 1
                mean = new_mean

            # Occasional restart-ish nudge when fully stagnating:
            # If centroid doesn't improve for a while, widen search modestly.
            if stagnation_counter >= 6 and remaining > 0:
                # Nudge mean toward the best with some noise, and bump sigma slightly.
                nudge = 0.5 * (x_best - mean)
                mean = clip(mean + nudge + 0.1 * sigma * np.random.randn(d))
                sigma = min(sigma_max, sigma * 1.1)
                stagnation_counter = 0

            # Also keep best feasible projected (already projected for candidates)
            x_best = clip(x_best)

            # Budget safety: loop condition handles remaining.
            # (Do not evaluate further than budget.)

        # If budget was 0 (shouldn't happen) return initial
        return x_best, y_best

    @staticmethod
    def _get_bounds(func):
        # Accept either func.lower/func.upper or func.bounds.lb/func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return func.lower, func.upper
        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return b.lb, b.ub
        raise AttributeError(
            "Objective function must provide bounds via either "
            "func.lower/func.upper or func.bounds.lb/func.bounds.ub"
        )
