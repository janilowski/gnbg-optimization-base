import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm for
# bounded continuous domains using a population-based evolutionary strategy with
# self-adaptive Gaussian mutations and occasional differential-evolution style
# steps. It is designed to be robust across dimensions and to never exceed the
# evaluation budget.
# Search state: The algorithm maintains a small population of candidate solutions
# (x vectors) and their objective values (y). It also tracks a step-size sigma
# used for Gaussian mutations.
# Candidate generation: Each iteration produces offspring by either (a) sampling
# mutations around selected parents using log-normal updates to sigma, or
# (b) performing a differential-evolution-like perturbation based on population
# differences. Offspring are clipped to bounds.
# Selection and replacement: A (μ+λ) style update is used: offspring are appended
# to the population, and the best μ individuals survive based on objective values.
# Adaptation: Sigma (mutation scale) is adapted via self-adaptation on successful
# steps and a light global shrink/expand based on whether improvements are found.
# Exploration mechanisms: Differential-evolution steps and occasional high-variance
# mutations encourage exploration, especially in early iterations or when progress
# stalls.
# Exploitation mechanisms: Gaussian mutations centered at good individuals and
# greedy survival bias the search toward lower objective values.
# Boundary handling: Candidates are always kept within provided bounds by clipping.
# Budget strategy: The number of iterations is computed from the budget and the
# population size; total objective evaluations are counted explicitly and the last
# iteration truncates offspring generation so the budget is never exceeded.
# Closest known influences: The design borrows from evolution strategies (self-adaptive
# sigma), differential evolution (difference-based proposals), and (μ+λ) survivor
# selection.
# Novelty or unusual aspects: The algorithm mixes two proposal types with a budget-
# aware truncation mechanism, while using a simple yet effective step-size schedule
# that keeps behavior stable across dimensions.
# Failure modes: If the budget is extremely small, the population may not have time
# to improve significantly beyond random sampling. For very narrow or ill-scaled
# bounds, clipping may cause many duplicates, reducing effective exploration.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        # Population sizing: small for low budgets, larger for higher budgets.
        # Keep it stable across dimensions.
        if self.budget <= 0 or self.dim <= 0:
            raise ValueError("budget and dim must be positive integers.")
        self.mu = max(4, min(12, self.budget // (2 * (self.dim + 1)) + 6))
        self.lam = max(2, self.mu * 2)  # offspring per generation

        # Clamp population sizes to avoid overshooting with small budgets.
        self.mu = min(self.mu, max(4, self.budget))  # at most budget individuals
        self.lam = min(self.lam, max(2, self.budget - self.mu))

        # Global sigma multiplier; adapted per generation.
        self.sigma0 = 0.3

    def __call__(self, func):
        dim = self.dim

        # Read bounds from func.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lo = np.asarray(func.lower, dtype=float)
            hi = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lo = np.asarray(func.bounds.lb, dtype=float)
            hi = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub.")

        if lo.shape == ():  # allow scalar bounds
            lo = np.full(dim, lo, dtype=float)
        if hi.shape == ():
            hi = np.full(dim, hi, dtype=float)

        lo = lo.reshape(-1)
        hi = hi.reshape(-1)
        if lo.size != dim or hi.size != dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Ensure proper ordering.
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)

        # If any bounds are degenerate, they effectively fix coordinates.
        span = hi2 - lo2
        span = np.where(span > 0, span, 1.0)  # avoid division by zero for sigma scaling

        def clip(x):
            return np.minimum(np.maximum(x, lo2), hi2)

        # Budget accounting.
        max_evals = self.budget
        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= max_evals:
                # If caller asks for too many evaluations, fail safe by returning inf.
                return float("inf")
            y = float(func(x))
            evals += 1
            return y

        # Determine initial sigma based on bounds span.
        # Use a modest fraction so that clipping does not dominate.
        sigma = self.sigma0 * span

        # Initialize population uniformly within bounds.
        # Evaluate μ initial points.
        pop = np.empty((self.mu, dim), dtype=float)
        for i in range(self.mu):
            r = np.random.rand(dim)
            pop[i] = lo2 + r * (hi2 - lo2)
        vals = np.array([evaluate(pop[i]) for i in range(self.mu)], dtype=float)

        # Track best.
        best_idx = int(np.argmin(vals))
        best_x = pop[best_idx].copy()
        best_y = float(vals[best_idx])

        # Generation control: estimate remaining generations by budget.
        # Each generation uses up to λ evaluations; we truncate to respect budget.
        # Minimum of 1 generation if possible.
        while evals < max_evals:
            remaining = max_evals - evals
            if remaining <= 0:
                break

            # Number of offspring to evaluate in this generation (budget aware).
            n_off = min(self.lam, remaining)

            # Parent selection: tournament among μ.
            # Tournament size grows slightly with dim to keep selection pressure.
            tsize = max(2, min(4, 2 + dim // 10))

            def tournament_select(k=tsize):
                idxs = np.random.randint(0, self.mu, size=k)
                winner = idxs[np.argmin(vals[idxs])]
                return int(winner)

            # Create offspring.
            off = np.empty((n_off, dim), dtype=float)
            off_vals = np.empty(n_off, dtype=float)

            # Success measure: for adaptation, track whether any offspring improves best.
            any_improved = False

            # Mix rate between differential-style and gaussian mutation.
            # Increase exploration early (when best_y is not yet particularly good),
            # but keep it simple: use a decreasing probability for DE as progress continues.
            # We'll approximate progress via sigma magnitude and generation count through remaining.
            # (No explicit generation count needed.)
            explore_prob = 0.55 if evals < max_evals * 0.5 else 0.25

            for j in range(n_off):
                if np.random.rand() < explore_prob and self.mu >= 3:
                    # Differential-evolution style proposal:
                    # x = best + F*(x_r1 - x_r2) + noise
                    r1, r2, r3 = np.random.randint(0, self.mu, size=3)
                    # Ensure difference vectors not degenerate when possible.
                    # (If equal, the difference becomes 0, which is fine.)
                    base = pop[np.argmin(vals)]
                    F = 0.5 + 0.4 * np.random.rand()
                    diff = pop[r1] - pop[r2]
                    noise = np.random.randn(dim) * 0.05 * span
                    x = base + F * diff + noise
                    # Self-adaptive scaling: also apply a small gaussian around x.
                    x = x + np.random.randn(dim) * (0.2 * sigma)
                else:
                    # Gaussian mutation around a tournament-selected parent,
                    # with self-adaptive step-size update (per-offspring global sigma).
                    p = tournament_select()
                    parent = pop[p]

                    # Log-normal multiplicative update inspired by ES.
                    # The per-offspring update makes behavior more resilient.
                    tau = 1.0 / np.sqrt(2.0 * np.sqrt(dim) + 1e-12)
                    # Use shared local multiplier to keep it compact.
                    global_norm = np.random.randn()
                    sigma_j = sigma * np.exp(tau * global_norm)

                    # Coordinate-wise jitter with slight correlation.
                    z = np.random.randn(dim)
                    # Bias: use smaller mutation as sigma shrinks via adaptation.
                    x = parent + z * sigma_j

                x = clip(x)
                y = evaluate(x)
                off[j] = x
                off_vals[j] = y

                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                    any_improved = True

            # (μ+λ) selection.
            # Combine and select μ best.
            comb = np.vstack([pop, off])
            comb_vals = np.concatenate([vals, off_vals])
            idx = np.argsort(comb_vals)[: self.mu]
            pop = comb[idx]
            vals = comb_vals[idx]

            # Adaptation: shrink sigma when improving, expand slightly when not.
            # Keep sigma bounded to avoid numerical issues.
            if any_improved:
                sigma = sigma * 0.90
            else:
                sigma = sigma * 1.06

            # Soft bounds on sigma based on span.
            sigma_min = 1e-12 + 1e-6 * span
            sigma_max = 2.5 * span
            sigma = np.minimum(np.maximum(sigma, sigma_min), sigma_max)

            # Safety: if we've filled budget exactly, exit.
            if evals >= max_evals:
                break

        return best_x, best_y
