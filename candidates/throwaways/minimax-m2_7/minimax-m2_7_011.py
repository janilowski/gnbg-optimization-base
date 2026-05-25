# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: The algorithm is a lightweight (μ+λ) Evolution Strategy with isotropic Gaussian mutation and a step‑size adaptation rule inspired by the 1/5‑th success rule. It maintains a small population of candidate solutions, generates offspring by perturbing randomly chosen parents with a zero‑mean Gaussian vector scaled by the current step size σ, selects the best individuals for the next generation, and adapts σ based on the proportion of offspring that improve over their parent. The method respects the evaluation budget and clips all proposals to the problem bounds.
# Search state: A population of μ candidates together with their objective values; the best observed solution is tracked globally.
# Candidate generation: Offspring are produced by adding a Gaussian mutation (σ·N(0,1)) to a randomly selected parent, then clipping the result to the admissible range.
# Selection and replacement: After evaluating all offspring, the best μ individuals among the combined parent‑offspring set are kept for the next generation (μ+λ strategy).
# Adaptation: The mutation step size σ is adjusted each generation using the 1/5‑th rule: if the success rate (offspring better than their parent) exceeds 0.2, σ is multiplied by 1.1; otherwise it is multiplied by 0.9. σ is kept within a safe interval relative to the problem’s bound ranges.
# Exploration mechanisms: Gaussian mutation provides continuous exploration; the population size and σ adaptation balance exploration vs. exploitation.
# Exploitation mechanisms: Keeping only the best μ individuals pushes the population toward promising regions.
# Boundary handling: All candidate points are clipped to the supplied lower/upper bounds to stay in the feasible space.
# Budget strategy: The algorithm stops as soon as the number of performed evaluations reaches the given budget; it uses all remaining evaluations for the last batch of offspring.
# Closest known influences: Classic (μ+λ) Evolution Strategies with the 1/5‑th rule; CMA‑ES‑inspired isotropic mutation without covariance adaptation.
# Novelty or unusual aspects: Using random parent selection for each offspring adds diversity while keeping the population and computational overhead minimal.
# Failure modes: For highly multi‑modal or deceptive functions, a very small population may converge prematurely; the fixed σ adaptation may be slow in very high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple (μ+λ) Evolution Strategy for black‑box minimization.
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        """
        Run the optimizer and return the best found solution.

        Parameters
        ----------
        func : callable
            A black‑box objective function that accepts a 1‑D numpy array
            and returns a scalar value.

        Returns
        -------
        best_x : numpy.ndarray
            The solution point that achieved the smallest objective value.
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------------
        # Determine problem bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'lower'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # If bounds are not provided, treat the space as unbounded.
            lower = np.full(self.dim, -np.inf)
            upper = np.full(self.dim, np.inf)

        # Ensure lower/upper are of the correct shape.
        if lower.shape != (self.dim,):
            lower = np.broadcast_to(lower, self.dim).copy()
        if upper.shape != (self.dim,):
            upper = np.broadcast_to(upper, self.dim).copy()

        # ------------------------------------------------------------------
        # Initialise evolution strategy parameters
        # ------------------------------------------------------------------
        # Population sizes – kept small for minimal overhead.
        mu = min(4, self.budget // 2)
        if mu < 1:
            mu = 1
        # Ensure we do not request more evaluations than available.
        if mu > self.budget:
            mu = self.budget

        lam = max(mu * 2, 10)   # offspring per generation

        # Initial step size (isotropic).  Using roughly 1/6 of the range.
        sigma = (upper - lower) / 6.0
        sigma = np.maximum(sigma, 1e-12)   # avoid degenerate step size

        # ------------------------------------------------------------------
        # Initial random population
        # ------------------------------------------------------------------
        pop_x = lower + (upper - lower) * np.random.rand(mu, self.dim)
        pop_f = np.empty(mu)
        for i in range(mu):
            pop_f[i] = func(pop_x[i])

        evals = mu
        # Track global best
        best_idx = np.argmin(pop_f)
        best_x = pop_x[best_idx].copy()
        best_y = pop_f[best_idx]

        # ------------------------------------------------------------------
        # Adaptation bookkeeping
        # ------------------------------------------------------------------
        success_count = 0
        total_count = 0

        # ------------------------------------------------------------------
        # Main evolution loop
        # ------------------------------------------------------------------
        while evals < self.budget:
            # Determine how many offspring we can afford now.
            remaining = self.budget - evals
            cur_lam = min(lam, remaining)

            # Produce offspring by mutating a randomly chosen parent.
            offs_x = np.empty((cur_lam, self.dim))
            parent_idx = np.random.randint(0, mu, size=cur_lam)

            for i in range(cur_lam):
                # Gaussian mutation clipped to bounds
                offs_x[i] = np.clip(pop_x[parent_idx[i]] + sigma * np.random.randn(self.dim),
                                    lower, upper)

            # Evaluate offspring
            offs_f = np.empty(cur_lam)
            for i in range(cur_lam):
                offs_f[i] = func(offs_x[i])
                evals += 1
                # Update global best if needed
                if offs_f[i] < best_y:
                    best_y = offs_f[i]
                    best_x = offs_x[i].copy()

            # Count successful mutations (offspring beats its parent)
            for i in range(cur_lam):
                if offs_f[i] < pop_f[parent_idx[i]]:
                    success_count += 1
            total_count += cur_lam

            # ------------------------------------------------------------------
            # Step‑size adaptation (1/5‑th rule)
            # ------------------------------------------------------------------
            if total_count > 0:
                success_rate = success_count / total_count
                if success_rate > 0.2:
                    sigma *= 1.1
                else:
                    sigma *= 0.9

                # Keep sigma within a reasonable range relative to the domain size.
                sigma = np.clip(sigma, (upper - lower) * 1e-6, (upper - lower) * 0.5)

            # ------------------------------------------------------------------
            # Selection: keep the best μ individuals out of parents + offspring
            # ------------------------------------------------------------------
            combined_x = np.vstack([pop_x, offs_x])
            combined_f = np.concatenate([pop_f, offs_f])
            # Sort by fitness and retain the top μ
            best_local = np.argsort(combined_f)[:mu]
            pop_x = combined_x[best_local]
            pop_f = combined_f[best_local]

            # Reset adaptation counters for the next generation
            success_count = 0
            total_count = 0

        # Return the best solution found
        return best_x, float(best_y)
