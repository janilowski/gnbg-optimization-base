# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a simple (mu+lambda) Evolution Strategy (ES) for
#          black‑box minimization.  The algorithm maintains a population of candidate
#          solutions (parents), creates offspring by adding Gaussian noise scaled by a
#          step‑size vector sigma, and uses a 1/5‑th rule to adapt sigma each generation.
#          Selection is elitist: after evaluating both parents and offspring the best
#          mu individuals survive.  The algorithm respects the evaluation budget and
#          never performs more than the allowed function evaluations.
#
# Search state:
#   - Population matrix pop of shape (mu, dim) holding current parents.
#   - Vector sigma of shape (dim,) controlling the mutation strength per dimension.
#   - Counters for total evaluations performed and the number of successful offspring
#     in the current generation (used for sigma adaptation).
#
# Candidate generation:
#   - Lambda offspring are produced by selecting a parent uniformly at random and
#     adding independent Gaussian perturbations: x_child = parent + sigma * N(0,I).
#   - Offspring are clipped to the problem’s bounds to stay feasible.
#
# Selection and replacement:
#   - Parents and offspring are evaluated and then concatenated.
#   - The best mu individuals (according to objective value) become the next parent
#     population (elitist (mu+lambda) strategy).
#
# Adaptation:
#   - After each generation we compute the success rate = successful_offspring / lambda.
#   - If the rate exceeds 0.2, sigma is increased by factor 1.1; otherwise it is
#     decreased by factor 0.9 (the classic 1/5‑th rule).  Sigma is also clipped to
#     avoid extreme values relative to the domain size.
#
# Exploration mechanisms:
#   - Large sigma encourages broad exploration across the search space.
#   - Gaussian mutations inject randomness in all dimensions.
#
# Exploitation mechanisms:
#   - Small sigma focuses search around the current best solutions.
#   - Elitism retains high‑quality solutions, guiding the search toward promising
#     regions.
#
# Boundary handling:
#   - Offspring values are clipped to the lower/upper bounds of the problem
#     (accessed via func.lower/func.upper or func.bounds.lb/ub).
#
# Budget strategy:
#   - The algorithm tracks the number of performed evaluations and checks after each
#     evaluation whether the budget is exhausted.  If the budget is reached, the
#     algorithm terminates immediately and returns the best solution found so far.
#
# Closest known influences:
#   - Classic Evolution Strategies with (mu+lambda) selection (Bäck et al., 1993).
#   - The 1/5‑th rule for step‑size adaptation, similar to the adaptation mechanism
#     used in CMA‑ES but without covariance matrix updates.
#
# Novelty or unusual aspects:
#   - Population size and offspring number are chosen as modest functions of dimension
#     (mu = max(4, dim), lambda = max(10, 4*mu)) to keep the algorithm compact and
#     robust across a range of problem dimensions.
#   - No covariance matrix is maintained, which reduces overhead and makes the
#     implementation simple while still offering competitive performance on many
#     smooth landscapes.
#
# Failure modes:
#   - If the evaluation budget is smaller than the initial population size, the
#     algorithm may not converge to a good solution.
#   - The simple sigma adaptation can cause premature shrinkage on very flat or
#     rugged landscapes, leading to early stagnation.
#   - In very high dimensions the lack of covariance adaptation may limit the
#     algorithm’s ability to exploit coordinate correlations, reducing effectiveness.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple (mu+lambda) Evolution Strategy for black‑box minimization.

    Public interface:
        __init__(self, budget, dim)
        __call__(self, func) -> (best_x, best_y)
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
        self.budget = budget
        self.dim = dim
        # Population sizes that scale modestly with dimension
        self.mu = max(4, dim)
        self.lambda_ = max(10, 4 * self.mu)

    def __call__(self, func):
        """
        Run the evolution strategy on the given objective function.

        Parameters
        ----------
        func : callable
            Black‑box objective to be minimized.  Must expose either
            ``func.lower`` / ``func.upper`` or ``func.bounds.lb`` / ``func.bounds.ub``
            providing the domain bounds.

        Returns
        -------
        best_x : ndarray
            Best solution found (vector of length dim).
        best_y : float
            Corresponding objective value.
        """
        # ---------- Determine problem bounds ----------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: use a wide default range if bounds are not provided
            lower = np.full(self.dim, -5.0)
            upper = np.full(self.dim, 5.0)

        if lower.shape != (self.dim,) or upper.shape != (self.dim,):
            raise ValueError("Bounds dimensions do not match the search dimension.")
        if np.any(lower >= upper):
            raise ValueError("Lower bounds must be strictly smaller than upper bounds.")

        # ---------- Initial step size (sigma) ----------
        # Initial sigma is set to 20% of the bound range; it will be adapted later.
        sigma = (upper - lower) * 0.2

        # ---------- Initial population (parents) ----------
        pop = np.random.uniform(lower, upper, size=(self.mu, self.dim))

        evals = 0
        fitness = np.empty(self.mu, dtype=float)

        # Evaluate initial parents
        for i in range(self.mu):
            fitness[i] = func(pop[i])
            evals += 1
            if evals >= self.budget:
                # Not enough budget to evaluate whole population – return best seen
                best_idx = np.argmin(fitness[:evals])
                return pop[best_idx].copy(), fitness[best_idx]

        # Track global best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # ---------- Main evolution loop ----------
        success_count = 0  # number of offspring improving over their parent

        while evals < self.budget:
            # Determine how many offspring we can afford to evaluate this round
            remaining = self.budget - evals
            offs_size = min(self.lambda_, remaining)

            # ----- Generate offspring -----
            # Choose parents uniformly at random
            parent_indices = np.random.randint(0, self.mu, size=offs_size)
            # Gaussian mutation: x + sigma * N(0, I)
            noise = np.random.normal(0.0, 1.0, size=(offs_size, self.dim))
            offspring = pop[parent_indices] + noise * sigma  # broadcasting sigma across rows
            # Keep solutions inside the feasible region
            np.clip(offspring, lower, upper, out=offspring)

            # ----- Evaluate offspring -----
            offs_fitness = np.empty(offs_size, dtype=float)
            for i in range(offs_size):
                offs_fitness[i] = func(offspring[i])
                evals += 1
                # Track success (offspring better than its parent)
                if offs_fitness[i] < fitness[parent_indices[i]]:
                    success_count += 1
                # If budget exhausted, combine evaluated set and pick best
                if evals >= self.budget:
                    # Include all parents and the offspring evaluated so far
                    all_x = np.vstack([pop, offspring[:i + 1]])
                    all_f = np.concatenate([fitness, offs_fitness[:i + 1]])
                    best_local = np.argmin(all_f)
                    return all_x[best_local].copy(), all_f[best_local]

            # ----- (mu+lambda) selection: keep best mu individuals -----
            combined_x = np.vstack([pop, offspring])
            combined_f = np.concatenate([fitness, offs_fitness])
            # Sort by fitness and keep top mu
            best_mu_idx = np.argsort(combined_f)[:self.mu]
            pop = combined_x[best_mu_idx]
            fitness = combined_f[best_mu_idx]

            # ----- Adapt sigma (1/5‑th rule) -----
            success_rate = success_count / offs_size
            if success_rate > 0.2:
                sigma *= 1.1
            else:
                sigma *= 0.9
            # Prevent sigma from becoming too small or too large relative to bounds
            sigma = np.clip(sigma, (upper - lower) * 1e-4, (upper - lower) * 0.5)

            # Reset success counter for next generation
            success_count = 0

            # Update global best if any new individual is better
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_y:
                best_y = fitness[current_best_idx]
                best_x = pop[current_best_idx].copy()

        return best_x, best_y
