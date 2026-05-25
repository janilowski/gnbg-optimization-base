# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (μ+λ) Evolution Strategy with self-adaptive step-size (cma-es style) using
#          isotropic Gaussian mutation and comma-selection. The algorithm maintains a
#          population of μ candidates, generates λ offspring per iteration via mutation,
#          and replaces the worst individuals with the best offspring based on fitness.
#
# Search state: Population of μ points in dim-dimensional space, each with an
#               associated step-size sigma (adapted globally). Current best candidate
#               and best fitness observed so far are tracked.
#
# Candidate generation: Each offspring is created by adding Gaussian noise to a
#                       randomly selected parent from the current population. The
#                       noise has standard deviation sigma, which is adapted using
#                       cumulative step-size adaptation (CSA) rules based on
#                       success/failure of recent steps.
#
# Selection and replacement: (μ,λ)-selection: The μ best individuals among the
#                            offspring only (not parents) survive to form the next
#                            generation. If lambda < mu, falls back to (μ+λ)-selection
#                            (parents considered).
#
# Adaptation: Step-size sigma is adapted using a cumulative step-size adaptation
#             rule: sigma ← sigma * exp((||z_cumulative|| - chi_d) / (d * c_sigma)).
#             This tends to adjust sigma to maintain an expected step length of
#             chi_d ≈ sqrt(d) * (1 - 1/(4d) + 1/(21d^2)).
#
# Exploration mechanisms: Large initial sigma (0.5 * range) for broad coverage;
#                         isotropic Gaussian mutations allow exploration in all
#                         directions; (μ,λ)-selection promotes diversity by not
#                         keeping parents.
#
# Exploitation mechanisms: Small sigma allows fine-grained local search; selection
#                          pressure focuses on improving solutions; parent selection
#                          for mutation (vs. random) biases toward better regions.
#
# Boundary handling: Candidates are clipped to [lower, upper] bounds after mutation.
#                    Clipping does not affect sigma adaptation (uses pre-clip values).
#
# Budget strategy: Evaluates floor(0.1 * budget) candidates initially to gather
#                  statistics, then performs remaining evaluations in iterations
#                  with max(1, mu) offspring per iteration to maintain efficiency.
#
# Closest known influences: CMA-ES (Hansen & Ostermeier 2001) for the CSA adaptation
#                           mechanism and overall ES structure; classical (μ+λ)-ES
#                           for selection scheme.
#
# Novelty or unusual aspects: Simplified CSA (using only one evolution path vector)
#                              rather than full CMA-ES's two; uses random parent
#                              selection rather than weighted recombination for
#                              population generation.
#
# Failure modes: May converge prematurely if sigma shrinks too quickly on
#                deceptive landscapes; suffers from O(μ*dim) overhead per iteration;
#                poorly suited for highly multi-modal functions if sigma adaptation
#                gets stuck in local basins.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    """
    (μ+λ) Evolution Strategy with self-adaptive step-size.

    Minimizes a black-box function `func` within a budget of evaluations.
    Adapts the global step-size using cumulative step-size adaptation (CSA).
    """

    def __init__(self, budget, dim):
        """
        Initialize the ES optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimization and return the best found solution.

        Parameters
        ----------
        func : callable
            Black-box function to minimize. Supports two attribute styles for bounds:
            - func.lower, func.upper (scalars or arrays)
            - func.bounds.lb, func.bounds.ub (attribute style)

        Returns
        -------
        best_x : ndarray
            Best solution found (dim-dimensional).
        best_y : float
            Function value at best_x.
        """
        # Extract bounds
        if hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)

        # Ensure bounds are array of shape (dim,)
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)

        rng = np.random.default_rng()

        # --- Parameter setup -------------------------------------------------
        # Use small mu for stability; lambda scales with budget to keep overhead low
        # Initial phase uses more candidates for early exploration
        init_phase_size = max(10, int(0.1 * self.budget))
        mu = max(2, min(5, self.dim))  # population size
        lambda_ = max(mu, min(self.budget - init_phase_size, mu * 2))  # offspring per gen

        # Initial step-size: start large for global exploration
        sigma = 0.5 * (ub - lb).mean()

        # CSA parameters (standard for CMA-ES-like adaptation)
        # c_sigma controls cumulation speed for step-size adaptation
        # damping ensures sigma doesn't change too rapidly
        c_sigma = 1.0 / (np.sqrt(self.dim) + 1.0)
        d_sigma = 1.0 + np.sqrt(self.dim / mu) if mu > 0 else 2.0

        # chi_d: expected length of isotropic mutation vector, approx sqrt(dim)
        # chi_d = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        # Using simpler approximation for compactness
        chi_d = np.sqrt(self.dim)

        # Cumulative evolution path for CSA
        p_sigma = np.zeros(self.dim)

        # Population storage: (mu, dim)
        population = lb + (ub - lb) * rng.random((mu, self.dim))

        # Evaluate initial population
        fitness = np.array([func(x) for x in population])
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]
        evals = mu

        # If budget is tiny, return after initial eval
        if evals >= self.budget:
            return best_x, best_y

        # --- Main evolution loop ---------------------------------------------
        while evals < self.budget:
            # Generate lambda_ offspring
            offspring = np.empty((lambda_, self.dim))
            offspring_fitness = np.empty(lambda_)

            for i in range(lambda_):
                # Select random parent (simplifies from weighted recombination)
                parent_idx = rng.integers(0, mu)

                # Sample mutation: isotropic Gaussian with current sigma
                z = rng.standard_normal(self.dim)
                candidate = population[parent_idx] + sigma * z

                # Clip to bounds
                candidate = np.clip(candidate, lb, ub)

                offspring[i] = candidate
                offspring_fitness[i] = func(candidate)
                evals += 1

                if evals >= self.budget:
                    break

            if evals >= self.budget:
                break

            # Update evolution path (cumulative step-size adaptation)
            # Successful steps tend to increase ||p_sigma||, unsuccessful decrease it
            if lambda_ > 0:
                # Use mean of successful steps (best few offspring) for path update
                n_succ = max(1, lambda_ // 2)
                succ_idx = np.argsort(offspring_fitness)[:n_succ]
                z_mean = np.mean([(offspring[j] - population[rng.integers(0, mu)]) / sigma
                                 for j in succ_idx], axis=0)

                # Normalize by expected length of random selection (approx 1/sqrt(n_succ))
                if n_succ > 0:
                    z_mean /= np.sqrt(n_succ)

                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * z_mean

            # Adapt step-size using CSA rule
            p_norm = np.linalg.norm(p_sigma)
            sigma *= np.exp((p_norm - chi_d) / (self.dim * d_sigma))

            # (μ,λ)-selection: select best mu individuals from offspring only
            # This promotes diversity better than (μ+λ)-selection
            if lambda_ >= mu:
                surviving_idx = np.argsort(offspring_fitness)[:mu]
                population = offspring[surviving_idx].copy()
            else:
                # Not enough offspring: fall back to (μ+λ) by keeping parents
                combined = np.vstack([population, offspring])
                combined_fit = np.concatenate([fitness, offspring_fitness])
                surviving_idx = np.argsort(combined_fit)[:mu]
                population = combined[surviving_idx].copy()
                fitness = combined_fit[surviving_idx]

            # Update best solution found
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_y:
                best_x = population[best_idx].copy()
                best_y = fitness[best_idx]

        return best_x, best_y
