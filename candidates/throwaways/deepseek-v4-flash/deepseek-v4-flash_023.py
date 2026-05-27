import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (mu, lambda)-Evolution Strategy (ES) with self-adaptive per-dimension
#          step sizes for continuous black-box minimization. It uses intermediate
#          recombination for both object variables and step sizes, and log-normal
#          mutation to adapt step sizes. The algorithm is population-based,
#          compact, and intended to work robustly across dimensions.
#
# Search state: A set of 'mu' parents, each with a vector x (current candidate
#               solution) and a vector sigma (step size per dimension). The best
#               solution ever seen is stored separately.
#
# Candidate generation: Each generation produces 'lambda' offspring. For each
#                       offspring, two distinct parents are randomly chosen and
#                       their x vectors are averaged (intermediate recombination).
#                       Their sigma vectors are averaged geometrically (log domain)
#                       to produce the offspring’s initial sigma. Then sigma is
#                       mutated by multiplying each component by
#                       exp(tau * global_normal + tau' * per_dim_normal).
#                       The object variable x is then mutated by adding
#                       sigma * standard_normal per dimension.
#
# Selection and replacement: (mu,lambda) selection: after all lambda offspring
#                            are evaluated, the best mu offspring (by fitness)
#                            become the new parent set for the next generation.
#                            The best ever solution is updated whenever a better
#                            one is found.
#
# Adaptation: Step sizes are adapted via the log-normal mutation parameters
#             tau and tau', which are set to standard ES values:
#             tau = 1 / sqrt(2 * dim), tau' = 1 / sqrt(2 * sqrt(dim)).
#             This allows global and per-dimension adaptation rates.
#
# Exploration mechanisms: Large sigma values early (initial sigma ~ 0.2 * range)
#                         and stochastic offspring generation with Gaussian
#                         perturbations provide exploration. The comma selection
#                         ensures that worse parents are not kept.
#
# Exploitation mechanisms: Intermediate recombination blends good solutions.
#                          Step sizes shrink when the population is near the
#                          optimum due to the selection pressure and adaptation.
#
# Boundary handling: After mutation, each coordinate is clamped to the
#                    [lower, upper] bounds. No reflection or re‑sampling.
#
# Budget strategy: The total number of function evaluations is counted. The
#                  algorithm stops immediately when the counter reaches the
#                  budget. The budget is allocated efficiently: initial parents
#                  consume mu evaluations, then each generation consumes lambda
#                  evaluations.
#
# Closest known influences: Standard Evolution Strategy with self‑adaptation,
#                           similar to the (mu,lambda)-ES described in
#                           "Evolution Strategies" by Beyer & Schwefel.
#
# Novelty or unusual aspects: None. This is a straightforward implementation of
#                             a classical ES. The code is minimal and relies
#                             only on numpy.
#
# Failure modes: In very high dimensions, the heuristic adaptation constants
#                may not be optimal, but the algorithm should still find
#                reasonable solutions given enough budget. For very small
#                budgets (<mu), the algorithm will simply perform random search
#                with the initial population.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # ES parameters
        self.mu = 4                # number of parents
        self.lmbda = 8             # number of offspring
        # Standard self-adaptation learning rates
        self.tau = 1.0 / np.sqrt(2 * dim)
        self.tau_prime = 1.0 / np.sqrt(2 * np.sqrt(dim))

    def __call__(self, func):
        # Read bounds (both interface variants)
        try:
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lower = np.asarray(func.bounds.lb, dtype=float)
                upper = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise RuntimeError("Cannot read bounds from func")

        dim = self.dim
        if lower.shape != (dim,) or upper.shape != (dim,):
            raise RuntimeError("Bounds shape mismatch")

        # Initial step size: 20% of the range per dimension
        sigma0 = 0.2 * (upper - lower)

        # Seed the random state (harness set the seed, but we use numpy directly)
        rng = np.random.default_rng()

        # ---- Initialise parent population ----
        # Parents: arrays of shape (mu, dim)
        x_parents = rng.uniform(lower, upper, size=(self.mu, dim))
        sigma_parents = np.full((self.mu, dim), sigma0)   # all parents start with same sigma
        fitness_parents = np.full(self.mu, np.inf)

        evals = 0
        best_x = None
        best_y = np.inf

        # Evaluate initial parents
        for i in range(self.mu):
            y = func(x_parents[i])
            evals += 1
            fitness_parents[i] = y
            if y < best_y:
                best_y = y
                best_x = x_parents[i].copy()

        # ---- Main evolution loop ----
        while evals < self.budget:
            offspring_x = np.empty((self.lmbda, dim))
            offspring_sigma = np.empty((self.lmbda, dim))
            offspring_fitness = np.full(self.lmbda, np.inf)

            for i in range(self.lmbda):
                # Select two distinct parents uniformly for recombination
                idx = rng.choice(self.mu, size=2, replace=False)
                p1, p2 = idx[0], idx[1]

                # Intermediate recombination for x and sigma (geometric for sigma)
                x_recomb = 0.5 * (x_parents[p1] + x_parents[p2])
                # Sigma: log‑domain average -> geometric mean
                sigma_recomb = np.exp(0.5 * (np.log(sigma_parents[p1]) + np.log(sigma_parents[p2])))

                # Mutate step sizes (self‑adaptation)
                global_noise = rng.normal()    # one global normal
                per_dim_noise = rng.normal(size=dim)
                sigma_child = sigma_recomb * np.exp(self.tau * global_noise + self.tau_prime * per_dim_noise)

                # Mutate object variables
                x_child = x_recomb + sigma_child * rng.normal(size=dim)

                # Boundary handling: clamp
                x_child = np.clip(x_child, lower, upper)

                # Evaluate if budget allows
                if evals >= self.budget:
                    break
                y = func(x_child)
                evals += 1

                offspring_x[i] = x_child
                offspring_sigma[i] = sigma_child
                offspring_fitness[i] = y

                # Update best ever
                if y < best_y:
                    best_y = y
                    best_x = x_child.copy()

            # If no offspring were evaluated (budget exhausted), break
            if evals >= self.budget:
                break

            # (mu,lambda) selection: sort offspring by fitness, keep best mu
            order = np.argsort(offspring_fitness)
            x_parents = offspring_x[order[:self.mu]]
            sigma_parents = offspring_sigma[order[:self.mu]]
            fitness_parents = offspring_fitness[order[:self.mu]]

        return best_x, best_y
