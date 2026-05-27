import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (μ, λ)-Evolution Strategy with isotropic Gaussian mutation and
# cumulative step-size adaptation (CSA).  The algorithm maintains a population
# of candidate solutions, recombines them to produce offspring, mutates with
# a global step size, and selects the μ best offspring to become the next
# generation.  The step size is adapted online using the cumulative evolution
# path, which helps maintain a desired progress rate.
#
# Search state: A list of μ parent solutions (real vectors of dimension `dim`)
# and their corresponding fitness values, plus the current step size (sigma)
# and the cumulative step-size adaptation path (p_sigma).
#
# Candidate generation: Each generation produces λ offspring by first
# recombining two distinct parents chosen uniformly at random (discrete
# recombination per coordinate) and then adding isotropic Gaussian noise with
# standard deviation sigma.  The offspring are always clamped to the allowed
# bounds.
#
# Selection and replacement: Truncation selection: from the λ offspring, the
# μ best (lowest objective values) become the new parents.  No elitism
# (the previous parents are discarded entirely).  The overall best solution
# found so far is tracked separately and returned at the end.
#
# Adaptation: Step-size adaptation uses the cumulative step-size mechanism
# from CMA-ES (simplified).  An evolution path p_sigma is updated with the
# mean of the selected offspring (weighted average).  The length of the path
# is compared to the expected length under random selection, and sigma is
# increased if the path is longer than expected (indicating a consistent
# direction) and decreased if it is shorter (indicating oscillations).
#
# Exploration mechanisms: Large isotropic Gaussian mutations when sigma is
# large; discrete recombination shuffles parent coordinates and provides
# further diversity.
#
# Exploitation mechanisms: Selection of the best offspring focuses the
# population on promising regions.  The step-size adaptation automatically
# reduces sigma when the population converges, enabling fine-grained local
# search.
#
# Boundary handling: After mutation, each coordinate is clipped to the
# feasible bounds defined by func.lower/upper (or func.bounds.lb/ub).  No
# reflection or re‑sampling is used.
#
# Budget strategy: All function evaluations are counted.  The algorithm
# stops as soon as the budget of `budget` evaluations is exhausted.  The
# final generation may be incomplete (fewer offspring evaluated), but the
# best solution found is always returned.
#
# Closest known influences: CMA-ES (cumulative step-size adaptation idea)
# and classic (μ,λ)-ES.  The recombination and selection are simpler than
# full CMA-ES.
#
# Novelty or unusual aspects: The implementation uses the cumulative step-
# size path even though there is no covariance matrix update; this keeps the
# adaptation mechanism cheap while still providing principled step-size
# control.  The population size is set to λ = 4*dim and μ = λ//2, which is
# a common default.
#
# Failure modes: The algorithm may converge prematurely on very rugged
# landscapes if sigma shrinks too fast.  It can also get stuck if the initial
# sigma is too small relative to the search space.  For separable, smooth
# functions it converges reliably.  The clamping of coordinates can distort
# the mutation distribution near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Population size – rule of thumb: λ = 4 * dim, μ = λ // 2
        self.lam = max(4 * dim, 10)   # ensure at least 10 offspring
        self.mu = self.lam // 2

        # Step-size adaptation parameters (typical values)
        self.c_c = (self.mu + 2) / (self.dim + self.mu + 3)   # path cumulation rate
        self.c_sigma = (self.mu + 2) / (self.dim + self.mu + 3) + 0.3  # damping factor
        self.d_sigma = 1.0 + 2 * max(0, np.sqrt((self.mu - 1) / (self.dim + 1)) - 1)
        self.chi_n = np.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim)
                                          + 1.0 / (21.0 * self.dim ** 2))

        # State variables
        self.sigma = 1.0                     # initial step size
        self.p_sigma = np.zeros(self.dim)    # evolution path for step size
        self.mean = None                     # will be set after bounds are known
        self.parents = None                  # list of (x, y) for mu parents

        # Best found so far
        self.best_x = None
        self.best_y = np.inf

        # Evaluation counter
        self.evaluations = 0

    def _get_bounds(self, func):
        """Extract lower and upper bounds from the objective function."""
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)
        else:
            raise ValueError("Cannot find bounds in func object")
        # Ensure same shape
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)
        return lb, ub

    def _evaluate(self, x, func):
        """Evaluate a point if budget remains, update best."""
        if self.evaluations >= self.budget:
            return None
        y = func(x)
        self.evaluations += 1
        if y < self.best_y:
            self.best_y = y
            self.best_x = x.copy()
        return y

    def __call__(self, func):
        # Determine bounds
        lb, ub = self._get_bounds(func)
        # Initial mean: center of search space
        self.mean = (lb + ub) / 2.0
        self.sigma = 0.2 * (ub - lb).mean()  # initial step size ~20% of domain

        # Generate initial population uniformly
        pop = []
        for _ in range(self.mu):
            x = np.random.uniform(lb, ub, self.dim)
            y = self._evaluate(x, func)
            if y is None:  # budget exhausted during initial sampling
                return self.best_x, self.best_y
            pop.append((x, y))

        # Initial parents from the best in the randomly sampled set
        pop.sort(key=lambda t: t[1])
        self.parents = pop[:self.mu]   # keep the best mu

        # Initial evolution path (zero mean already)
        self.p_sigma = np.zeros(self.dim)

        # Main generation loop
        while self.evaluations < self.budget:
            # Generate λ offspring
            offspring = []
            # Recombination weights: uniform for simplicity
            # We'll use discrete recombination: each coordinate from a random parent
            # (among the μ parents) – this is a standard (μ,λ)-ES variant.
            for _ in range(self.lam):
                # Choose two distinct parents for recombination
                idx1, idx2 = np.random.choice(self.mu, size=2, replace=False)
                x1, _ = self.parents[idx1]
                x2, _ = self.parents[idx2]
                # Discrete recombination: each coordinate taken from parent1 or parent2
                coin = np.random.randint(0, 2, self.dim).astype(float)
                z = coin * x1 + (1 - coin) * x2
                # Mutation
                z += self.sigma * np.random.randn(self.dim)
                # Clamp to bounds
                z = np.clip(z, lb, ub)
                # Evaluate
                y = self._evaluate(z, func)
                if y is None:
                    # Budget exhausted
                    return self.best_x, self.best_y
                offspring.append((z, y))

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[1])

            # Select μ best
            self.parents = offspring[:self.mu]

            # Recompute mean as weighted average of selected (equal weights)
            # Weighted recombination is used for the step-size update.
            # We use the best μ offspring.
            x_old = self.mean.copy()
            w = np.ones(self.mu) / self.mu   # equal weights
            self.mean = np.sum(w[:, None] * np.array([p[0] for p in self.parents]), axis=0)

            # Update cumulative path for step-size adaptation
            # p_sigma = (1 - c_c) * p_sigma + sqrt(c_c * (2 - c_c) * mu_eff) * (mean - x_old) / sigma
            mu_eff = 1.0 / np.sum(w**2)  # effective sample size
            z = (self.mean - x_old) / self.sigma
            c = self.c_c
            self.p_sigma = (1 - c) * self.p_sigma + \
                           np.sqrt(c * (2 - c) * mu_eff) * z

            # Update sigma
            # sigma = sigma * exp( (c_sigma/d_sigma) * (||p_sigma|| / chi_n - 1) )
            factor = (self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_n - 1)
            self.sigma = self.sigma * np.exp(factor)
            # Keep sigma from getting ridiculously small or large
            self.sigma = np.clip(self.sigma, 1e-10, (ub - lb).max())

        return self.best_x, self.best_y
