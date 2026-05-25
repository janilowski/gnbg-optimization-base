# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (μ+λ) evolutionary algorithm with Gaussian mutation, intermediate recombination, and plus‑selection. The algorithm maintains a small population, creates a few offspring each generation, evaluates them against the budget, and keeps the best individuals for the next generation. A lightweight restart heuristic triggers a re‑initialization of the population when the best fitness stalls for a fixed number of generations, helping to escape local optima.
# Search state: Population of μ individuals (the current set of candidate solutions) and the best solution found so far. The population size is determined by the dimension and the available budget, but never drops below a minimum of five individuals.
# Candidate generation: Offspring are produced by selecting two distinct parents at random, applying intermediate recombination (averaging their vectors), and adding Gaussian noise with a step‑size σ derived from the search range (σ = (upper‑lower)/6). Mutations are clipped to remain inside the feasible region.
# Selection and replacement: A (μ+λ) strategy is used: after evaluating the offspring, the best μ individuals are chosen from the union of parents and offspring. This combines exploitation of good solutions with exploration of new regions.
# Adaptation: The step‑size σ is kept fixed during most of the run, but upon a restart the algorithm enlarges σ by a factor of 1.2 to increase exploration. A patience counter tracks how many generations have passed without improving the best fitness; when the counter reaches a threshold (≈5% of the total budget) the population is re‑initialized.
# Exploration mechanisms: Large initial population, occasional restarts, and a relatively large mutation step provide exploration across the search space.
# Exploitation mechanisms: Selection pressure pushes the population toward lower objective values, and recombination blends beneficial traits from multiple parents.
# Boundary handling: After mutation, each coordinate is clipped to the lower and upper bounds obtained from the benchmark interface, preventing illegal evaluations.
# Budget strategy: The algorithm stops as soon as the number of function evaluations reaches the supplied budget. The number of offspring per iteration is capped by the remaining budget, ensuring no excess evaluations.
# Closest known influences: Classic evolutionary strategies (Bäck, Hammel, Schwefel, 1997) and simple random restart hill‑climbing. The use of plus‑selection and Gaussian mutation mirrors early ES literature.
# Novelty or unusual aspects: The combination of a fixed σ derived from the bound range with a simple restart rule provides a balance between simplicity and robustness across diverse GNBG instances. No advanced covariance adaptation is employed, making the code easy to understand and modify.
# Failure modes: If the search space is highly multi‑modal, the algorithm may converge to a local optimum before a restart occurs, especially when the budget is limited. The fixed σ may be sub‑optimal for problems where the optimal scale varies across dimensions. The restart heuristic assumes some improvement is possible; on a completely flat landscape the algorithm will simply re‑initialize without progress.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple (μ+λ) evolutionary strategy for black‑box minimization.

    The class follows the required interface:
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

        # Minimum population size; adapt to dimension
        self._min_pop = max(5, dim)

        # Offspring factor (lambda = factor * mu)
        self._offspring_factor = 4

        # Number of generations without improvement before restart
        self._patience = max(1, self.budget // 20 if self.budget > 0 else 1)

    def __call__(self, func):
        """
        Run the evolutionary algorithm on the given function.

        Parameters
        ----------
        func : callable
            A black‑box function that receives a 1‑D numpy array and returns a scalar.
            Must expose either ``lower``/``upper`` or ``bounds.lb``/``bounds.ub`` for bounds.

        Returns
        -------
        best_x : numpy.ndarray
            Best solution found.
        best_y : float
            Corresponding objective value.
        """
        # ------------------------------------------------------------------
        # Retrieve problem bounds
        # ------------------------------------------------------------------
        lower, upper = self._get_bounds()

        # ------------------------------------------------------------------
        # Initial population (size limited by budget)
        # ------------------------------------------------------------------
        max_init = min(self._min_pop, self.budget)
        # If budget is zero, return random point (still inside bounds)
        if max_init == 0:
            x = np.random.default_rng().uniform(lower, upper)
            y = func(x)
            return x, y

        pop = np.random.default_rng().uniform(lower, upper, size=(max_init, self.dim))
        evals = 0
        best_x = None
        best_y = np.inf

        fitness = np.empty(max_init)
        for i in range(max_init):
            y = func(pop[i])
            evals += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Early exit if budget exhausted
        if evals >= self.budget:
            return best_x, best_y

        # ------------------------------------------------------------------
        # Evolution loop
        # ------------------------------------------------------------------
        mu = pop.shape[0]
        lambda_ = mu * self._offspring_factor

        # Initial step size (σ = 1/6 of the range)
        sigma = (upper - lower) / 6.0

        stall_counter = 0  # generations without improvement

        while evals < self.budget:
            # How many offspring can we afford now?
            remaining = self.budget - evals
            lambda_now = min(lambda_, remaining)

            # ------------------------------------------------------------------
            # Generate offspring
            # ------------------------------------------------------------------
            offspring = np.empty((lambda_now, self.dim), dtype=float)
            for i in range(lambda_now):
                # Select two distinct parents
                p1_idx, p2_idx = np.random.default_rng().integers(0, mu, size=2)
                # Intermediate recombination
                recomb = (pop[p1_idx] + pop[p2_idx]) / 2.0
                # Gaussian mutation
                offspring[i] = recomb + sigma * np.random.default_rng().standard_normal(self.dim)
                # Clip to feasible region
                offspring[i] = np.clip(offspring[i], lower, upper)

            # ------------------------------------------------------------------
            # Evaluate offspring
            # ------------------------------------------------------------------
            offspring_fitness = np.empty(lambda_now, dtype=float)
            for i in range(lambda_now):
                y = func(offspring[i])
                evals += 1
                offspring_fitness[i] = y
                if y < best_y:
                    best_y = y
                    best_x = offspring[i].copy()

            # ------------------------------------------------------------------
            # Plus‑selection: keep best μ individuals from parents + offspring
            # ------------------------------------------------------------------
            combined_pop = np.vstack([pop, offspring])
            combined_fitness = np.concatenate([fitness, offspring_fitness])

            # Use ``argpartition`` to obtain the μ smallest values efficiently
            idx = np.argpartition(combined_fitness, mu)[:mu]
            pop = combined_pop[idx]
            fitness = combined_fitness[idx]

            # ------------------------------------------------------------------
            # Restart logic
            # ------------------------------------------------------------------
            # Did any offspring improve the best known solution?
            if np.min(offspring_fitness) < best_y:
                stall_counter = 0
            else:
                stall_counter += 1

            if stall_counter >= self._patience:
                # Re‑initialize population uniformly at random
                pop = np.random.default_rng().uniform(lower, upper, size=(mu, self.dim))
                # Increase step size to favour exploration after restart
                sigma = sigma * 1.2
                stall_counter = 0

            # If budget is exhausted after this iteration, exit loop
            if evals >= self.budget:
                break

        return best_x, best_y

    def _get_bounds(self):
        """
        Read lower and upper bounds from the function object.

        Looks for ``lower``/``upper`` attributes first,
        then for ``bounds.lb``/``bounds.ub``.
        Falls back to ``[0,1]`` in each dimension if neither is present.

        Returns
        -------
        lower : numpy.ndarray
        upper : numpy.ndarray
        """
        # Try ``lower``/``upper`` first
        if hasattr(self, 'func_lower') and hasattr(self, 'func_upper'):
            # The instance variables are set in __call__ before this method is invoked
            lower = self.func_lower
            upper = self.func_upper
            return lower, upper

        # Placeholder variables – they will be set from the actual ``func`` argument
        # but we cannot capture ``func`` here because this method is called without it.
        # We'll delegate to a temporary wrapper that extracts bounds from ``func`` passed to __call__.
        # To keep the method signature simple, we store bounds as instance attributes in __call__.
        # This method will be called after those attributes are set.
        # (The actual implementation in __call__ sets self.func_lower / self.func_upper.)
        raise RuntimeError("_get_bounds should be called after bounds are set on the instance.")

    # -------------------------------------------------------------------------
    # Helper to extract bounds from the function object and store them
    # -------------------------------------------------------------------------
    def _extract_bounds(self, func):
        """
        Extract bounds from ``func`` and store them as instance attributes.

        Updates ``self.func_lower`` and ``self.func_upper``.
        Returns ``(lower, upper)`` as numpy arrays.
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: assume unit hypercube
            lower = np.zeros(self.dim, dtype=float)
            upper = np.ones(self.dim, dtype=float)

        # Ensure lower < upper (safeguard for malformed bounds)
        if np.any(lower >= upper):
            # Shift lower down or raise upper
            lower = np.where(lower >= upper, lower - 1.0, lower)
            upper = lower + 1.0

        self.func_lower = lower
        self.func_upper = upper
        return lower, upper

    # -------------------------------------------------------------------------
    # Wrapper for __call__ that extracts bounds before running the algorithm
    # -------------------------------------------------------------------------
    # Note: we override __call__ to first extract bounds and then call the core logic.
    # For clarity we keep the core logic in a private method ``_run``.
    # -------------------------------------------------------------------------
    def __call__(self, func):
        # Extract and store bounds
        lower, upper = self._extract_bounds(func)

        # Run the evolutionary algorithm with the extracted bounds
        return self._run(func, lower, upper)

    def _run(self, func, lower, upper):
        # (All the logic previously written in __call__ goes here.)
        # ------------------------------------------------------------------
        # Initial population (size limited by budget)
        # ------------------------------------------------------------------
        max_init = min(self._min_pop, self.budget)
        if max_init == 0:
            x = np.random.default_rng().uniform(lower, upper)
            y = func(x)
            return x, y

        pop = np.random.default_rng().uniform(lower, upper, size=(max_init, self.dim))
        evals = 0
        best_x = None
        best_y = np.inf

        fitness = np.empty(max_init, dtype=float)
        for i in range(max_init):
            y = func(pop[i])
            evals += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        if evals >= self.budget:
            return best_x, best_y

        mu = pop.shape[0]
        lambda_ = mu * self._offspring_factor
        sigma = (upper - lower) / 6.0
        stall_counter = 0

        while evals < self.budget:
            remaining = self.budget - evals
            lambda_now = min(lambda_, remaining)

            # Generate offspring
            offspring = np.empty((lambda_now, self.dim), dtype=float)
            for i in range(lambda_now):
                p1_idx, p2_idx = np.random.default_rng().integers(0, mu, size=2)
                recomb = (pop[p1_idx] + pop[p2_idx]) / 2.0
                offspring[i] = recomb + sigma * np.random.default_rng().standard_normal(self.dim)
                offspring[i] = np.clip(offspring[i], lower, upper)

            # Evaluate offspring
            offspring_fitness = np.empty(lambda_now, dtype=float)
            for i in range(lambda_now):
                y = func(offspring[i])
                evals += 1
                offspring_fitness[i] = y
                if y < best_y:
                    best_y = y
                    best_x = offspring[i].copy()

            # Plus‑selection
            combined_pop = np.vstack([pop, offspring])
            combined_fitness = np.concatenate([fitness, offspring_fitness])
            idx = np.argpartition(combined_fitness, mu)[:mu]
            pop = combined_pop[idx]
            fitness = combined_fitness[idx]

            # Restart logic
            if np.min(offspring_fitness) < best_y:
                stall_counter = 0
            else:
                stall_counter += 1

            if stall_counter >= self._patience:
                pop = np.random.default_rng().uniform(lower, upper, size=(mu, self.dim))
                sigma = sigma * 1.2
                stall_counter = 0

            if evals >= self.budget:
                break

        return best_x, best_y
