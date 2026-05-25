# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (μ + λ) Evolution Strategy with a single global step‑size that adapts over time. The algorithm starts with a small population, mutates individuals by adding Gaussian noise, selects the best μ candidates each generation, and adjusts the mutation spread using a 1/5 rule.
# Search state: Maintains a population of μ vectors, the best solution observed so far, and the current step‑size σ.
# Candidate generation: Offspring are created by randomly picking a parent from the current population and perturbing it with Gaussian noise of standard deviation σ; the resulting vector is clipped to the problem’s bounds.
# Selection and replacement: After evaluating the offspring, the combined parent + offspring set (μ + λ) is ranked by fitness and the top μ individuals become the next generation.
# Adaptation: σ is increased by 20 % when any offspring improves the global best, otherwise it is decreased by 10 %. σ is also bounded relative to the search range to avoid extremely small or large steps.
# Exploration mechanisms: Random initialization across the whole domain and a relatively large σ early in the run promote global exploration.
# Exploitation mechanisms: As σ shrinks, mutations become finer, allowing the population to refine promising solutions.
# Boundary handling: Every candidate is clipped to the user‑provided lower/upper bounds before evaluation.
# Budget strategy: The algorithm respects the evaluation budget strictly. The initial population consumes μ evaluations; each subsequent iteration consumes λ (or fewer if the remaining budget is insufficient). The main loop stops exactly when the budget is exhausted.
# Closest known influences: Classic (μ + λ) Evolution Strategies (Rechenberg, 1973) combined with a global step‑size adaptation rule.
# Novelty or unusual aspects: Population size grows only logarithmically with dimension, keeping the approach cheap for high‑dimensional problems. The simple 1/5 rule avoids covariance matrix estimation while still providing a reasonable adaptation mechanism.
# Failure modes: May converge prematurely on highly multi‑modal landscapes if σ shrinks too quickly. Lacks directional scaling, so it can be inefficient on badly scaled functions where optimal step sizes differ across dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple (μ + λ) Evolution Strategy for black‑box minimization.

    The algorithm respects the evaluation budget, reads bounds from the
    provided function object, and returns the best found solution.
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimization and return the best solution found.

        Parameters
        ----------
        func : callable
            Black‑box objective. It must support one of the following:
            - func.lower / func.upper (array‑like)
            - func.bounds.lb / func.bounds.ub

        Returns
        -------
        best_x : np.ndarray
            Solution vector that achieved the lowest objective value.
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------------
        # Determine search space bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: use a large hypercube
            lower = np.full(self.dim, -10.0)
            upper = np.full(self.dim, 10.0)

        # Ensure lower/upper are vectors of length dim
        if lower.ndim == 0:
            lower = np.full(self.dim, lower.item())
        if upper.ndim == 0:
            upper = np.full(self.dim, upper.item())

        # ------------------------------------------------------------------
        # Initialisation
        # ------------------------------------------------------------------
        # Population size (μ) grows logarithmically with dimension
        mu = max(1, int(5 + 3 * np.log(self.dim)))
        if mu > self.budget:
            mu = self.budget          # cannot exceed total budget
        lambda_ = mu                  # number of offspring per generation

        # Generate initial population uniformly within the bounds
        pop_x = [np.random.uniform(lower, upper) for _ in range(mu)]
        pop_y = [func(x) for x in pop_x]
        evals = mu

        # Track best solution
        best_idx = np.argmin(pop_y)
        best_x = pop_x[best_idx].copy()
        best_y = pop_y[best_idx]

        # Initial global step size (σ)
        sigma = np.mean(upper - lower) / 3.0

        # ------------------------------------------------------------------
        # Main evolution loop
        # ------------------------------------------------------------------
        while evals < self.budget:
            # How many offspring can we afford in this iteration?
            remaining = self.budget - evals
            num_offspring = min(lambda_, remaining)

            # Produce and evaluate offspring
            for _ in range(num_offspring):
                # Choose a parent uniformly at random
                parent_idx = np.random.randint(mu)
                parent = pop_x[parent_idx]

                # Mutate by adding Gaussian noise
                child = parent + sigma * np.random.randn(self.dim)

                # Clip to feasible region
                child = np.clip(child, lower, upper)

                # Evaluate
                child_y = func(child)
                evals += 1

                # Update global best if needed
                if child_y < best_y:
                    best_x = child.copy()
                    best_y = child_y

                # Add child to the temporary population for selection
                pop_x.append(child)
                pop_y.append(child_y)

            # ------------------------------------------------------------------
            # Selection: keep the best μ individuals
            # ------------------------------------------------------------------
            indices = np.argsort(pop_y)[:mu]
            pop_x = [pop_x[i] for i in indices]
            pop_y = [pop_y[i] for i in indices]

            # ------------------------------------------------------------------
            # Adapt step size (1/5 rule)
            # ------------------------------------------------------------------
            # If any offspring improved the best known solution, increase σ
            improved = any(y < best_y for y in pop_y[-num_offspring:])
            if improved:
                sigma *= 1.2
            else:
                sigma *= 0.9

            # Keep σ within reasonable bounds relative to the search range
            range_mean = np.mean(upper - lower)
            sigma = np.clip(sigma, range_mean * 0.01, range_mean * 2.0)

        return best_x, best_y
