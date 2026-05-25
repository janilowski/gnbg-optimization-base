# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact (μ+λ) Evolutionary Algorithm for continuous black‑box minimization. It maintains a population of candidate solutions, generates offspring by mutating selected parents, evaluates them, and keeps the best individuals for the next generation. An isotropic Gaussian mutation with an adaptive step size balances exploration and exploitation.
#
# Search state:
#   - Population: array of shape (μ, dim) containing current candidate points.
#   - Fitness array: shape (μ,) with the objective value of each individual.
#   - Best solution found so far (best_x, best_y).
#   - Evaluation counter (evals) tracking how many function calls have been made.
#   - Current mutation step size sigma controlling the spread of Gaussian perturbations.
#
# Candidate generation:
#   - For each offspring, a parent is chosen by binary tournament from the current population.
#   - The selected parent is mutated by adding zero‑mean Gaussian noise scaled by sigma.
#   - The resulting vector is clipped to the problem’s lower and upper bounds.
#
# Selection and replacement:
#   - Parents and offspring are concatenated into a combined pool.
#   - The μ individuals with the lowest fitness are selected to form the next generation (elitist replacement).
#
# Adaptation:
#   - sigma is increased by a factor of 1.1 when any offspring improves the global best.
#   - Otherwise sigma is decreased by a factor of 0.9.
#   - sigma is kept within a reasonable range relative to the domain size to avoid premature convergence or excessive jitter.
#
# Exploration mechanisms:
#   - Large initial sigma and isotropic Gaussian mutations promote exploration across all dimensions.
#   - Maintaining a diverse pool of parents and offspring helps explore the search space.
#
# Exploitation mechanisms:
#   - When no improvement is observed, sigma shrinks, focusing the search around promising regions.
#   - Elitism preserves high‑quality solutions across generations.
#
# Boundary handling:
#   - After mutation, each coordinate is clipped to the provided lower and upper bounds, ensuring feasibility.
#
# Budget strategy:
#   - The algorithm strictly counts each function evaluation and stops as soon as the pre‑defined budget is reached.
#   - It never performs more evaluations than allowed, even if the main loop would continue.
#
# Closest known influences:
#   - Classic (μ+λ) Evolutionary Strategies and simple adaptive step‑size mechanisms such as the 1/5‑th rule.
#
# Novelty or unusual aspects:
#   - The sigma adaptation is intentionally minimal (based solely on whether the global best improved) to keep the code compact and robust across arbitrary dimensions and limited budgets.
#
# Failure modes:
#   - With very tight evaluation budgets the algorithm may not converge to a good solution.
#   - If sigma shrinks too rapidly, the search can become trapped in local minima, especially in high‑dimensional landscapes.
#   - The heuristic population size may be insufficient for highly multi‑modal problems, leading to premature convergence.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple (μ+λ) Evolutionary Algorithm for continuous black‑box minimization.

    The algorithm maintains a population of μ candidates, generates λ offspring
    each generation by mutating selected parents, evaluates them, and retains the
    best μ individuals for the next generation. An adaptive Gaussian mutation
    step size (sigma) balances exploration and exploitation.

    Parameters
    ----------
    budget : int
        Maximum number of objective function evaluations allowed.
    dim : int
        Dimensionality of the problem (number of decision variables).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # Population size (μ) is chosen heuristically but capped by the budget.
        # It must be at least 1 and at most budget//5 to leave room for offspring.
        self.mu = min(max(5, dim), max(1, (budget - 1) // 3))

        # Number of offspring (λ) produced each generation.
        self.lambda_ = self.mu * 2

    def __call__(self, func):
        """
        Run the evolutionary algorithm on the given objective function.

        Parameters
        ----------
        func : callable
            A black‑box objective function that accepts a 1‑D array of shape (dim,)
            and returns a scalar (the objective value). The function must expose
            either ``lower``/``upper`` attributes or a ``bounds`` object with
            ``lb``/``ub`` attributes.

        Returns
        -------
        best_x : np.ndarray
            The decision vector that achieved the lowest objective value.
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------------
        # Determine problem bounds (lower and upper limits for each variable).
        # ------------------------------------------------------------------
        bounds = getattr(func, 'bounds', None)
        if bounds is not None:
            lower = getattr(bounds, 'lb', None)
            upper = getattr(bounds, 'ub', None)
        else:
            lower = getattr(func, 'lower', None)
            upper = getattr(func, 'upper', None)

        if lower is None or upper is None:
            raise ValueError("Cannot determine problem bounds: expected "
                             "func.lower/func.upper or func.bounds.lb/func.bounds.ub")

        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        # Ensure lower/upper are arrays of length dim (broadcast if necessary).
        if lower.shape != (self.dim,):
            lower = np.broadcast_to(lower, (self.dim,)).copy()
        if upper.shape != (self.dim,):
            upper = np.broadcast_to(upper, (self.dim,)).copy()

        # ------------------------------------------------------------------
        # Initialize population.
        # ------------------------------------------------------------------
        mu = self.mu
        lam = self.lambda_

        # If the budget is too small to evaluate a full population, shrink μ.
        if self.budget < mu:
            mu = self.budget
            lam = 0

        # Generate random individuals uniformly within the bounds.
        pop = lower + (upper - lower) * np.random.rand(mu, self.dim)

        # Evaluate initial population.
        fit = np.empty(mu, dtype=float)
        evals = 0
        for i in range(mu):
            fit[i] = func(pop[i])
            evals += 1

        # Track the best solution found so far.
        best_idx = int(np.argmin(fit))
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # ------------------------------------------------------------------
        # Initial mutation step size (sigma). Heuristic: ~1/6 of the average range.
        # ------------------------------------------------------------------
        sigma = np.mean(upper - lower) / 6.0

        # ------------------------------------------------------------------
        # Main evolutionary loop.
        # ------------------------------------------------------------------
        while evals < self.budget:
            # Determine how many offspring we can afford given the remaining budget.
            remaining = self.budget - evals
            k = min(lam, remaining)
            if k == 0:
                break

            # Allocate arrays for offspring.
            offs = np.empty((k, self.dim), dtype=float)
            off_fit = np.empty(k, dtype=float)

            # ----- Generate and evaluate offspring -----
            for i in range(k):
                # Binary tournament selection.
                p1 = np.random.randint(mu)
                p2 = np.random.randint(mu)
                if fit[p1] < fit[p2]:
                    parent = pop[p1]
                else:
                    parent = pop[p2]

                # Gaussian mutation.
                mutant = parent + sigma * np.random.randn(self.dim)

                # Clip to bounds.
                mutant = np.clip(mutant, lower, upper)

                offs[i] = mutant
                off_fit[i] = func(mutant)
                evals += 1

                # Update best solution if this offspring is better.
                if off_fit[i] < best_y:
                    best_y = off_fit[i]
                    best_x = offs[i].copy()

            # ----- Combine parents and offspring -----
            combined_pop = np.vstack([pop, offs])
            combined_fit = np.concatenate([fit, off_fit])

            # ----- Elitist selection: keep the μ best individuals -----
            order = np.argsort(combined_fit)
            pop = combined_pop[order[:mu]]
            fit = combined_fit[order[:mu]]

            # ----- Adapt mutation step size sigma -----
            # Increase sigma when any offspring improved the global best.
            if np.any(off_fit < best_y):
                sigma *= 1.1
            else:
                sigma *= 0.9

            # Keep sigma within a sensible range relative to the domain size.
            sigma = np.clip(sigma, np.mean(upper - lower) / 1e4, np.mean(upper - lower) / 2.0)

        # Return the best solution found.
        return best_x, best_y
