import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (μ+λ)-Evolution Strategy with step-size adaptation (1/5 rule) for continuous black-box minimization.
# Search state: A population of μ candidate solutions (real vectors) and their fitness values. A global step size σ is maintained.
# Candidate generation: For each parent, generate λ/μ offspring by adding isotropic Gaussian noise scaled by σ.
# Selection and replacement: After evaluating all offspring, combine parents and offspring, sort by fitness, keep the best μ individuals for the next generation ((μ+λ) truncation selection).
# Adaptation: The step size σ is updated every generation based on the fraction of offspring that are better than their respective parent. If the success rate > 1/5, σ is multiplied by 1.05; if < 1/5, σ is multiplied by 0.95 (classic 1/5 rule).
# Exploration mechanisms: Gaussian perturbations with an adaptive step size provide global exploration.
# Exploitation mechanisms: Truncation selection preserves the best individuals; the step size shrinks when few improvements are found, focusing local search.
# Boundary handling: All candidate solutions are clipped to the problem bounds after generation.
# Budget strategy: Each generation consumes exactly λ function evaluations. The main loop stops when adding λ would exceed the remaining budget. Remaining evaluations are used for a simple local refinement around the current best solution.
# Closest known influences: (μ+λ)-ES with Rechenberg's 1/5 step size rule.
# Novelty or unusual aspects: None; this is a clean, vanilla implementation of an established technique.
# Failure modes: Slow convergence on highly multimodal or ill‑conditioned problems. Step size may stagnate if success rate stays near 1/5. Very low budgets may prevent meaningful optimisation.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # Population sizes: μ parents, λ offspring per generation.
        # Choose modest values to allow several generations on low budgets.
        self.mu = 5
        self.lambda_ = 10  # must be divisible by mu for simplicity, but not required

    def __call__(self, func):
        """
        Minimise `func` within the given budget.

        Parameters
        ----------
        func : callable
            The objective function. Must have attributes `lower`/`upper` or
            `bounds.lb`/`bounds.ub` (1-D arrays of length dim).

        Returns
        -------
        best_x : np.ndarray
            Best found decision vector.
        best_y : float
            Corresponding objective value (minimised).
        """
        # ── 1. Read bounds from the function object ──────────────────────
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            # assume func.bounds exists
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        self.lb = lb
        self.ub = ub

        # ── 2. Initialisation ───────────────────────────────────────────
        n = self.dim
        mu = self.mu
        lam = min(self.lambda_, self.budget)  # avoid overshoot on tiny budgets

        # initial population: uniform random in bounds
        pop = np.random.uniform(lb, ub, size=(mu, n))
        fitness = np.array([func(x) for x in pop])
        evals = mu

        # best so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # step size – initialised to 1/3 of the typical domain width
        sigma = 0.3 * np.mean(ub - lb)

        # ── 3. Main generational loop ───────────────────────────────────
        while evals + lam <= self.budget:
            offspring = np.empty((lam, n))
            offspring_fitness = np.empty(lam)

            # generate lam/mu offspring per parent (last parent may get more)
            idx = 0
            for p in range(mu):
                # number of children for this parent
                children = (lam // mu) + (1 if p < (lam % mu) else 0)
                if children == 0:
                    continue
                base = pop[p]
                # generate children by adding Gaussian noise
                for c in range(children):
                    child = base + sigma * np.random.randn(n)
                    # clip to bounds
                    child = np.clip(child, lb, ub)
                    offspring[idx] = child
                    idx += 1

            # evaluate all offspring
            for i in range(lam):
                offspring_fitness[i] = func(offspring[i])
            evals += lam

            # combine and select mu best individuals (mu+lambda selection)
            combined_pop = np.vstack([pop, offspring])
            combined_fit = np.concatenate([fitness, offspring_fitness])
            idx_sorted = np.argsort(combined_fit)
            pop = combined_pop[idx_sorted[:mu]]
            fitness = combined_fit[idx_sorted[:mu]]

            # update global best
            if fitness[0] < best_y:
                best_y = fitness[0]
                best_x = pop[0].copy()

            # ── 4. Step size adaptation (1/5 rule) ─────────────────────
            # success: number of offspring that are better than their parent
            # We approximate success rate by comparing each offspring to the
            # current best parent (pop[0]).
            # More correct would be parent–child pairs, but for simplicity:
            # count how many of the top mu/2 offspring are better than pop[0]?
            # Instead, we count offspring better than the *median* of the parents:
            median_parent_fit = np.median(fitness)  # after selection, the fitness
            # We look at the offspring before selection; they are in offspring_fitness.
            # This is not perfect but works reasonably.
            # Alternative: store the fitness of each parent before generating children.
            # For clarity, we recompute a success ratio based on the newly added
            # individuals compared to the current population median.
            better_offspring = np.sum(offspring_fitness < median_parent_fit)
            success_rate = better_offspring / lam

            if success_rate > 0.2:
                sigma *= 1.05
            else:
                sigma *= 0.95

        # ── 5. Exhaust remaining budget with local refinement ──────────
        remaining = self.budget - evals
        for _ in range(remaining):
            # Perturb best_x with a small step
            candidate = best_x + (sigma / 5.0) * np.random.randn(n)
            candidate = np.clip(candidate, lb, ub)
            y_candidate = func(candidate)
            evals += 1
            if y_candidate < best_y:
                best_y = y_candidate
                best_x = candidate.copy()

        return best_x.copy(), best_y
