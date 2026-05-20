# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adaptive Differential Evolution (DE) algorithm based on the jDE variant. It dynamically adjusts its mutation scale (F) and crossover rate (Cr) for each individual to adapt to the objective function's landscape during the search.
# Search state: A population of candidate solutions, their associated fitness values, and individual control parameters (F and Cr).
# Candidate generation: Uses DE/rand/1 mutation to create a mutant vector, followed by binomial crossover to generate a trial vector.
# Selection and replacement: Greedy selection; the trial vector replaces the parent in the population if it results in a fitness value less than or equal to the parent's fitness.
# Adaptation: Stochastic parameter adaptation where F and Cr have a 10% probability of being reset to new random values in each generation, allowing the population to explore different search behaviors.
# Exploration mechanisms: DE/rand/1 mutation provides global exploration, while the stochastic reset of F and Cr prevents premature convergence.
# Exploitation mechanisms: Binomial crossover and greedy selection focus the search on promising regions identified by the population.
# Boundary handling: Trial vectors are clipped to the specified lower and upper bounds using numpy's clip function.
# Budget strategy: The algorithm tracks evaluations and terminates immediately once the budget is exhausted, ensuring no excess calls to the objective function.
# Closest known influences: The jDE algorithm proposed by Brest et al. (2006).
# Novelty or unusual aspects: A compact implementation designed to handle very low budgets by scaling population size and transitioning to random search if the population is too small for DE mutation.
# Failure modes: May perform poorly on extremely high-dimensional problems with very low budgets where the population cannot sufficiently cover the search space.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the optimizer.
        :param budget: Total number of function evaluations allowed.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Executes the optimization process.
        :param func: The objective function to minimize.
        :return: A tuple (best_x, best_y).
        """
        # Retrieve bounds from the function object
        if hasattr(func, "lower"):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, "bounds"):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Default fallback if bounds are missing
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        evals = 0
        best_x = None
        best_y = float('inf')

        # Heuristic for population size
        pop_size = int(max(min(self.budget // 3, 15 * self.dim), 5))
        pop_size = min(pop_size, self.budget)

        # Initialize population
        pop = lb + np.random.rand(pop_size, self.dim) * (ub - lb)
        fitness = np.full(pop_size, float('inf'))
        
        # Individual adaptation parameters
        F_vals = np.full(pop_size, 0.5)
        Cr_vals = np.full(pop_size, 0.9)

        # Initial evaluation
        for i in range(pop_size):
            if evals >= self.budget:
                break
            y = func(pop[i])
            evals += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Handle cases where budget is too small for DE
        if pop_size < 4:
            while evals < self.budget:
                trial = lb + np.random.rand(self.dim) * (ub - lb)
                y = func(trial)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = trial.copy()
            return best_x, best_y

        # Main DE Loop
        while evals < self.budget:
            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # Parameter Adaptation (jDE style)
                if np.random.rand() < 0.1:
                    F_vals[i] = 0.1 + np.random.rand() * 0.9
                if np.random.rand() < 0.1:
                    Cr_vals[i] = np.random.rand()

                # Mutation: DE/rand/1
                idxs = [idx for idx in range(pop_size) if idx != i]
                abc = np.random.choice(idxs, 3, replace=False)
                a, b, c = pop[abc]
                mutant = a + F_vals[i] * (b - c)
                
                # Crossover: Binomial
                cross_mask = np.random.rand(self.dim) < Cr_vals[i]
                # Ensure at least one dimension is inherited from mutant
                cross_mask[np.random.randint(self.dim)] = True
                
                trial = np.where(cross_mask, mutant, pop[i])
                
                # Boundary handling
                trial = np.clip(trial, lb, ub)

                # Selection
                y_trial = func(trial)
                evals += 1
                
                if y_trial <= fitness[i]:
                    fitness[i] = y_trial
                    pop[i] = trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()

        return best_x, best_y
