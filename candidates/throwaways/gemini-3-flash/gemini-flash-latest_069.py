# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A minimalist Differential Evolution (DE/rand/1/bin) implementation designed for robustness. It adjusts its population size based on the available budget and dimensionality.
# Search state: A population of candidate solution vectors and their associated fitness values, along with the best solution found so far.
# Candidate generation: Uses differential mutation (F=0.5) and binomial crossover (CR=0.7) to create trial vectors.
# Selection and replacement: Standard DE greedy selection where trial vectors replace parents if they improve the objective value.
# Adaptation: The population size is dynamically determined at the start to balance between global exploration and the number of iterations allowed by the budget.
# Exploration mechanisms: Mutation based on random differences between population members provides diverse search directions.
# Exploitation mechanisms: Crossover and greedy selection focus the search on promising regions identified by the population.
# Boundary handling: Candidate vectors are clipped to the hypercube defined by the problem bounds.
# Budget strategy: Strictly monitors evaluations and stops precisely at the limit; uses random sampling if the budget is too small for a full DE population.
# Closest known influences: Storn and Price's Differential Evolution algorithm.
# Novelty or unusual aspects: A budget-dependent fallback to random search for extremely low-budget scenarios to ensure reliability.
# Failure modes: May converge too early on complex landscapes with low budgets or struggle with scaling in extremely high-dimensional spaces.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from either interface
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.asarray(func.lower), np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb, ub = np.asarray(func.bounds.lb), np.asarray(func.bounds.ub)
        else:
            # Default bounds if not provided, though not expected per requirements
            lb, ub = np.zeros(self.dim), np.ones(self.dim)
        
        # Determine population size based on budget and dimension
        # We want at least some generations, so pop_size shouldn't consume the whole budget
        pop_size = max(4, min(self.budget // 5, 10 * self.dim))
        if pop_size > self.budget:
            pop_size = self.budget
            
        best_x = None
        best_y = float('inf')
        evals = 0
        
        # Handle extremely small budgets with simple random search
        if pop_size < 4:
            while evals < self.budget:
                candidate = lb + (ub - lb) * np.random.rand(self.dim)
                val = func(candidate)
                evals += 1
                if val < best_y:
                    best_y = val
                    best_x = candidate
            return best_x, best_y

        # Initialize population
        pop = lb + (ub - lb) * np.random.rand(pop_size, self.dim)
        fitness = np.full(pop_size, np.inf)
        
        for i in range(pop_size):
            if evals >= self.budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_y:
                best_y = val
                best_x = np.copy(pop[i])
        
        # Parameters for DE
        F = 0.5
        CR = 0.7
        
        # Evolution Loop
        while evals < self.budget:
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Mutation: DE/rand/1
                # Select three unique indices excluding the current index i
                idxs = [j for j in range(pop_size) if j != i]
                choice_idxs = np.random.choice(len(idxs), 3, replace=False)
                r1, r2, r3 = [idxs[idx] for idx in choice_idxs]
                
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) < CR
                # Ensure at least one component is taken from the mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary Handling: Clipping
                trial = np.clip(trial, lb, ub)
                
                # Evaluation and Greedy Selection
                val = func(trial)
                evals += 1
                
                if val < fitness[i]:
                    fitness[i] = val
                    pop[i] = trial
                    if val < best_y:
                        best_y = val
                        best_x = np.copy(trial)
                        
        return best_x, best_y
