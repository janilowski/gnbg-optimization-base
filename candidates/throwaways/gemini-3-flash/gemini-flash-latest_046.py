# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Robust Differential Evolution (DE/best/1/bin strategy) with population scaling.
# Search state: Population array of shape (pop_size, dim), fitness scores array, and the global best (x, y).
# Candidate generation: Generates trials using the 'best' individual as mutation base, adds scaled difference of two random members, and performs binomial crossover.
# Selection and replacement: Simple one-to-one greedy replacement where a child replaces its parent if its fitness is strictly better.
# Adaptation: Population size is dynamically scaled based on dimensionality and available budget to ensure at least some generations occur.
# Exploration mechanisms: Use of random population members in mutation and crossover to maintain diversity.
# Exploitation mechanisms: The 'best' individual drives the mutation, pulling the population towards currently known optima.
# Boundary handling: Hard clipping of candidate vectors to the search space bounds.
# Budget strategy: Iterative loop that terminates immediately when the evaluation count reaches the provided budget.
# Closest known influences: Storn and Price (1997) Differential Evolution.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness across varying dimensionality and budget constraints without external dependencies.
# Failure modes: On extremely rugged or high-dimensional landscapes with very small budgets, it might not converge significantly; on flat landscapes, it may lose diversity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    A robust Differential Evolution (DE) implementation designed for black-box minimization.
    It adapts to the problem dimensions and available budget.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.n_evals = 0
        
        # Hyperparameters for DE
        self.F = 0.8   # Scaling factor for mutation
        self.CR = 0.9  # Crossover probability
        
        # Adjust population size based on dimension and budget
        # We want at least some generations, so pop_size shouldn't be too large relative to budget.
        suggested_pop = 10 * self.dim
        self.pop_size = max(5, min(suggested_pop, 50, self.budget // 4))

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback if no bounds are provided (though benchmark usually provides them)
            lb = np.full(self.dim, -10.0)
            ub = np.full(self.dim, 10.0)

        # Initialize tracking
        best_x = None
        best_y = float('inf')

        def evaluate(x):
            nonlocal best_x, best_y
            if self.n_evals >= self.budget:
                return best_y  # Should not be reached if logic is correct
            
            y = func(x)
            self.n_evals += 1
            
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Initial Population: Uniform random sampling within bounds
        population = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.zeros(self.pop_size)

        for i in range(self.pop_size):
            if self.n_evals < self.budget:
                fitness[i] = evaluate(population[i])
            else:
                # If budget exhausted during initialization, return what we have
                return best_x, best_y

        # DE Evolution Loop
        while self.n_evals < self.budget:
            for i in range(self.pop_size):
                if self.n_evals >= self.budget:
                    break
                
                # 1. Mutation: DE/best/1
                # Find best individual in current population for exploitation
                best_idx = np.argmin(fitness)
                
                # Pick two random distinct indices other than i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                
                mutant = population[best_idx] + self.F * (population[r1] - population[r2])
                
                # 2. Crossover: Binomial
                cross_points = np.random.rand(self.dim) < self.CR
                # Ensure at least one dimension is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # 3. Boundary Handling: Clipping
                trial = np.clip(trial, lb, ub)
                
                # 4. Selection: Greedy replacement
                f_trial = evaluate(trial)
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

        return best_x, best_y
