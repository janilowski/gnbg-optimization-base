# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Self-Adaptive Differential Evolution (jDE) implementation. It adjusts scaling factors and crossover rates per individual to navigate various landscapes without manual tuning.
# Search state: A population of candidate vectors, their associated fitness values, and per-individual control parameters (F and Cr).
# Candidate generation: Uses the DE/rand/1 mutation strategy combined with binomial crossover.
# Selection and replacement: Simple greedy one-to-one selection where a trial vector replaces its parent only if its fitness is better or equal.
# Adaptation: Employs the jDE mechanism where F and Cr parameters have a small probability (0.1) of being randomized in each generation, allowing successful parameters to propagate through survival.
# Exploration mechanisms: Driven by the stochastic nature of the rand/1 mutation and the maintenance of a diverse population.
# Exploitation mechanisms: Greedy selection ensures the population moves towards local or global minima over time.
# Boundary handling: Trial vectors are clipped to the hyper-cube defined by the problem bounds.
# Budget strategy: Monitored at every function evaluation. The algorithm stops immediately and returns the best found solution when the budget is exhausted.
# Closest known influences: jDE (Brest et al., 2006).
# Novelty or unusual aspects: Standard robust jDE implementation optimized for a single-class Python module structure.
# Failure modes: May struggle with extremely high-dimensional problems where the budget is insufficient to allow the population to converge or if the landscape is highly discontinuous with very narrow optima.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the Differential Evolution algorithm.
        
        Args:
            budget (int): Maximum number of allowed function evaluations.
            dim (int): Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        
        # Population size heuristic: between 10 and 100 based on dimension.
        # Adjusted if the budget is very small.
        self.pop_size = max(10, min(100, int(10 * dim)))
        if self.pop_size * 2 > budget:
            self.pop_size = max(4, budget // 10)

    def __call__(self, func):
        """
        Executes the minimization search.
        
        Args:
            func (callable): Objective function to minimize.
            
        Returns:
            tuple: (best_x, best_y)
        """
        # Extract bounds from the function object
        lb, ub = self._get_bounds(func)
        
        # Internal state
        eval_count = 0
        
        # Initialize population and parameters
        # pop: (pop_size, dim)
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        
        # F: scaling factor, Cr: crossover rate (jDE strategy)
        F = np.full(self.pop_size, 0.5)
        Cr = np.full(self.pop_size, 0.9)
        
        fitness = np.zeros(self.pop_size)
        
        # Initial evaluation
        for i in range(self.pop_size):
            if eval_count >= self.budget:
                # If budget is extremely low, return best seen so far
                idx = np.argmin(fitness[:max(1, i)])
                return pop[idx], fitness[idx]
            
            fitness[i] = func(pop[i])
            eval_count += 1
            
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]
        
        # Evolutionary loop
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                
                # 1. Parameter Adaptation (jDE)
                # Randomize F and Cr with a small probability
                curr_f = F[i]
                curr_cr = Cr[i]
                if np.random.rand() < 0.1:
                    curr_f = 0.1 + np.random.rand() * 0.9
                if np.random.rand() < 0.1:
                    curr_cr = np.random.rand()
                
                # 2. Mutation (DE/rand/1)
                # Select 3 distinct indices different from i
                indices = [idx for idx in range(self.pop_size) if idx != i]
                abc_idx = np.random.choice(indices, 3, replace=False)
                a, b, c = pop[abc_idx]
                
                mutant = a + curr_f * (b - c)
                
                # 3. Crossover (Binomial)
                cross_points = np.random.rand(self.dim) < curr_cr
                # Ensure at least one dimension is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # 4. Boundary Handling
                trial = np.clip(trial, lb, ub)
                
                # 5. Selection
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    F[i] = curr_f
                    Cr[i] = curr_cr
                    
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_y

    def _get_bounds(self, func):
        """
        Retrieves lower and upper bounds from the objective function.
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            return np.array(func.lower), np.array(func.upper)
        if hasattr(func, 'bounds'):
            # Handling both potential attribute styles for bounds
            lb = getattr(func.bounds, 'lb', None)
            ub = getattr(func.bounds, 'ub', None)
            if lb is not None and ub is not None:
                return np.array(lb), np.array(ub)
        
        # Fallback to a standard range if bounds are not detectable
        return np.full(self.dim, -5.0), np.full(self.dim, 5.0)
