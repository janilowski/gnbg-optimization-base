# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust Differential Evolution (DE) implementation using the rand/1/bin strategy.
# Search state: A population of candidate solutions and their corresponding objective values.
# Candidate generation: Mutant vectors are created by adding a scaled difference between two random population members to a third random member. Trial vectors are then formed via binomial crossover with the current parent.
# Selection and replacement: A one-to-one replacement strategy where the trial vector replaces the parent only if its objective value is lower or equal.
# Adaptation: The mutation factor (F) and crossover probability (Cr) are held constant at values generally considered robust (0.8 and 0.9 respectively).
# Exploration mechanisms: Random mutation vectors and a relatively high crossover probability ensure global search capability.
# Exploitation mechanisms: Selection pressure ensures that only improvements are kept, gradually converging the population towards optima.
# Boundary handling: Simple clipping of candidate vectors to the problem's lower and upper bounds.
# Budget strategy: The population size is dynamically determined based on the total budget and dimensionality. The algorithm terminates and returns the best solution as soon as the evaluation budget is exhausted.
# Closest known influences: Storn and Price's original Differential Evolution algorithm.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness and strict adherence to budget constraints without external dependencies beyond NumPy.
# Failure modes: May converge prematurely on highly multi-modal functions if the population size is too small, or may not converge sufficiently if the budget is very low relative to the dimension.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the Differential Evolution algorithm.
        
        Args:
            budget (int): Total number of function evaluations allowed.
            dim (int): Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        
        # Heuristic for population size: balance between exploration and generation count.
        # Ensure pop_size is at least 4 for DE/rand/1 and doesn't exceed budget.
        self.pop_size = max(5, min(10 * dim, budget // 2, 100))
        
        # DE parameters
        self.F = 0.8  # Mutation scale factor
        self.Cr = 0.9 # Crossover probability
        
    def __call__(self, func):
        """
        Executes the optimization process.
        
        Args:
            func: The objective function to minimize.
        
        Returns:
            tuple: (best_x, best_y) representing the best solution found.
        """
        # Resolve bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, 'bounds'):
            lb = func.bounds.lb
            ub = func.bounds.ub
        else:
            # Fallback to standard hypercube if bounds are missing
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)

        # Ensure bounds are numpy arrays
        lb = np.asarray(lb)
        ub = np.asarray(ub)

        eval_count = 0
        
        # Initialize population
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        fitness = np.zeros(self.pop_size)
        
        best_x = None
        best_y = float('inf')

        # Initial evaluation of the population
        for i in range(self.pop_size):
            if eval_count >= self.budget:
                break
            val = func(pop[i])
            fitness[i] = val
            eval_count += 1
            
            if val < best_y:
                best_y = val
                best_x = pop[i].copy()

        # Evolution loop
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                
                # Mutation: rand/1 strategy
                # Pick 3 indices different from i and each other
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                
                # Boundary handling: clipping
                mutant = np.clip(mutant, lb, ub)
                
                # Crossover: binomial
                cross_points = np.random.rand(self.dim) <= self.Cr
                # Ensure at least one dimension is inherited from mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                val = func(trial)
                eval_count += 1
                
                if val <= fitness[i]:
                    fitness[i] = val
                    pop[i] = trial
                    
                    if val < best_y:
                        best_y = val
                        best_x = trial.copy()

        return best_x, best_y

# The harness provides the seed for reproducibility via numpy.random.seed.
# The Algorithm class adheres to the minimization objective and budget constraints.
