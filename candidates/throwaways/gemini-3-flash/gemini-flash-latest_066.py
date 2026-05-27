# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Self-Adaptive Differential Evolution (jDE) implementation. 
# It adapts the mutation scale (F) and crossover rate (CR) for each individual 
# in the population during the search process.
# Search state: A population of candidate solutions, their objective values, 
# and their specific F and CR parameters.
# Candidate generation: Uses the DE/rand/1 mutation strategy (base vector plus 
# scaled difference of two random vectors) followed by binomial crossover.
# Selection and replacement: Simple greedy selection; a trial vector replaces 
# its parent only if its objective value is lower or equal.
# Adaptation: jDE logic where control parameters F and CR have a probability 
# (tau) of being randomly reset to new values, allowing successful parameters 
# to persist through survivors.
# Exploration mechanisms: Differential mutation provides search directions 
# derived from the current population distribution.
# Exploitation mechanisms: Greedy selection and adaptation toward parameters 
# that yield improvements facilitate local refinement.
# Boundary handling: Mutant vectors are clipped to the hypercube defined by 
# the problem's lower and upper bounds.
# Budget strategy: The algorithm tracks evaluations and terminates immediately 
# once the budget is exhausted, even within a generation.
# Closest known influences: "Self-Adapting Control Parameters in Differential 
# Evolution" by Brest et al. (2006).
# Novelty or unusual aspects: Lightweight and compact implementation using only 
# numpy, designed for robustness across various dimensions.
# Failure modes: Like most DE variants, it may struggle with highly 
# non-separable functions or extremely limited budgets in high dimensions 
# where population-based search is slow to start.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the Differential Evolution algorithm.
        
        Args:
            budget: Maximum number of function evaluations.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        
        # Adaptive population size logic
        # Enough members for DE/rand/1 (min 4), scaled by dim, capped by budget
        self.pop_size = max(5, min(budget // 4, 10 * dim))
        if self.pop_size > 100:
            self.pop_size = 100
            
        # jDE Hyperparameters (standard defaults)
        self.tau1 = 0.1 # Probability to update F
        self.tau2 = 0.1 # Probability to update CR
        self.fl = 0.1   # Lower bound for F
        self.fu = 0.9   # Upper bound for F range

    def __call__(self, func):
        """
        Executes the optimization process.
        """
        # 1. Extract bounds from the function object
        try:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb)
                ub = np.asarray(func.bounds.ub)
            except AttributeError:
                # Fallback to a standard range if no bounds are found
                lb = np.full(self.dim, -5.0)
                ub = np.full(self.dim, 5.0)

        # 2. Initialization
        eval_count = 0
        best_x = None
        best_y = np.inf
        
        # Population state
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        scores = np.full(self.pop_size, np.inf)
        
        # Individual-specific control parameters
        F = np.full(self.pop_size, 0.5)
        CR = np.full(self.pop_size, 0.9)

        # Initial evaluation of the population
        for i in range(self.pop_size):
            if eval_count >= self.budget:
                break
            
            val = func(pop[i])
            eval_count += 1
            scores[i] = val
            
            if val < best_y:
                best_y = val
                best_x = pop[i].copy()
        
        # Fallback for zero or tiny budget
        if best_x is None:
            best_x = lb + (ub - lb) * np.random.rand(self.dim)

        # 3. Optimization Loop
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break

                # Parameter Adaptation (jDE)
                current_f = F[i]
                current_cr = CR[i]
                if np.random.rand() < self.tau1:
                    current_f = self.fl + np.random.rand() * self.fu
                if np.random.rand() < self.tau2:
                    current_cr = np.random.rand()

                # Differential Mutation (rand/1)
                # Select 3 distinct individuals from population excluding current index i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                abc_indices = np.random.choice(candidates, 3, replace=False)
                a, b, c = pop[abc_indices]
                
                mutant = a + current_f * (b - c)
                
                # Boundary Handling (Clipping)
                mutant = np.clip(mutant, lb, ub)

                # Binomial Crossover
                cross_points = np.random.rand(self.dim) < current_cr
                # Ensure at least one dimension is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_score = func(trial)
                eval_count += 1

                if trial_score <= scores[i]:
                    pop[i] = trial
                    scores[i] = trial_score
                    F[i] = current_f
                    CR[i] = current_cr
                    
                    if trial_score < best_y:
                        best_y = trial_score
                        best_x = trial.copy()

        return best_x, best_y
