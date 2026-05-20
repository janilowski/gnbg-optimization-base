# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adaptive Differential Evolution (jDE variant) designed for robust black-box minimization.
# Search state: A population of candidate solution vectors, their associated objective values, and individual control parameters (F, CR).
# Candidate generation: Offspring are created using DE/rand/1 mutation (three random distinct parents) followed by binomial crossover.
# Selection and replacement: A greedy one-to-one replacement strategy where an offspring replaces its parent if its objective value is less than or equal to the parent's.
# Adaptation: The mutation scale factor (F) and crossover probability (CR) are adapted for each individual. With a probability of 0.1, these parameters are re-randomized, allowing the algorithm to automatically find effective settings for different landscapes.
# Exploration mechanisms: Initialized with uniform random sampling across the search space and maintained via the stochastic nature of the mutation vectors.
# Exploitation mechanisms: Selection pressure preserves better solutions, and successful control parameters are propagated through the population.
# Boundary handling: Offspring coordinates are clipped to the specified lower and upper bounds.
# Budget strategy: A strict evaluation counter ensures the search terminates as soon as the provided budget is exhausted.
# Closest known influences: Brest et al.'s jDE algorithm for self-adaptive control parameters in Differential Evolution.
# Novelty or unusual aspects: Compact logic combining parameter adaptation and budget-aware execution in a single loop.
# Failure modes: High-dimensional, extremely needle-in-a-haystack landscapes may lead to premature convergence or stagnation if the population size is too small for the complexity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    A Self-Adaptive Differential Evolution (jDE) implementation.
    Optimized for robustness across various dimensions and budgets.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Adjust population size based on dimensionality, clamped between 10 and 100.
        self.pop_size = max(10, min(100, 5 * dim))
        
    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        else:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
            
        # Initialize population and parameters
        # pop: [pop_size, dim]
        # F: Scale factor, CR: Crossover rate
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        F = np.full(self.pop_size, 0.5)
        CR = np.full(self.pop_size, 0.9)
        
        # Evaluate initial population
        scores = np.zeros(self.pop_size)
        eval_count = 0
        best_y = float('inf')
        best_x = None
        
        for i in range(self.pop_size):
            if eval_count >= self.budget:
                break
            scores[i] = func(pop[i])
            eval_count += 1
            if scores[i] < best_y:
                best_y = scores[i]
                best_x = pop[i].copy()
                
        # Main DE loop
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                
                # Self-adaptation of parameters F and CR
                curr_F = F[i]
                curr_CR = CR[i]
                if np.random.rand() < 0.1:
                    curr_F = 0.1 + np.random.rand() * 0.9
                if np.random.rand() < 0.1:
                    curr_CR = np.random.rand()
                
                # Mutation: DE/rand/1
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + curr_F * (b - c)
                
                # Crossover: Binomial
                cross_mask = np.random.rand(self.dim) < curr_CR
                # Ensure at least one component is changed
                cross_mask[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_mask, mutant, pop[i])
                
                # Boundary handling: Clipping
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                score_trial = func(trial)
                eval_count += 1
                
                # Selection
                if score_trial <= scores[i]:
                    pop[i] = trial
                    scores[i] = score_trial
                    F[i] = curr_F
                    CR[i] = curr_CR
                    
                    if score_trial < best_y:
                        best_y = score_trial
                        best_x = trial.copy()
                        
        return best_x, best_y

# The harness provides the seed for reproducibility. 
# Ensure the evaluation budget is never exceeded.
# Standard library and numpy are the only dependencies.
# The algorithm is self-contained within the Algorithm class.
