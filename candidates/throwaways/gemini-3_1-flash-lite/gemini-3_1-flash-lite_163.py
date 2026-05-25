# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple Population-based Incremental Learning (PBIL) inspired stochastic solver.
# Search state: Maintains a probability distribution vector (mean) that shifts toward successful samples.
# Candidate generation: Generates samples using a multivariate normal distribution centered on the current mean.
# Selection and replacement: Keeps the best sample found so far; updates the mean towards the best sample found in the current iteration.
# Adaptation: Employs a learning rate and decaying standard deviation (sigma) to transition from global exploration to local exploitation.
# Exploration mechanisms: Initial large sigma provides wide search space coverage.
# Exploitation mechanisms: Mean-shifting and narrowing sigma focuses the search on the vicinity of the best individual.
# Boundary handling: Clamps candidates to the function-defined hypercube bounds.
# Budget strategy: Divides total budget into fixed-size generations, processing until the limit is reached.
# Closest known influences: Evolutionary Strategies, PBIL.
# Novelty or unusual aspects: Uses a simple adaptive cooling scheme for the step size.
# Failure modes: Can get stuck in local minima if the initial spread is too small or cooling is too aggressive.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        else:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)

        # Initialization
        pop_size = min(20, self.budget // 10)
        if pop_size < 4: pop_size = 4
        
        # State: track best result and current adaptive mean
        best_x = None
        best_y = float('inf')
        
        # Start mean at the center of the domain
        mean = (lb + ub) / 2.0
        # Initial sigma covers the domain
        sigma = (ub - lb) / 4.0
        
        evals = 0
        lr = 0.1  # Learning rate for the mean
        
        while evals < self.budget:
            # Generate candidates
            batch_size = min(pop_size, self.budget - evals)
            candidates = np.random.normal(mean, sigma, (batch_size, self.dim))
            
            # Boundary handling: Clamp
            candidates = np.clip(candidates, lb, ub)
            
            # Evaluate
            current_best_in_batch = None
            current_best_val = float('inf')
            
            for x in candidates:
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                
                if y < current_best_val:
                    current_best_val = y
                    current_best_in_batch = x.copy()
            
            # Adaptation: Move mean towards best performer of the batch
            if current_best_in_batch is not None:
                mean = mean * (1 - lr) + current_best_in_batch * lr
            
            # Cooling: Gradually reduce sigma to exploit
            sigma *= 0.95
            if np.all(sigma < 1e-7):
                sigma = (ub - lb) * 0.01

        return best_x, best_y
