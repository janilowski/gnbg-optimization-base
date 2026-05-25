# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A pure-numpy implementation of a Random Search with local Gaussian mutation (Evolutionary Strategy).
# Search state: Tracks the current 'best_x' and 'best_y' found throughout iterations.
# Candidate generation: Generates new candidates by applying Gaussian noise to the current best_x, with a shrinking step-size.
# Selection and replacement: Greedy selection; updates best_x if func(x) results in a lower objective value.
# Adaptation: The mutation scale (sigma) decays linearly over the budget to transition from exploration to fine-grained exploitation.
# Exploration mechanisms: Initial large-scale random sampling within bounds, followed by Gaussian perturbations.
# Exploitation mechanisms: Local hill-climbing using decreasing step sizes around the current best candidate.
# Boundary handling: Clamping candidates to the provided bounds using numpy clip.
# Budget strategy: Divides budget into an initial uniform sample phase and a subsequent iterative local search phase.
# Closest known influences: Simple (1+1)-ES or Adaptive Random Search.
# Novelty or unusual aspects: Minimalist implementation prioritizing robustness for black-box functions without gradient access.
# Failure modes: Susceptible to getting trapped in narrow local optima if the initial global exploration misses the basin of attraction.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        range_span = ub - lb
        
        # Initialize
        best_x = None
        best_y = float('inf')
        
        # Budget allocation: 20% global exploration, 80% local refinement
        global_budget = int(self.budget * 0.2)
        local_budget = self.budget - global_budget
        
        # Global exploration phase
        for _ in range(global_budget):
            x = np.random.uniform(lb, ub)
            y = func(x)
            if y < best_y:
                best_y = y
                best_x = x
        
        # Local refinement phase (Evolutionary Strategy)
        # Sigma starts at 1/4 of range and decays to avoid stagnation
        initial_sigma = 0.25 * range_span
        
        for i in range(local_budget):
            # Linearly decay mutation intensity
            progress = i / local_budget
            sigma = initial_sigma * (1.0 - progress)
            
            # Mutate current best with clipping
            mutation = np.random.normal(0, sigma, self.dim)
            x_new = np.clip(best_x + mutation, lb, ub)
            
            y_new = func(x_new)
            
            # Greedy replacement
            if y_new < best_y:
                best_y = y_new
                best_x = x_new
                
        return best_x, best_y
