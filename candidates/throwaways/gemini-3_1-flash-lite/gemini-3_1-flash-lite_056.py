# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free stochastic local search using a shrinking Gaussian mutation strategy (a (1+1)-Evolution Strategy variant with self-adaptation).
# Search state: Maintains a single current best candidate solution and a dynamic step-size (sigma).
# Candidate generation: Generates a new candidate by applying Gaussian noise N(0, sigma) to the current best solution.
# Selection and replacement: Uses a greedy selection; the new candidate replaces the current best if it yields a lower function value.
# Adaptation: Employs the 1/5th success rule: sigma is increased if the success rate is high and decreased if low.
# Exploration mechanisms: Initial sigma is large (fraction of domain) to allow global search, which decays over time.
# Exploitation mechanisms: Local refinement occurs as sigma shrinks around the current local optimum.
# Boundary handling: Candidates are clipped to the provided search domain bounds.
# Budget strategy: Uniformly consumes the budget across iterations until exhaustion.
# Closest known influences: (1+1)-CMA-ES/Evolution Strategies.
# Novelty or unusual aspects: Minimalist implementation of self-adaptive search for black-box problems in higher dimensions.
# Failure modes: May prematurely converge if the initial domain is massive and the function is highly multimodal; suffers from drift in poorly scaled spaces.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x = np.random.uniform(lb, ub)
        y = func(x)
        remaining = self.budget - 1
        
        # Parameters for (1+1)-ES
        sigma = (ub - lb) * 0.2
        best_x = np.copy(x)
        best_y = y
        
        success_count = 0
        total_steps = 0
        
        while remaining > 0:
            # Generate candidate using Gaussian mutation
            step = np.random.normal(0, sigma)
            candidate = np.clip(best_x + step, lb, ub)
            
            # Evaluate
            f_val = func(candidate)
            remaining -= 1
            total_steps += 1
            
            # Selection
            if f_val < best_y:
                best_y = f_val
                best_x = np.copy(candidate)
                success_count += 1
            
            # Adaptation of sigma (1/5th success rule)
            if total_steps >= 10:
                ratio = success_count / total_steps
                if ratio > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                success_count = 0
                total_steps = 0
                
            # Convergence check: if sigma is negligible, reset to explore
            if np.mean(sigma) < 1e-10:
                sigma = (ub - lb) * 0.1
                
        return best_x, best_y
