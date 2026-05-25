# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant (simplified).
# Search state: Maintains a Gaussian distribution defined by a mean vector and a covariance matrix (simplified as a diagonal standard deviation vector for efficiency).
# Candidate generation: Samples candidate solutions from a multivariate normal distribution centered on the current mean.
# Selection and replacement: Picks the top-performing fraction of samples to update the mean.
# Adaptation: Updates the mean using a weighted average of successful candidates and increases/decreases step-size based on success (1/5th rule).
# Exploration mechanisms: Gaussian noise controlled by a global step-size (sigma).
# Exploitation mechanisms: Mean shifting toward successful regions of the search space.
# Boundary handling: Clipping candidates to the defined box constraints.
# Budget strategy: Exhausts the entire provided budget by strictly tracking calls.
# Closest known influences: (1+1)-ES and simple Adaptive Step-size Random Search.
# Novelty or unusual aspects: Minimalist implementation of adaptive step-size control without full covariance matrix storage.
# Failure modes: Can get trapped in local optima in highly multi-modal landscapes or diverge if the step-size adaptation stalls.
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
        x = np.random.uniform(lb, ub, self.dim)
        best_x = np.copy(x)
        best_y = func(x)
        
        sigma = 0.2 * (ub - lb)
        evals = 1
        
        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        
        while evals < self.budget:
            # Generate offspring
            candidates = []
            for _ in range(pop_size):
                if evals >= self.budget: break
                
                # Sample and clip
                trial = np.clip(x + np.random.normal(0, sigma), lb, ub)
                val = func(trial)
                evals += 1
                
                candidates.append((val, trial))
                
                if val < best_y:
                    best_y = val
                    best_x = np.copy(trial)
            
            # Sort candidates by fitness
            candidates.sort(key=lambda item: item[0])
            
            # Selection: Use top half to update mean
            num_keep = max(1, len(candidates) // 2)
            successful_x = [c[1] for c in candidates[:num_keep]]
            
            new_mean = np.mean(successful_x, axis=0)
            
            # Adaptation: Move mean and adjust sigma (1/5th success rule)
            if candidates[0][0] < best_y:
                # Success: shift mean and increase step
                x = new_mean
                sigma *= 1.2
            else:
                # Failure: decrease step
                sigma *= 0.8
                
        return best_x, best_y
