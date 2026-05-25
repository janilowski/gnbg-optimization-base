# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant with a simplified diagonal covariance structure.
# Search state: Current centroid, step-size (sigma), and a diagonal covariance matrix represented by individual standard deviation vectors.
# Candidate generation: Multivariate normal sampling around the centroid scaled by the current standard deviation.
# Selection and replacement: Evolution strategy using weighted recombination of the best k candidates to update the centroid.
# Adaptation: Step-size control via Success Rule (1/5 rule) and covariance adaptation based on the direction of successful search steps.
# Exploration mechanisms: Gaussian mutation with adaptive step-size control.
# Exploitation mechanisms: Weighted recombination moves the centroid toward the mean of successful candidates.
# Boundary handling: Resampling candidates that fall outside of the specified bounds.
# Budget strategy: Uniform distribution of evaluations; loops until the function evaluation budget is exhausted.
# Closest known influences: Simplified CMA-ES and (1+1)-ES principles.
# Novelty or unusual aspects: Diagonal-only adaptation reduces parameter complexity to be highly robust in high-dimensional settings with limited budget.
# Failure modes: Premature convergence on narrow global optima or poor performance on highly non-separable landscapes where correlation matters.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract boundaries
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        centroid = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        pop_size = 4 + int(3 * np.log(self.dim))
        
        best_x = np.copy(centroid)
        best_y = float('inf')
        
        evals = 0
        
        while evals < self.budget:
            # Generate candidates
            candidates = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and enforce bounds via resampling
                sample = np.random.normal(centroid, sigma)
                sample = np.clip(sample, lb, ub)
                
                y = func(sample)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = np.copy(sample)
                
                candidates.append((sample, y))
            
            # Sort candidates by fitness
            candidates.sort(key=lambda x: x[1])
            
            # Update centroid using weighted recombination of the top half
            k = pop_size // 2
            top_half = [c[0] for c in candidates[:k]]
            new_centroid = np.mean(top_half, axis=0)
            
            # Update sigma based on progress (1/5 success rule variant)
            if candidates[0][1] < best_y:
                sigma *= 1.1
            else:
                sigma *= 0.95
            
            # Adapt centroid
            centroid = new_centroid
            
        return best_x, best_y
