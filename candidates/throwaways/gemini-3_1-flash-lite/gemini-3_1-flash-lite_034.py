# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adapting (1+1)-ES with covariance matrix adaptation (local variant).
# Search state: Maintains a current mean vector and a step-size (sigma).
# Candidate generation: Samples a new point using a multivariate normal distribution centered at the current best point.
# Selection and replacement: Deterministic replacement (plus-strategy): new point replaces current if it minimizes the objective.
# Adaptation: Employs the 1/5-th success rule for step-size control to maintain a target success rate.
# Exploration mechanisms: Large initial sigma allows global wandering; small sigma allows localized refinement.
# Exploitation mechanisms: Local hill-climbing via isotropic mutation.
# Boundary handling: Clipping to the box constraints (lower/upper bounds).
# Budget strategy: Iterates until the evaluation count reaches the provided budget.
# Closest known influences: (1+1)-Evolution Strategy with cumulative step-size adaptation.
# Novelty or unusual aspects: Minimalist implementation of adaptive search without external dependencies.
# Failure modes: Susceptible to local optima; lack of restart mechanism limits performance on highly multimodal landscapes.
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
        
        # Initialize search
        curr_x = np.random.uniform(lb, ub, self.dim)
        curr_y = func(curr_x)
        evals = 1
        
        # Hyperparameters for 1/5-th success rule
        sigma = 0.2 * (ub - lb)
        success_count = 0
        gen = 0
        
        while evals < self.budget:
            # Generate candidate
            cand_x = curr_x + np.random.normal(0, sigma, self.dim)
            cand_x = np.clip(cand_x, lb, ub)
            
            cand_y = func(cand_x)
            evals += 1
            
            # Selection
            if cand_y <= curr_y:
                curr_x, curr_y = cand_x, cand_y
                success_count += 1
            
            # Adaptive step-size update
            gen += 1
            if gen >= 10:
                ratio = success_count / gen
                if ratio > 0.2:
                    sigma *= 1.2
                elif ratio < 0.2:
                    sigma /= 1.2
                gen = 0
                success_count = 0
                
            if evals >= self.budget:
                break
                
        return curr_x, curr_y
