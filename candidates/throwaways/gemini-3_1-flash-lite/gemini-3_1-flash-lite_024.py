# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the (1+1)-CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for derivative-free optimization.
# Search state: Maintains a mean vector (m), a step-size (sigma), and an identity-based covariance evolution (C), though simplified to isotropic step-size adaptation here.
# Candidate generation: Samples Gaussian noise around the current mean scaled by the step-size.
# Selection and replacement: Deterministic (1+1) selection; the candidate replaces the mean if it yields a lower functional value.
# Adaptation: Uses the success rule: increases sigma if the candidate succeeds, decreases if it fails.
# Exploration mechanisms: Gaussian mutation provides global exploration capability influenced by the adaptive sigma.
# Exploitation mechanisms: Local search centered around the best-found solution (mean vector).
# Boundary handling: Projects candidates back into the valid search space using clipping.
# Budget strategy: Exhaustive usage; terminates exactly when the evaluation count reaches the budget.
# Closest known influences: (1+1)-ES with two-thirds success rule adaptation.
# Novelty or unusual aspects: Compact implementation focusing on robustness in low to moderate dimensions.
# Failure modes: Can get trapped in local optima; performance degrades in extremely high dimensions or highly non-smooth landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        x = np.random.uniform(lb, ub)
        y = func(x)
        self.evals += 1
        
        # Strategy parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        while self.evals < self.budget:
            # Generate candidate
            z = np.random.normal(0, 1, self.dim)
            x_candidate = np.clip(x + sigma * z, lb, ub)
            
            # Evaluate
            y_candidate = func(x_candidate)
            self.evals += 1
            
            # (1+1) Selection
            if y_candidate <= y:
                x = x_candidate
                y = y_candidate
                success_count += 1
            else:
                success_count = 0
            
            # Sigmas adaptation (1/5 rule variant)
            if success_count >= 1:
                sigma *= 1.2
            else:
                sigma /= 1.2
            
            # Ensure sigma doesn't collapse
            sigma = np.clip(sigma, 1e-9 * (ub - lb), 1.0 * (ub - lb))
            
            if self.evals >= self.budget:
                break
                
        return x, y
