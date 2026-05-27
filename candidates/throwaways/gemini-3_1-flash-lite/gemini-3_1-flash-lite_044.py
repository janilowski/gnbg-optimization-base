# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (A)daptive (S)tep size (SAS).
# Search state: Maintains the current best point found so far as the anchor.
# Candidate generation: Generates candidates by adding Gaussian noise to the current best.
# Selection and replacement: Greedy selection: replace best if a candidate improves the objective.
# Adaptation: The standard deviation of the noise (step size) shrinks or expands based on local success rates (1/5th rule-like).
# Exploration mechanisms: Initial global search is wide; radius tightens as progress slows.
# Exploitation mechanisms: Local refinement via Gaussian sampling around the best known point.
# Boundary handling: Clipping candidates to the feasible region defined by the function.
# Budget strategy: Iterative loop that terminates exactly when the budget is exhausted.
# Closest known influences: Adaptive Random Search and simple (1+1)-ES.
# Novelty or unusual aspects: Minimalist implementation of adaptive step size control.
# Failure modes: Can get trapped in local optima; performance depends on the initial step size.
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
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        evals = 1
        
        # Initial step size (10% of the range)
        sigma = 0.1 * (ub - lb)
        
        # Adaptation parameters
        success_count = 0
        
        while evals < self.budget:
            # Generate candidate
            candidate = best_x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            y = func(candidate)
            evals += 1
            
            # Simple greedy selection
            if y < best_y:
                best_x = candidate
                best_y = y
                success_count += 1
            else:
                success_count -= 0.25 # Slowly penalize failures
                
            # Adaptive step size logic: expand if successful, shrink if failing
            if success_count > 5:
                sigma *= 1.2
                success_count = 0
            elif success_count < -5:
                sigma *= 0.8
                success_count = 0
                
            # Emergency stop for budget
            if evals >= self.budget:
                break
                
        return best_x, best_y
