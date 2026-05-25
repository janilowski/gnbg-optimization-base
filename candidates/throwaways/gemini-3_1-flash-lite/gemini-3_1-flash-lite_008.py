# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the $(1+1)$-ES algorithm with self-adaptive step size.
# Search state: Stores the current best point (x) and a global step size (sigma).
# Candidate generation: Generates a new candidate by adding Gaussian noise (sigma * N(0, 1)) to the current best.
# Selection and replacement: The candidate replaces the current best if it achieves a lower function value.
# Adaptation: Employs the 1/5 success rule: if the success rate is high, increase step size (expand); otherwise, decrease it (contract).
# Exploration mechanisms: Gaussian mutation guided by the step size sigma.
# Exploitation mechanisms: Local hill-climbing via elitist replacement.
# Boundary handling: Candidates are clipped to the domain bounds after mutation.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: Rechenberg’s (1+1)-Evolution Strategy.
# Novelty or unusual aspects: Compact and efficient for low-to-medium dimensional continuous optimization.
# Failure modes: Can get stuck in local optima; performance degrades on high-dimensional or rugged landscapes.
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
        
        # Initial point within bounds
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        
        # Initialize search parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        evals = 1
        
        # For adaptive step size logic
        history_size = 10
        success_history = []
        
        while evals < self.budget:
            # Generate candidate
            z = np.random.normal(0, 1, self.dim)
            x_new = np.clip(x + sigma * z, lb, ub)
            
            y_new = func(x_new)
            evals += 1
            
            # Selection
            if y_new <= y:
                x, y = x_new, y_new
                success_history.append(1)
            else:
                success_history.append(0)
            
            # Adaptation logic (1/5 Success Rule)
            if len(success_history) >= history_size:
                success_rate = sum(success_history[-history_size:]) / history_size
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma /= 1.1
                # Optional: limit sigma
                sigma = np.clip(sigma, 1e-6 * (ub - lb), (ub - lb))
        
        return x, y
