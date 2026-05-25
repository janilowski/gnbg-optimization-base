# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (1+1)-ES with adaptive step-size (1/5th rule).
# Search state: Current best point and global mutation scale (sigma).
# Candidate generation: Gaussian mutation of the current best point.
# Selection and replacement: Greedy selection; if mutant is better, replace current.
# Adaptation: Adaptive step-size schema (1/5th success rule) to regulate exploration/exploitation.
# Exploration mechanisms: Gaussian mutation with adaptive sigma.
# Exploitation mechanisms: Greedy descent moves towards successful mutants.
# Boundary handling: Clipping candidates to defined bounds.
# Budget strategy: Exhaustive iterative loop until evaluation budget is reached.
# Closest known influences: Evolutionary Strategies, (1+1)-ES, Rechenberg adaptation.
# Novelty or unusual aspects: Minimalist, self-contained implementation suitable for black-box tasks.
# Failure modes: Local optima trapping; convergence to boundary in ill-conditioned landscapes.
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
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        evals = 1

        # State for adaptation
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        # 1/5th rule parameters
        target_success = 1.0 / 5.0
        c_increase = 1.2
        c_decrease = 0.8
        
        while evals < self.budget:
            # Generate candidate
            z = x + np.random.normal(0, sigma)
            z = np.clip(z, lb, ub)
            
            # Evaluate
            fz = func(z)
            evals += 1
            
            # Selection
            if fz <= y:
                x, y = z, fz
                success_count += 1
            else:
                success_count = 0
            
            # Adapt sigma periodically or after failure/success streaks
            # Every 10 iterations, adjust sigma based on 1/5th rule
            if evals % 10 == 0:
                success_rate = success_count / 10.0
                if success_rate > target_success:
                    sigma *= c_increase
                elif success_rate < target_success:
                    sigma *= c_decrease
                success_count = 0
                
            if evals >= self.budget:
                break
                
        return x, y
