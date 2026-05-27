# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free covariance matrix adaptation evolution strategy (CMA-ES) variant.
# Search state: Maintains a mean vector, a step-size (sigma), and a simplified isotropic covariance matrix.
# Candidate generation: Samples individuals from a multivariate normal distribution centered at the mean.
# Selection and replacement: Uses rank-based selection (mu-lambda evolution strategy) to update the mean.
# Adaptation: Updates the mean toward the best-performing individuals; sigma is kept constant for simplicity.
# Exploration mechanisms: Gaussian noise controlled by a scaling factor sigma.
# Exploitation mechanisms: The mean vector gradually shifts towards discovered optima.
# Boundary handling: Clamping candidate solutions to the specified hyper-rectangular bounds.
# Budget strategy: Iterative population generation until remaining evaluations are exhausted.
# Closest known influences: Simplified (1+1)-ES and basic CMA-ES principles.
# Novelty or unusual aspects: Extremely lightweight implementation optimized for low-overhead black-box tasks.
# Failure modes: Can get stuck in local optima on highly multi-modal landscapes with limited budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        n_pop = 4 + int(3 * np.log(self.dim))
        mu = n_pop // 2
        sigma = 0.3 * (ub - lb)
        x_mean = np.random.uniform(lb, ub)
        
        best_x = np.copy(x_mean)
        best_y = float('inf')
        
        # Main optimization loop
        while self.eval_count < self.budget:
            # Generate population
            pop_x = []
            pop_y = []
            
            for _ in range(n_pop):
                if self.eval_count >= self.budget:
                    break
                
                # Sample and clamp
                x = np.clip(x_mean + np.random.normal(0, sigma, self.dim), lb, ub)
                y = func(x)
                self.eval_count += 1
                
                pop_x.append(x)
                pop_y.append(y)
                
                # Global tracker
                if y < best_y:
                    best_y = y
                    best_x = np.copy(x)
            
            if not pop_x:
                break
                
            # Selection: Sort by performance
            indices = np.argsort(pop_y)
            
            # Update mean using top mu individuals
            elite_x = np.array([pop_x[i] for i in indices[:mu]])
            x_mean = np.mean(elite_x, axis=0)
            
            # Decay sigma slightly to refine search
            sigma *= 0.995
            
        return best_x, best_y
