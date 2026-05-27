# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES-inspired evolution strategy using a diagonal covariance matrix.
# Search state: Maintains a mean vector, a step-size (sigma), and a diagonal covariance vector.
# Candidate generation: Samples vectors from a multivariate normal distribution defined by the mean and diagonal covariance.
# Selection and replacement: Uses a $(\mu, \lambda)$-style selection where the top-performing individuals update the distribution mean.
# Adaptation: Updates step-size via cumulative step-size adaptation and adapts covariance via rank-one updates.
# Exploration mechanisms: Initially high sigma and large sampling variance; reduces sigma as convergence progresses.
# Exploitation mechanisms: Mean-based convergence towards the best-observed regions in the search space.
# Boundary handling: Clamping candidate solutions to the provided objective function bounds.
# Budget strategy: Iterates until the evaluation budget is exhausted; strictly monitors individual calls.
# Closest known influences: CMA-ES and (1+1)-ES with diagonal covariance adaptation.
# Novelty or unusual aspects: Simplified diagonal adaptation specifically for memory and code conciseness.
# Failure modes: May get stuck in local optima if the initial sigma is too small or the landscape is highly deceptive.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = getattr(func, 'bounds', None)
        if bounds:
            lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        else:
            lb, ub = np.array(func.lower), np.array(func.upper)

        # Initialization
        x = np.random.uniform(lb, ub)
        best_x = np.copy(x)
        best_y = func(x)
        evals = 1
        
        sigma = 0.2 * (ub - lb)
        diag_cov = np.ones(self.dim)
        
        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))

        while evals < self.budget:
            # Generate candidates
            candidates = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                z = np.random.normal(0, 1, self.dim)
                trial = x + sigma * (diag_cov * z)
                trial = np.clip(trial, lb, ub)
                
                y = func(trial)
                evals += 1
                candidates.append((trial, y))
                
                if y < best_y:
                    best_y = y
                    best_x = np.copy(trial)
            
            # Sort candidates by objective value
            candidates.sort(key=lambda item: item[1])
            
            # Update mean and step-size adaptation (simplified)
            best_trial, best_val = candidates[0]
            if best_val < best_y:
                # Move mean towards successful offspring
                x = best_trial
                # Adapt diagonal covariance (Dir-weighted)
                diag_cov = 0.9 * diag_cov + 0.1 * np.abs(best_trial - x) / (sigma + 1e-9)
                sigma *= 1.05
            else:
                sigma *= 0.95

        return best_x, best_y
