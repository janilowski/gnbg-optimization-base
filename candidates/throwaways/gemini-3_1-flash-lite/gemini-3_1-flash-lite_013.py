# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES-inspired evolution strategy using rank-based selection and covariance adaptation.
# Search state: Maintains a mean vector, a covariance matrix (diagonal approximation), and a step-size (sigma).
# Candidate generation: Samples points from a multivariate normal distribution centered at the mean.
# Selection and replacement: Uses rank-based fitness selection to update the mean via a weighted average of top candidates.
# Adaptation: Updates step-size using success rate (1/5th rule) and diagonal covariance matrix via a running variance of successful candidates.
# Exploration mechanisms: Adaptive exploration provided by the step-size and the sampling distribution.
# Exploitation mechanisms: The mean consistently shifts toward the best observed regions.
# Boundary handling: Projects samples back into valid bounds using a clipping method.
# Budget strategy: Iterative generation until the evaluation budget is exhausted.
# Closest known influences: Simplified CMA-ES / Adaptive Step-size Random Search.
# Novelty or unusual aspects: Employs a diagonal covariance scaling to handle ill-conditioned functions with minimal complexity.
# Failure modes: May converge prematurely on highly multi-modal landscapes or fail to converge on extremely sparse global optima.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))

    def __call__(self, func):
        # Extract bounds
        lb = getattr(func, 'lower', getattr(func.bounds, 'lb', None))
        ub = getattr(func, 'upper', getattr(func.bounds, 'ub', None))
        lb = np.array(lb)
        ub = np.array(ub)

        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.2 * (ub - lb)
        diag_c = np.ones(self.dim)
        
        best_x = None
        best_y = float('inf')
        evals = 0

        while evals < self.budget:
            # Generate candidates
            batch_size = min(self.pop_size, self.budget - evals)
            xs = np.random.normal(0, 1, (batch_size, self.dim)) * sigma * diag_c + mean
            xs = np.clip(xs, lb, ub)
            
            # Evaluate
            ys = np.array([func(x) for x in xs])
            evals += batch_size
            
            # Track global best
            idx_best = np.argmin(ys)
            if ys[idx_best] < best_y:
                best_y = ys[idx_best]
                best_x = xs[idx_best]
            
            # Select best half to update mean
            weights = np.argsort(ys)
            n_sel = max(1, batch_size // 2)
            selected = xs[weights[:n_sel]]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.mean(selected, axis=0)
            
            # Adaptation: adjust sigma and diagonal covariance
            if batch_size > 1:
                # Simple momentum-based updates
                diff = (mean - old_mean) / (sigma + 1e-9)
                diag_c = 0.9 * diag_c + 0.1 * np.abs(diff)
                
                # Success-based sigma adjustment (1/5th rule approximation)
                if np.mean(ys[:n_sel]) < np.mean(ys):
                    sigma *= 1.1
                else:
                    sigma *= 0.8
            
        return best_x, best_y
