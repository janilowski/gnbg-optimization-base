# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES inspired evolution strategy (Simplified Covariance Matrix Adaptation).
# Search state: Maintains a mean vector, a global step size (sigma), and an identity-based covariance structure.
# Candidate generation: Samples population members from a multivariate normal distribution centered at the current mean.
# Selection and replacement: Selects the top 50% of candidates (truncation selection) to update the mean.
# Adaptation: Updates the mean based on the weighted center of mass of the best candidates.
# Exploration mechanisms: Initial sigma is set to a fraction of the domain range; search converges by decaying sigma.
# Exploitation mechanisms: The mean consistently shifts toward the best-performing regions of the sample batch.
# Boundary handling: Projects candidates back into the [lower, upper] box using a clip-and-reflect approach.
# Budget strategy: Iterative loop that terminates once the total function evaluation count reaches the budget.
# Closest known influences: Simplified (1+1) / (mu, lambda)-ES with isotropic covariance adaptation.
# Novelty or unusual aspects: Extremely compact implementation of a stochastic hill-climber with step-size decay.
# Failure modes: Susceptible to local optima in highly multi-modal landscapes due to the lack of long-range jumps.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        pop_size = 4 + int(3 * np.log(self.dim))
        mu = int(pop_size / 2)
        mean = np.random.uniform(lb, ub)
        sigma = 0.2 * (ub - lb)
        
        best_x = None
        best_y = float('inf')

        while self.evals < self.budget:
            # Generate candidates
            candidates = []
            for _ in range(pop_size):
                if self.evals >= self.budget:
                    break
                
                x = mean + sigma * np.random.randn(self.dim)
                x = np.clip(x, lb, ub)
                y = func(x)
                self.evals += 1
                
                candidates.append((y, x))
                
                if y < best_y:
                    best_y = y
                    best_x = x

            # Sort and select top mu
            candidates.sort(key=lambda item: item[0])
            top_candidates = [c[1] for c in candidates[:mu]]
            
            # Update mean to center of selected points
            new_mean = np.mean(top_candidates, axis=0)
            
            # Adaptation: simple step-size decay schedule
            sigma *= 0.95
            
            # Stop if step size is effectively zero
            if np.all(sigma < 1e-10):
                break
                
            mean = new_mean

        return best_x, best_y
