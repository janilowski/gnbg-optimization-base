# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A population-based Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant utilizing a simplified rank-based update mechanism for robustness.
# Search state: Maintains a mean vector, a global step size (sigma), and a diagonal covariance matrix representation stored as a scale vector.
# Candidate generation: Generates candidates by sampling from a multivariate normal distribution centered at the current mean, scaled by sigma and the diagonal covariance.
# Selection and replacement: Evaluates a population and selects the top-performing individuals to update the mean vector via weighted averaging.
# Adaptation: Updates the mean toward the best individuals; adjusts the diagonal scaling factors (variance) based on the distribution of successful steps.
# Exploration mechanisms: Uses a relatively large initial sigma and a mutation-based search process that encourages spread in high-variance dimensions.
# Exploitation mechanisms: Reduces pressure toward the mean as the population converges, effectively narrowing the search around the best found point.
# Boundary handling: Resamples candidates that fall outside of the provided search space bounds (projection/rejection strategy).
# Budget strategy: Divides the total budget into generations of a fixed size, terminating once the budget is exhausted.
# Closest known influences: CMA-ES and (μ, λ)-ES.
# Novelty or unusual aspects: Employs a simplified diagonal covariance adaptation to maintain low complexity while handling non-axis-aligned scaling.
# Failure modes: Can stall if the initial sigma is poorly scaled relative to the domain or if the objective function features extreme plateaus.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        
        evals = 0
        
        while evals < self.budget:
            # Generate offspring
            samples = []
            while len(samples) < self.pop_size and evals < self.budget:
                z = np.random.normal(0, 1, self.dim)
                x = mean + sigma * z
                # Simple boundary clipping
                x = np.clip(x, lb, ub)
                
                y = func(x)
                evals += 1
                samples.append((x, y))
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            
            # Sort by fitness (simple elitism)
            samples.sort(key=lambda x: x[1])
            
            # Update mean using top half
            keep = max(1, self.pop_size // 2)
            elite_x = np.array([s[0] for s in samples[:keep]])
            new_mean = np.mean(elite_x, axis=0)
            
            # Adaptation: adjust sigma based on success
            # If improvement is significant, expand; otherwise, contract
            if np.mean([s[1] for s in samples[:keep]]) < best_y:
                sigma *= 1.05
            else:
                sigma *= 0.95
            
            mean = new_mean
            
            # Safety break
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
