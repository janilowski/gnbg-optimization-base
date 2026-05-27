# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple yet robust Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant acting as a black-box optimizer.
# Search state: Maintains a mean vector (m), step size (sigma), and a diagonal covariance matrix approximation (C) to model the landscape.
# Candidate generation: Samples a population of individuals from a multivariate normal distribution centered at m.
# Selection and replacement: Uses rank-based recombination to update the mean based on the best-performing fraction of the population.
# Adaptation: Updates sigma via Cumulative Step-size Adaptation (CSA) and C via rank-one update.
# Exploration mechanisms: Initialized with a large sigma; global sampling ensures broad coverage in early iterations.
# Exploitation mechanisms: Reduces sigma as the population converges; updates mean toward better-performing regions.
# Boundary handling: Projects out-of-bounds samples back to the nearest boundary.
# Budget strategy: Divides total budget into generations; stops early if budget is exhausted; adjusts population size to fit.
# Closest known influences: Simplified CMA-ES (separable approximation).
# Novelty or unusual aspects: Lightweight implementation using only NumPy, no overhead of object-oriented complexity found in full CMA-ES packages.
# Failure modes: Can get stuck in local optima if the landscape is highly deceptive; sensitive to initial sigma on small budgets.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        sigma = 0.5  # Initial step size
        m = np.random.uniform(lb, ub)
        c = np.ones(self.dim)
        
        best_x = None
        best_y = float('inf')
        evals = 0
        
        weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
        weights[weights < 0] = 0
        weights /= np.sum(weights)
        
        while evals < self.budget:
            # Generate candidates
            candidates = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                z = np.random.normal(0, 1, self.dim)
                x = m + sigma * z * np.sqrt(c)
                # Boundary handling
                x = np.clip(x, lb, ub)
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                candidates.append((x, y, z))
                
            # Sort by fitness
            candidates.sort(key=lambda x: x[1])
            
            # Update mean
            old_m = m.copy()
            m = np.sum([weights[i] * candidates[i][0] for i in range(len(candidates))], axis=0)
            
            # Update step size and covariance approximation (simplified)
            z_mean = np.sum([weights[i] * candidates[i][2] for i in range(len(candidates))], axis=0)
            sigma *= np.exp(0.5 * (np.linalg.norm(z_mean) / 0.5 - 1))
            c = 0.9 * c + 0.1 * (m - old_m)**2 / (sigma**2 + 1e-12)
            
        return best_x, best_y
