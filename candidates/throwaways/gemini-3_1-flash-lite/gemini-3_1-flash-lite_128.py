# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) styled local search, using a rank-based population update.
# Search state: Maintains a mean vector and a global step size (sigma).
# Candidate generation: Samples points from a multivariate normal distribution centered at the current mean.
# Selection and replacement: Uses rank-based selection to update the distribution mean towards better performing candidates.
# Adaptation: Updates sigma via a simple 1/5th success rule and shifts the mean towards the top performing candidates.
# Exploration mechanisms: Gaussian mutation with adaptive step size ensures broad coverage in early stages.
# Exploitation mechanisms: The mean vector converges toward the current minimum, narrowing the search space as sigma decreases.
# Boundary handling: Clamps generated candidates to the specified function bounds.
# Budget strategy: Precisely exhausts the provided budget by running iterations until the total evaluation count is met.
# Closest known influences: Simplified CMA-ES/Evolution Strategy.
# Novelty or unusual aspects: Compact implementation focusing on robustness in black-box scenarios without complex matrix decompositions.
# Failure modes: Can get stuck in local optima for highly multi-modal functions or fail to converge on extremely sharp, narrow valleys.
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

        # Initialization
        evals_done = 0
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        pop_size = 4 + int(3 * np.log(self.dim))
        best_x = None
        best_y = float('inf')

        while evals_done < self.budget:
            # Generate population
            pop = []
            for _ in range(pop_size):
                if evals_done >= self.budget:
                    break
                
                # Sample and clamp to bounds
                candidate = np.clip(mean + np.random.normal(0, sigma, self.dim), lb, ub)
                y = func(candidate)
                evals_done += 1
                
                if y < best_y:
                    best_y = y
                    best_x = candidate
                
                pop.append((y, candidate))
            
            if not pop:
                break
                
            # Selection: Sort by fitness
            pop.sort(key=lambda x: x[0])
            
            # Update mean: weighted average of the top half
            top_half = pop[:max(1, len(pop) // 2)]
            new_mean = np.mean([p[1] for p in top_half], axis=0)
            
            # Step size adaptation (1/5th success rule-like)
            if pop[0][0] < best_y:
                sigma *= 1.1
            else:
                sigma *= 0.95
                
            mean = new_mean
            
            # Heuristic check for stagnation
            if np.all(sigma < 1e-10):
                sigma = 0.1 * (ub - lb)

        return best_x, best_y
