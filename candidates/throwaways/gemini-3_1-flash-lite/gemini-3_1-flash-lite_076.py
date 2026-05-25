# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Gaussian search algorithm using an adaptive step size and population-based candidate selection.
# Search state: Maintains a current mean position and a global step size (sigma).
# Candidate generation: Generates a population of candidates by sampling from a multivariate normal distribution centered at the mean.
# Selection and replacement: Uses a rank-based selection where the current mean is updated towards the best candidate found in the population.
# Adaptation: Sigma is updated based on the success rate of finding better solutions relative to the previous iteration.
# Exploration mechanisms: Controlled by the current sigma (step size) and random Gaussian perturbations.
# Exploitation mechanisms: The mean of the search distribution shifts toward the best observed objective values.
# Boundary handling: Candidates are clipped to the specified lower and upper bounds.
# Budget strategy: A fixed population size is used, and the loop terminates exactly when the budget is reached.
# Closest known influences: Simplified Evolution Strategy (ES) / (1+lambda)-ES.
# Novelty or unusual aspects: Minimalist implementation focused on parameter-less adaptation of the step size.
# Failure modes: Can get trapped in narrow local minima; sigma decay may be too fast in high-dimensional, highly rugged landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 8 + int(3 * np.log(dim))

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize
        x = np.random.uniform(lb, ub)
        best_x = np.copy(x)
        best_y = func(x)
        remaining_budget = self.budget - 1
        
        sigma = 0.2 * (ub - lb)
        
        while remaining_budget > 0:
            current_pop_size = min(int(remaining_budget), self.pop_size)
            if current_pop_size <= 0:
                break
                
            # Generate candidates
            candidates = []
            for _ in range(current_pop_size):
                cand = np.clip(x + np.random.normal(0, sigma), lb, ub)
                cand_y = func(cand)
                remaining_budget -= 1
                candidates.append((cand, cand_y))
                
                # Update global best
                if cand_y < best_y:
                    best_y = cand_y
                    best_x = np.copy(cand)
                    
            # Find the best in population
            best_in_pop = min(candidates, key=lambda item: item[1])
            
            # Adaptation: move mean if improvement found
            if best_in_pop[1] < best_y:
                x = best_in_pop[0]
                sigma *= 1.1 # Success: expand search
            else:
                sigma *= 0.9 # Failure: contract search
                
        return best_x, best_y
