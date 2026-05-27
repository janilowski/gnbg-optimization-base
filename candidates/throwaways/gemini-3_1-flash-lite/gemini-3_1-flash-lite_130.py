# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free trust-region based local search wrapper around a population-based sampling strategy.
# Search state: Maintains a current best solution (center) and a dynamically adjusted search radius (sigma).
# Candidate generation: Generates a population of candidate points using a multivariate normal distribution centered on the current best.
# Selection and replacement: Greedy selection; if a candidate improves the current best, the center moves.
# Adaptation: The search radius (sigma) is halved if no progress is made after a set number of samples, allowing for fine-tuning.
# Exploration mechanisms: Initial search radius is set to 20% of the bounding box size; random sampling ensures wide coverage.
# Exploitation mechanisms: Local convergence is driven by the shrinking trust-region radius as the budget depletes.
# Boundary handling: Candidates are clipped to the specified lower and upper bounds.
# Budget strategy: A fixed limit on total evaluations; stops exactly when budget is exhausted.
# Closest known influences: Simplified version of a (1, lambda)-ES with adaptive step-size.
# Novelty or unusual aspects: Minimalist implementation of adaptive sampling designed for robustness in black-box constraints.
# Failure modes: Susceptible to getting stuck in local minima for highly multimodal landscapes; performance sensitive to initial step size.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        range_val = ub - lb
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        eval_count = 1
        
        # Hyperparameters
        sigma = 0.2 * range_val
        pop_size = 10
        patience = 5
        no_improve_count = 0
        
        while eval_count < self.budget:
            # Generate candidate population
            candidates = []
            for _ in range(pop_size):
                if eval_count >= self.budget:
                    break
                
                # Sample within trust region
                trial_x = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
                trial_y = func(trial_x)
                eval_count += 1
                
                candidates.append((trial_x, trial_y))
            
            # Find best in population
            best_cand_x, best_cand_y = min(candidates, key=lambda c: c[1])
            
            # Update best or shrink search region
            if best_cand_y < best_y:
                best_x, best_y = best_cand_x, best_cand_y
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Adapt radius
            if no_improve_count >= patience:
                sigma *= 0.5
                no_improve_count = 0
                
            # Stop if the radius is effectively zero
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
