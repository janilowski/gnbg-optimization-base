# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using a shrinking adaptive Gaussian step size.
# Search state: Maintains the current best point found so far.
# Candidate generation: Generates a batch of candidates around the current best using multivariate normal noise.
# Selection and replacement: Greedy strategy where the current best is updated if a candidate produces a lower objective value.
# Adaptation: The step size (sigma) is adjusted based on success rate (1/5th rule idea) to balance exploration and exploitation.
# Exploration mechanisms: Initially uses a global scale, transitioning to local refinement through step size decay.
# Exploitation mechanisms: Local intensification around the current known minimum.
# Boundary handling: Projects candidates back into the bounds if they exceed defined limits.
# Budget strategy: Iterates until the evaluation budget is exhausted, with early termination if progress stalls.
# Closest known influences: Adaptive Random Search / (1+lambda)-ES.
# Novelty or unusual aspects: Combines adaptive step scaling with strict boundary enforcement.
# Failure modes: Susceptible to getting trapped in local optima for highly multi-modal functions.
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
        x_best = np.random.uniform(lb, ub, self.dim)
        y_best = func(x_best)
        evals = 1
        
        # Adaptive parameters
        sigma = 0.2 * (ub - lb)
        step_decay = 0.99
        batch_size = 8
        
        while evals < self.budget:
            # Generate local candidates
            candidates = []
            for _ in range(batch_size):
                if evals >= self.budget:
                    break
                
                # Perturb and project
                x_cand = np.clip(x_best + np.random.normal(0, sigma), lb, ub)
                y_cand = func(x_cand)
                evals += 1
                
                candidates.append((x_cand, y_cand))
            
            # Find best in batch
            best_batch = min(candidates, key=lambda x: x[1])
            
            # Update best or shrink search space
            if best_batch[1] < y_best:
                x_best, y_best = best_batch
                sigma = np.minimum(sigma * 1.1, (ub - lb) * 0.5)
            else:
                sigma *= step_decay
            
            # Convergence check: stop if step becomes negligible
            if np.sum(sigma) < 1e-12:
                break
                
        return x_best, y_best
