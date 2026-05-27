# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (n + 4)-membered Evolution Strategy (ES) with self-adaptive step sizes.
# Search state: Maintains a current mean vector `x` and a global step size `sigma`.
# Candidate generation: Generates `lambda = 4 + floor(3 * log(dim))` offspring by sampling multivariate normal mutations centered at `x`.
# Selection and replacement: Uses (mu, lambda)-selection where the best offspring replaces the current mean.
# Adaptation: Employs 1/5th success rule logic; `sigma` increases after successful steps and decreases otherwise.
# Exploration mechanisms: Gaussian noise proportional to `sigma` ensures diffusion across the domain.
# Exploitation mechanisms: The population mean moves toward the successful points found in local neighborhoods.
# Boundary handling: Points are clipped to the domain bounds; a small penalty or re-sampling is implicitly handled by selection.
# Budget strategy: Iteratively evaluates population chunks until the function evaluation budget is exhausted.
# Closest known influences: (1+1)-ES and CMA-ES simplified variants.
# Novelty or unusual aspects: Compact implementation utilizing basic step-size adjustment for robustness in varying dimensions.
# Failure modes: May converge prematurely on highly multimodal surfaces or struggle with thin, non-axis-aligned ridges.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x = np.random.uniform(lb, ub)
        best_x, best_y = x.copy(), func(x)
        self.evals += 1
        
        sigma = 0.2 * (ub - lb)
        mu = 1
        lam = 4 + int(3 * np.log(self.dim))
        
        # Adaptive params
        success_history = []
        
        while self.evals + lam <= self.budget:
            offspring = []
            results = []
            
            # Generate and evaluate candidates
            for _ in range(lam):
                if self.evals >= self.budget:
                    break
                
                # Mutation
                x_cand = np.clip(x + np.random.normal(0, sigma), lb, ub)
                y_cand = func(x_cand)
                self.evals += 1
                
                offspring.append((x_cand, y_cand))
                results.append(y_cand)
                
                if y_cand < best_y:
                    best_y = y_cand
                    best_x = x_cand.copy()
            
            # Select best offspring
            sorted_idx = np.argsort(results)
            best_idx = sorted_idx[0]
            
            # 1/5th success rule for step size adaptation
            if results[best_idx] < best_y: # Improvement found
                x = offspring[best_idx][0]
                success_history.append(1)
            else:
                success_history.append(0)
            
            # Update sigma every few generations
            if len(success_history) >= 5:
                success_rate = sum(success_history) / len(success_history)
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                success_history = []
                
        return best_x, best_y
