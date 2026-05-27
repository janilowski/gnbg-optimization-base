# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES inspired evolution strategy (Simplified Evolutionary Strategy).
# Search state: Maintains a current mean position and a global step size (sigma).
# Candidate generation: Samples new points from a multivariate normal distribution centered at the mean.
# Selection and replacement: Uses (mu, lambda) selection, keeping only the best performing offspring to update the mean.
# Adaptation: Step size is adapted using the "1/5th success rule" to manage exploration vs exploitation.
# Exploration mechanisms: Gaussian sampling scaled by the current step size.
# Exploitation mechanisms: Mean-shifting towards the best individuals found in the current generation.
# Boundary handling: Clamping samples to the defined search space bounds.
# Budget strategy: Divides budget into generations; terminates when no points remain.
# Closest known influences: CMA-ES and (mu, lambda)-ES.
# Novelty or unusual aspects: Extremely compact implementation of a robust evolutionary strategy.
# Failure modes: May converge to local optima on highly multimodal functions or get stuck if step size collapses too early.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        self.mu = self.pop_size // 2

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
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
            candidates = []
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and clamp
                x = np.clip(mean + sigma * np.random.randn(self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                candidates.append((x, y))
                
                if y < best_y:
                    best_y = y
                    best_x = x

            # Sort and select mu best
            candidates.sort(key=lambda item: item[1])
            new_mean = np.mean([c[0] for c in candidates[:self.mu]], axis=0)
            
            # Adaptation: 1/5th success rule (simplified)
            if candidates[0][1] < best_y:
                sigma *= 1.2
            else:
                sigma *= 0.8
            
            mean = new_mean
            
        return best_x, best_y
