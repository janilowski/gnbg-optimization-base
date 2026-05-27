# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (n + 1)-point Evolution Strategy using rank-based adaptation.
# Search state: Maintains a current mean position and a global step size (sigma).
# Candidate generation: Samples perturbations from a multivariate normal distribution added to the mean.
# Selection and replacement: Evaluates the population plus the current mean, replacing the mean with the best performer.
# Adaptation: Updates sigma via the 1/5th success rule—increasing if the best candidate outperforms the mean.
# Exploration mechanisms: Isotropic Gaussian mutation controlled by the sigma parameter.
# Exploitation mechanisms: Greedy movement towards the best-found candidate in each step.
# Boundary handling: Projects candidates back into the feasible region using a simple clip operation.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: (1+1)-ES and CMA-ES.
# Novelty or unusual aspects: Simplified implementation using only n+1 points per generation to maintain stability in low budgets.
# Failure modes: Can get stuck in local optima on highly multi-modal landscapes due to the lack of a large population.
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
        elif hasattr(func, 'bounds') and hasattr(func, 'bounds'):
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        else:
            lb, ub = np.full(self.dim, -5.0), np.full(self.dim, 5.0)

        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = (ub - lb) / 4.0
        best_x = mean.copy()
        best_y = func(mean)
        budget_left = self.budget - 1

        # Evolution loop
        while budget_left > 0:
            # Generate n samples (or at least 2)
            num_samples = max(2, self.dim)
            candidates = []
            for _ in range(num_samples):
                if budget_left <= 0:
                    break
                
                # Perturb and clip
                trial = np.clip(mean + np.random.normal(0, sigma, self.dim), lb, ub)
                val = func(trial)
                budget_left -= 1
                
                candidates.append((val, trial))
                
                # Track global best
                if val < best_y:
                    best_y = val
                    best_x = trial.copy()

            # Find best in current batch
            best_idx = np.argmin([c[0] for c in candidates])
            current_best_val, current_best_vec = candidates[best_idx]

            # 1/5th Success Rule for Sigma Adaptation
            if current_best_val < best_y:
                # Success: Expand search
                sigma *= 1.2
                mean = current_best_vec
            else:
                # Failure: Contract search
                sigma *= 0.8

            # Safety check on minimum sigma to prevent stagnation
            if np.all(sigma < 1e-10):
                sigma = (ub - lb) * 0.01

        return best_x, best_y
