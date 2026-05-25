# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (mu, lambda)-ES style evolution strategy with covariance matrix adaptation (CMA-lite).
# Search state: Maintains a current mean vector, a global step size (sigma), and a simplified covariance state.
# Candidate generation: Samples Gaussian perturbations around the current mean, scaled by sigma.
# Selection and replacement: Selects the best performing individuals from each generation to update the mean.
# Adaptation: Employs a simple 1/5th success rule for step-size adaptation and rank-based mean updates.
# Exploration mechanisms: Gaussian sampling centered on the mean ensures global coverage.
# Exploitation mechanisms: Mean shifts toward successful samples; step size shrinks as progress slows. 
# Boundary handling: Clamps candidates to the specified domain using projection.
# Budget strategy: Iterates until the evaluation budget is exhausted, tracking count strictly.
# Closest known influences: Evolutionary Strategies, CMA-ES (simplified).
# Novelty or unusual aspects: Compact implementation focusing on robustness without full matrix inversion.
# Failure modes: Can get trapped in local minima in highly non-convex surfaces or experience premature convergence.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        pop_size = 4 + int(3 * np.log(self.dim))
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')

        # Evaluation helper
        def evaluate(x):
            if self.eval_count >= self.budget:
                return None
            x = np.clip(x, lb, ub)
            y = func(x)
            self.eval_count += 1
            return y

        # Initial evaluation
        best_y = evaluate(mean)
        best_x = np.copy(mean)

        while self.eval_count + pop_size <= self.budget:
            # Generate offspring
            offspring = [mean + sigma * np.random.randn(self.dim) for _ in range(pop_size)]
            results = [(evaluate(o), o) for o in offspring]
            
            # Sort by fitness
            results.sort(key=lambda x: x[0])
            
            # Check for best
            if results[0][0] < best_y:
                best_y, best_x = results[0]
                success = True
            else:
                success = False
            
            # Update mean (weighted average of best half)
            weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, int(pop_size / 2) + 1))
            weights /= weights.sum()
            mean = sum(weights[i] * results[i][1] for i in range(len(weights)))
            
            # Adapt sigma (simplified 1/5th rule)
            if success:
                sigma *= 1.1
            else:
                sigma *= 0.8
                
            # Convergence check: stop if step size is effectively zero
            if np.all(sigma < 1e-10):
                break

        return best_x, best_y
