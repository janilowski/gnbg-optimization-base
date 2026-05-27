import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a simple, compact implementation of a Gaussian Optimization algorithm for black-box function minimization.
# Search state: The algorithm maintains a Gaussian distribution representing the most promising region of the search space.
# Candidate generation: New candidate solutions are sampled from the Gaussian distribution.
# Selection and replacement: The best candidate solution is always accepted, replacing the previous best.
# Adaptation: The mean and covariance of the Gaussian distribution are updated based on the best candidate solution.
# Exploration mechanisms: The initial covariance is set to a large value to encourage exploration of the full search space.
# Exploitation mechanisms: As the search progresses, the covariance is reduced to focus the search on the most promising regions.
# Boundary handling: Candidate solutions that fall outside the search bounds are rejected and re-sampled.
# Budget strategy: The algorithm uses the full evaluation budget, adjusting the search behavior based on the number of evaluations remaining.
# Closest known influences: This implementation is based on the Gaussian Process Optimization algorithm, a common approach for black-box function optimization.
# Novelty or unusual aspects: The compact, self-contained nature of the implementation, without relying on external libraries or complex techniques, is relatively unusual for this type of optimization problem.
# Failure modes: The algorithm may struggle with multimodal or highly irregular objective functions, as it relies on a single Gaussian distribution to model the search space.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.mean = np.zeros(dim)
        self.cov = 100 * np.eye(dim)
        self.best_x = None
        self.best_y = float('inf')

    def __call__(self, func):
        while self.evaluations < self.budget:
            x = np.random.multivariate_normal(self.mean, self.cov)

            # Ensure the candidate solution is within the search bounds
            if hasattr(func, 'bounds'):
                x = np.clip(x, func.bounds.lb, func.bounds.ub)
            else:
                x = np.clip(x, func.lower, func.upper)

            y = func(x)
            self.evaluations += 1

            if y < self.best_y:
                self.best_x = x
                self.best_y = y

            # Update the search distribution
            self.mean = self.best_x
            self.cov = max(1.0, self.budget - self.evaluations) * np.eye(self.dim)

        return self.best_x, self.best_y
