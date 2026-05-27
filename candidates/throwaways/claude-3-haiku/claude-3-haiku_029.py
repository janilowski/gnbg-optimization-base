import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm uses a combination of random search and gradient-based optimization to minimize the objective function within the given budget.
# Search state: The algorithm maintains the current best solution and its corresponding function value.
# Candidate generation: New candidate solutions are generated using a combination of random sampling and gradient-based perturbation.
# Selection and replacement: The new candidate solution replaces the current best solution if it has a lower function value.
# Adaptation: The algorithm adapts the step size for the gradient-based perturbation based on the success rate of the previous iterations.
# Exploration mechanisms: The random sampling component of the candidate generation provides exploration of the search space.
# Exploitation mechanisms: The gradient-based perturbation component of the candidate generation focuses the search around the current best solution.
# Boundary handling: The algorithm ensures that the generated candidate solutions are within the bounds of the search space.
# Budget strategy: The algorithm uses the entire evaluation budget, attempting to find the best solution within the given constraints.
# Closest known influences: The algorithm is inspired by a combination of random search and gradient-based optimization techniques, which are common approaches in black-box optimization.
# Novelty or unusual aspects: The combination of random search and gradient-based optimization, along with the adaptive step size, is a unique approach to the GNBG black-box minimization benchmark.
# Failure modes: The algorithm may struggle in high-dimensional or multimodal objective functions, where the gradient-based optimization component may get stuck in local minima.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = float('inf')
        self.step_size = 1.0

    def __call__(self, func):
        # Read bounds from the function
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            self.bounds = np.array([func.lower, func.upper]).T
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            self.bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        else:
            raise ValueError('Function does not provide bounds information')

        # Initialize the best solution
        self.best_x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.dim)
        self.best_y = func(self.best_x)
        self.budget -= 1

        # Optimize within the budget
        while self.budget > 0:
            # Generate a new candidate solution
            candidate_x = self.best_x + self.step_size * np.random.normal(0, 1, size=self.dim)
            candidate_x = np.clip(candidate_x, self.bounds[:, 0], self.bounds[:, 1])
            candidate_y = func(candidate_x)
            self.budget -= 1

            # Update the best solution if the candidate is better
            if candidate_y < self.best_y:
                self.best_x = candidate_x
                self.best_y = candidate_y

            # Adapt the step size
            self.step_size *= 0.9 if candidate_y < self.best_y else 1.1

        return self.best_x, self.best_y
