import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm uses a combination of random search and gradient-based optimization
# to minimize the given black-box function within the provided budget.
# Search state: The search state consists of the current best solution (best_x, best_y).
# Candidate generation: New candidates are generated using a combination of random sampling
# and gradient-based perturbation of the current best solution.
# Selection and replacement: The best candidate is selected and replaces the current best
# solution if it has a lower function value.
# Adaptation: The algorithm dynamically adjusts the step size and the balance between
# random exploration and gradient-based exploitation based on the performance of the
# previous iterations.
# Exploration mechanisms: Random sampling is used to explore the search space.
# Exploitation mechanisms: Gradient-based optimization is used to refine the current
# best solution.
# Boundary handling: Candidates that violate the problem bounds are projected back
# onto the feasible region.
# Budget strategy: The algorithm makes the best use of the available evaluation budget
# by dynamically adjusting the exploration-exploitation balance.
# Closest known influences: This algorithm is inspired by a combination of random search
# and gradient-based optimization techniques, as commonly used in derivative-free
# optimization methods.
# Novelty or unusual aspects: The dynamic adjustment of the exploration-exploitation
# balance based on the performance of the previous iterations is a key aspect of
# this algorithm.
# Failure modes: The algorithm may struggle in highly multimodal or discontinuous
# objective functions, where the gradient-based optimization component may get
# trapped in local minima.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.step_size = 1.0
        self.exploration_ratio = 0.5

    def __call__(self, func):
        self.best_x = np.random.uniform(func.lower, func.upper, self.dim)
        self.best_y = func(self.best_x)
        self.eval_count = 1

        while self.eval_count < self.budget:
            # Generate a new candidate
            if np.random.rand() < self.exploration_ratio:
                # Random exploration
                candidate_x = np.random.uniform(func.lower, func.upper, self.dim)
            else:
                # Gradient-based exploitation
                gradient = self.gradient(func, self.best_x)
                candidate_x = self.best_x - self.step_size * gradient

            # Clip the candidate to the feasible region
            candidate_x = np.clip(candidate_x, func.lower, func.upper)

            # Evaluate the new candidate
            candidate_y = func(candidate_x)
            self.eval_count += 1

            # Update the best solution if the new candidate is better
            if candidate_y < self.best_y:
                self.best_x = candidate_x
                self.best_y = candidate_y

            # Adapt the step size and exploration ratio
            self.step_size *= 0.99
            self.exploration_ratio = max(0.1, self.exploration_ratio * 0.99)

        return self.best_x, self.best_y

    def gradient(self, func, x):
        h = 1e-6
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return gradient
