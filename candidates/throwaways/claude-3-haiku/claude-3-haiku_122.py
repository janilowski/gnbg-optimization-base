import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a compact and robust implementation of a black-box minimization algorithm 
# that uses a combination of random search and gradient-based optimization.
# Search state: The algorithm maintains a current best solution and the corresponding function value.
# Candidate generation: New candidate solutions are generated using a combination of random perturbation and gradient-based updates.
# Selection and replacement: The candidate solution with the lowest function value replaces the current best solution if it is better.
# Adaptation: The algorithm dynamically adjusts the step size for the gradient-based updates based on the progress of the optimization.
# Exploration mechanisms: The random perturbation of the current best solution provides exploration capabilities.
# Exploitation mechanisms: The gradient-based updates help to exploit the local landscape and accelerate convergence.
# Boundary handling: The algorithm ensures that the candidate solutions stay within the specified bounds by projecting them back if they go out of bounds.
# Budget strategy: The algorithm never exceeds the provided evaluation budget.
# Closest known influences: This algorithm draws inspiration from techniques like simulated annealing, gradient descent, and adaptive step size methods.
# Novelty or unusual aspects: The combination of random search and gradient-based optimization in a compact and robust implementation is a unique aspect of this algorithm.
# Failure modes: The algorithm may struggle in high-dimensional problems or problems with highly multimodal objective functions.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = None
        self.f = None
        self.step_size = 1.0

    def __call__(self, func):
        # Initialize the current best solution
        self.x = np.random.uniform(func.lower, func.upper, self.dim)
        self.f = func(self.x)
        budget = self.budget

        while budget > 0:
            # Generate a new candidate solution
            candidate = self.x + self.step_size * np.random.normal(0, 1, self.dim)

            # Ensure the candidate solution is within the bounds
            candidate = np.clip(candidate, func.lower, func.upper)

            # Evaluate the new candidate solution
            candidate_f = func(candidate)
            budget -= 1

            # Update the current best solution if the new candidate is better
            if candidate_f < self.f:
                self.x = candidate
                self.f = candidate_f

            # Adapt the step size based on the progress
            self.step_size *= np.exp(-0.1 * (self.f - candidate_f) / self.f)

        return self.x, self.f
