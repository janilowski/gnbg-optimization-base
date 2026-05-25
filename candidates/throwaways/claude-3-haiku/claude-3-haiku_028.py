import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm uses a simple random search strategy to minimize the objective function within the provided budget.
# Search state: The algorithm maintains a single current best solution.
# Candidate generation: New candidate solutions are generated randomly within the problem bounds.
# Selection and replacement: The new candidate solution replaces the current best solution if it has a lower objective function value.
# Adaptation: There is no adaptation mechanism in this algorithm.
# Exploration mechanisms: The random generation of new candidate solutions provides exploration.
# Exploitation mechanisms: The replacement of the current best solution with a better candidate provides exploitation.
# Boundary handling: Candidate solutions that fall outside the problem bounds are clipped to the nearest valid value.
# Budget strategy: The algorithm simply runs for the provided budget, making as many function evaluations as possible.
# Closest known influences: This algorithm is a basic random search method, which is a common baseline for black-box optimization problems.
# Novelty or unusual aspects: This implementation is purposefully simple and straightforward, with no advanced techniques.
# Failure modes: The algorithm may struggle with high-dimensional or multimodal problems, as the random search strategy is unlikely to find the global optimum efficiently.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Generate a new candidate solution
            x = np.random.uniform(func.lower, func.upper, self.dim)

            # Evaluate the candidate solution
            y = func(x)

            # Update the best solution if the candidate is better
            if y < self.best_y:
                self.best_x = x
                self.best_y = y

        return self.best_x, self.best_y
