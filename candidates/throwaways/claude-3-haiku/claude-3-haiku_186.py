import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a compact implementation of a random search algorithm for
# the GNBG black-box minimization benchmark. It generates candidate solutions
# randomly within the search bounds, evaluates them, and keeps track of the
# best solution found so far.
# Search state: The algorithm maintains the current best solution (best_x,
# best_y) found so far.
# Candidate generation: Candidates are generated randomly within the search
# bounds using numpy's random number generator.
# Selection and replacement: The best candidate solution replaces the current
# best solution if it has a lower objective value.
# Adaptation: There is no adaptation mechanism in this simple random search
# algorithm.
# Exploration mechanisms: The algorithm explores the search space randomly,
# without any biases or preferences.
# Exploitation mechanisms: The algorithm exploits the best solution found so
# far by replacing it with a better candidate if one is found.
# Boundary handling: The algorithm ensures that candidate solutions are
# generated within the search bounds by clipping them if necessary.
# Budget strategy: The algorithm uses the entire evaluation budget provided
# and does not prematurely stop the search.
# Closest known influences: This algorithm is a basic random search approach,
# similar to the Monte Carlo method for global optimization.
# Novelty or unusual aspects: The algorithm is intentionally kept simple and
# compact, focusing on the core functionality required for the GNBG benchmark.
# Failure modes: The algorithm may struggle in high-dimensional problems or
# problems with complex, multimodal objective functions, as it relies solely
# on random exploration without any advanced search mechanisms.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = float('inf')

    def __call__(self, func):
        # Initialize the best solution to None
        self.best_x = None
        self.best_y = float('inf')

        # Iterate through the evaluation budget
        for _ in range(self.budget):
            # Generate a random candidate solution within the search bounds
            x = np.random.uniform(func.lower, func.upper, self.dim)

            # Evaluate the candidate solution
            y = func(x)

            # Update the best solution if the candidate is better
            if y < self.best_y:
                self.best_x = x
                self.best_y = y

        return self.best_x, self.best_y
