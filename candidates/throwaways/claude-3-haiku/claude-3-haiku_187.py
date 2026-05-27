import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm is a simple random search approach for black-box
#          optimization problems. It generates candidate solutions randomly
#          within the given bounds, evaluates them, and keeps track of the
#          best solution found so far.
# Search state: The algorithm maintains a single best solution, which is
#               updated whenever a better candidate is found.
# Candidate generation: Candidates are generated randomly within the search
#                       bounds using a uniform distribution.
# Selection and replacement: The algorithm always selects the best candidate
#                            found so far to be the new best solution.
# Adaptation: There is no adaptation mechanism in this algorithm.
# Exploration mechanisms: The algorithm explores the search space randomly
#                         by generating new candidate solutions.
# Exploitation mechanisms: The algorithm exploits the best solution found
#                          so far by using it as the starting point for
#                          generating new candidates.
# Boundary handling: The algorithm generates candidates within the given
#                    search bounds. Any candidate that falls outside the
#                    bounds is automatically discarded.
# Budget strategy: The algorithm uses the provided evaluation budget and
#                  stops when the budget is exhausted.
# Closest known influences: This algorithm is a basic random search
#                          approach, which is a common technique for
#                          black-box optimization.
# Novelty or unusual aspects: This algorithm does not have any
#                            unusual or novel aspects.
# Failure modes: The algorithm may not be able to find the global
#                optimum, especially for complex, high-dimensional,
#                or multimodal problems, as it relies on random
#                exploration without any specialized mechanisms.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            x = np.random.uniform(func.lower, func.upper, self.dim)
            y = func(x)
            if y < self.best_y:
                self.best_x = x
                self.best_y = y
        return self.best_x, self.best_y
