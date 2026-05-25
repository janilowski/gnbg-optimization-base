import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm uses a simple random search approach to minimize the
#          given black-box function within the specified budget and dimensionality.
# Search state: The algorithm maintains the current best solution (best_x, best_y).
# Candidate generation: New candidate solutions are sampled uniformly at random
#                      within the function's bounded domain.
# Selection and replacement: The best (lowest) objective function value seen so
#                           far is always kept as the current best solution.
# Adaptation: There is no adaptation mechanism; the search is purely random.
# Exploration mechanisms: The random sampling of new candidates provides the
#                        exploration mechanism.
# Exploitation mechanisms: The algorithm selects the best solution found so far,
#                         which provides a simple exploitation mechanism.
# Boundary handling: Candidate solutions that fall outside the function's bounded
#                   domain are discarded and a new candidate is sampled.
# Budget strategy: The algorithm uses the full evaluation budget provided and
#                  does not attempt to manage the budget.
# Closest known influences: This algorithm is a basic random search approach,
#                          which is a common baseline for black-box optimization.
# Novelty or unusual aspects: There are no novel or unusual aspects to this
#                            implementation; it is a straightforward random
#                            search approach.
# Failure modes: The algorithm may struggle to find good solutions for complex
#               objective functions, especially in high-dimensional spaces,
#               as it does not employ any advanced search strategies.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        # Initialize the best solution
        if self.best_x is None:
            self.best_x = np.random.uniform(func.lower, func.upper, self.dim)
            self.best_y = func(self.best_x)

        # Perform random search
        for _ in range(self.budget - 1):
            # Sample a new candidate solution
            x = np.random.uniform(func.lower, func.upper, self.dim)
            y = func(x)

            # Update the best solution if the new candidate is better
            if y < self.best_y:
                self.best_x = x
                self.best_y = y

        return self.best_x, self.best_y
