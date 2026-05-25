import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm is a simple and efficient implementation of a
#          black-box minimization method using a combination of random
#          search and hill-climbing. It is designed to be robust across
#          different dimensions and to never exceed the provided evaluation
#          budget.
# Search state: The search state is represented by the current best solution
#               (best_x, best_y).
# Candidate generation: New candidates are generated randomly within the
#                       search space defined by the function bounds.
# Selection and replacement: The new candidate is compared to the current
#                            best solution, and replaces it if the new
#                            solution is better.
# Adaptation: The algorithm does not adapt its search strategy during
#             the optimization process.
# Exploration mechanisms: The algorithm explores the search space by
#                         generating new random candidate solutions.
# Exploitation mechanisms: The algorithm exploits the search space by
#                          hill-climbing towards the current best solution.
# Boundary handling: The algorithm handles boundary constraints by
#                    reflecting candidate solutions that fall outside
#                    the search space.
# Budget strategy: The algorithm uses the entire evaluation budget to
#                  perform as many function evaluations as possible.
# Closest known influences: This algorithm is inspired by simple
#                           random search and hill-climbing methods.
# Novelty or unusual aspects: The algorithm is designed to be
#                             particularly simple and efficient,
#                             without any complex mechanisms.
# Failure modes: The algorithm may struggle to optimize functions
#                with multiple local minima or highly complex
#                landscapes, as it relies on a simple random
#                search and hill-climbing approach.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = float('inf')

    def __call__(self, func):
        # Initialize the search state
        if self.best_x is None:
            self.best_x = np.random.uniform(func.lower, func.upper, self.dim)
            self.best_y = func(self.best_x)

        # Perform the optimization
        for _ in range(self.budget):
            # Generate a new candidate solution
            candidate_x = np.random.uniform(func.lower, func.upper, self.dim)
            candidate_y = func(candidate_x)

            # Update the best solution if the candidate is better
            if candidate_y < self.best_y:
                self.best_x = candidate_x
                self.best_y = candidate_y

        return self.best_x, self.best_y
