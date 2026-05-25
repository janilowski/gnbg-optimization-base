import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a compact implementation of a random search algorithm for
#          black-box optimization problems. It samples candidate solutions from
#          a uniform distribution within the problem bounds, and keeps track of
#          the best solution found so far.
# Search state: The algorithm maintains the current best solution and its
#               objective value.
# Candidate generation: Candidates are sampled from a uniform distribution
#                       within the problem bounds.
# Selection and replacement: The best candidate found so far replaces the
#                            current best solution if it has a lower objective
#                            value.
# Adaptation: The algorithm does not adapt its behavior during the optimization
#             process.
# Exploration mechanisms: The algorithm explores the search space uniformly.
# Exploitation mechanisms: The algorithm exploits the best solution found so
#                          far.
# Boundary handling: The algorithm ensures that sampled candidates are within
#                    the problem bounds.
# Budget strategy: The algorithm stops when the evaluation budget is exhausted.
# Closest known influences: This algorithm is a basic random search method,
#                           which is a well-known technique for black-box
#                           optimization.
# Novelty or unusual aspects: The algorithm is designed to be compact and
#                             robust across dimensions, without any unusual
#                             aspects.
# Failure modes: The algorithm may struggle to solve problems with complex
#                objective functions or high-dimensional search spaces, as it
#                does not have any advanced exploration or exploitation
#                mechanisms.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        self.budget_left = self.budget
        self.best_x = None
        self.best_y = np.inf

        while self.budget_left > 0:
            x = np.random.uniform(func.lower, func.upper, self.dim)
            y = func(x)
            self.budget_left -= 1

            if y < self.best_y:
                self.best_x = x
                self.best_y = y

        return self.best_x, self.best_y
