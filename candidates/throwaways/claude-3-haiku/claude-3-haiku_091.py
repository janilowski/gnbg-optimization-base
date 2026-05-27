import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm uses a simple random search strategy to minimize the
#          given objective function within the provided evaluation budget. It
#          generates candidate solutions uniformly within the bounded domain,
#          evaluates them, and keeps track of the best solution found so far.
# Search state: The search state is represented by the current best solution
#                (best_x, best_y).
# Candidate generation: New candidate solutions are generated uniformly within
#                       the bounded domain.
# Selection and replacement: The best candidate solution (if any) replaces the
#                            current best solution.
# Adaptation: There is no adaptation mechanism in this simple random search
#             algorithm.
# Exploration mechanisms: The algorithm explores the search space uniformly by
#                         generating random candidate solutions.
# Exploitation mechanisms: The algorithm exploits the current best solution by
#                          replacing it with a better candidate if found.
# Boundary handling: Candidate solutions are generated within the bounded
#                    domain, so no special boundary handling is required.
# Budget strategy: The algorithm stops when the evaluation budget is exhausted.
# Closest known influences: This algorithm is a basic random search method,
#                           which is a simple and robust optimization approach.
# Novelty or unusual aspects: This algorithm does not have any novel or
#                             unusual aspects, as it is a straightforward
#                             random search implementation.
# Failure modes: The algorithm may perform poorly on highly multimodal or
#                deceptive objective functions, as it relies solely on random
#                exploration without any additional mechanisms to escape local
#                minima.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        self.budget_left = self.budget
        while self.budget_left > 0:
            # Generate a new candidate solution
            candidate_x = np.random.uniform(func.lower, func.upper, self.dim)
            candidate_y = func(candidate_x)
            self.budget_left -= 1

            # Update the best solution if the candidate is better
            if candidate_y < self.best_y:
                self.best_x = candidate_x
                self.best_y = candidate_y

        return self.best_x, self.best_y
