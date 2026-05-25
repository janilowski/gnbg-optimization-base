import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a simple implementation of a randomized search algorithm for
#          the GNBG black-box minimization benchmark. It generates candidate
#          solutions randomly within the given bounds, evaluates them, and keeps
#          track of the best solution found so far.
# Search state: The algorithm maintains the current best solution (best_x,
#                best_y) and the evaluation budget.
# Candidate generation: Candidates are generated randomly within the given
#                      bounds using a uniform distribution.
# Selection and replacement: The best candidate solution replaces the current
#                           best solution if it has a lower function value.
# Adaptation: There is no adaptation mechanism in this simple implementation.
# Exploration mechanisms: The random generation of candidates provides
#                        exploration.
# Exploitation mechanisms: There are no explicit exploitation mechanisms.
# Boundary handling: Candidate solutions that fall outside the given bounds are
#                   projected back onto the boundary.
# Budget strategy: The algorithm uses the entire evaluation budget and does
#                 not have a stopping criterion.
# Closest known influences: This algorithm is similar to a basic random search
#                         approach.
# Novelty or unusual aspects: There are no novel or unusual aspects to this
#                            implementation.
# Failure modes: The algorithm may struggle in high-dimensional problems or
#                problems with complex, multi-modal landscapes, as it relies
#                solely on random search.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            x = np.random.uniform(func.lower, func.upper, size=self.dim)
            y = func(x)
            if y < self.best_y:
                self.best_x = x
                self.best_y = y
        return self.best_x, self.best_y
