import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a simple implementation of a random search algorithm for
# the GNBG black-box minimization benchmark. It generates candidate solutions
# randomly within the search bounds, evaluates them, and keeps track of the
# best solution found so far.
# Search state: The algorithm maintains a single best solution found so far.
# Candidate generation: Candidates are generated randomly within the search
# bounds, using a uniform distribution.
# Selection and replacement: The best candidate found so far is always kept.
# Adaptation: There is no adaptation mechanism in this algorithm.
# Exploration mechanisms: The algorithm explores the search space randomly.
# Exploitation mechanisms: The algorithm exploits the best solution found so
# far.
# Boundary handling: Candidates that fall outside the search bounds are
# clipped to the nearest valid value.
# Budget strategy: The algorithm uses the full evaluation budget provided.
# Closest known influences: This algorithm is a basic random search, similar
# to a Monte Carlo method.
# Novelty or unusual aspects: There are no novel or unusual aspects to this
# implementation.
# Failure modes: The algorithm may struggle on high-dimensional problems with
# complex landscapes, as it has no mechanisms for exploiting structure in the
# objective function.
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
