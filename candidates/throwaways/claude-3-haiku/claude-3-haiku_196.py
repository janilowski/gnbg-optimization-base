import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a compact, randomized direct search algorithm for
#   black-box minimization problems. It samples points uniformly within the
#   search space and keeps track of the best-so-far point.
# Search state: The algorithm maintains a single current best point.
# Candidate generation: Candidate points are sampled uniformly within the
#   search space.
# Selection and replacement: At each iteration, the new candidate point
#   replaces the current best point if it has a lower function value.
# Adaptation: There is no adaptation mechanism in this algorithm.
# Exploration mechanisms: Uniform random sampling within the bounds provides
#   exploration.
# Exploitation mechanisms: Focusing the search on the current best point
#   provides a form of exploitation.
# Boundary handling: Candidate points that fall outside the search space
#   bounds are clipped to the nearest valid point.
# Budget strategy: The algorithm simply evaluates the function the requested
#   number of times and returns the best point found.
# Closest known influences: This algorithm is similar to a basic random
#   search or pure random restart strategy.
# Novelty or unusual aspects: The algorithm is designed to be very simple
#   and compact, relying only on the Python standard library and NumPy.
# Failure modes: The algorithm may struggle on high-dimensional problems or
#   problems with highly multimodal or deceptive landscapes, as it has no
#   mechanism for escaping local minima.
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
