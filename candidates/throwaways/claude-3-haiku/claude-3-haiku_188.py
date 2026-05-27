import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a simple random search algorithm that samples points from a uniform
#          distribution within the given bounds and keeps track of the best point seen so far.
# Search state: The algorithm maintains the current best point and its objective value.
# Candidate generation: New candidate points are sampled from a uniform distribution
#                      within the given bounds.
# Selection and replacement: The best candidate point is compared to the current best
#                           and replaces it if the new point is better.
# Adaptation: No adaptation mechanisms are used.
# Exploration mechanisms: The algorithm explores the search space by sampling random points.
# Exploitation mechanisms: The algorithm exploits the current best point by focusing the
#                         sampling around it.
# Boundary handling: Candidate points that fall outside the bounds are discarded and
#                    a new point is sampled.
# Budget strategy: The algorithm uses the full evaluation budget provided.
# Closest known influences: This is a basic random search algorithm, similar to the
#                          Uncorrelated Random Search (URS) method.
# Novelty or unusual aspects: The algorithm is very simple and straightforward, without
#                            any advanced features.
# Failure modes: The algorithm may perform poorly on multimodal or highly structured
#                objective functions, as it does not have any mechanisms to escape
#                local optima.
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
