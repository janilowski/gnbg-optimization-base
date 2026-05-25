import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a simple random search algorithm that generates candidate
#   solutions uniformly within the search bounds and selects the best one seen
#   so far.
# Search state: The algorithm maintains the best solution seen so far.
# Candidate generation: Candidates are generated uniformly at random within the
#   search bounds.
# Selection and replacement: The best candidate seen so far is always kept.
# Adaptation: There is no adaptation mechanism.
# Exploration mechanisms: Uniform random search provides exploration.
# Exploitation mechanisms: There is no explicit exploitation mechanism.
# Boundary handling: Candidate solutions that fall outside the search bounds are
#   discarded.
# Budget strategy: The algorithm consumes the full evaluation budget.
# Closest known influences: This is a basic random search algorithm.
# Novelty or unusual aspects: There are no novel or unusual aspects to this
#   algorithm.
# Failure modes: The algorithm may perform poorly on multimodal or highly
#   structured problems where random search is unlikely to find the global
#   optimum.
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
