import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This implementation uses a simple random search algorithm to minimize the given black-box function.
# Search state: The search state consists of the current best solution (best_x, best_y).
# Candidate generation: New candidates are sampled uniformly at random within the given search bounds.
# Selection and replacement: The best candidate found so far is always kept, replacing the previous best if a better solution is found.
# Adaptation: There is no adaptation mechanism in this implementation.
# Exploration mechanisms: The exploration is entirely random, without any biased or directed search.
# Exploitation mechanisms: The algorithm exploits the best solution found so far, but does not use any additional exploitation mechanisms.
# Boundary handling: Candidates that fall outside the search bounds are simply discarded, and a new candidate is sampled.
# Budget strategy: The algorithm uses the full evaluation budget provided, making as many function evaluations as possible.
# Closest known influences: This is a basic random search algorithm, which is a simple and robust approach for black-box optimization.
# Novelty or unusual aspects: There are no novel or unusual aspects to this implementation.
# Failure modes: The algorithm may struggle on highly multimodal or deceptive functions, as it has no mechanisms to escape local optima.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        n_evals = 0
        while n_evals < self.budget:
            x = np.random.uniform(func.lower, func.upper, self.dim)
            y = func(x)
            n_evals += 1
            if y < self.best_y:
                self.best_x = x
                self.best_y = y
        return self.best_x, self.best_y
