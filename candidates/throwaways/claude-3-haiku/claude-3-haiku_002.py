import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a simple stochastic search algorithm that generates random candidate solutions uniformly within the search domain and selects the best one seen so far.
# Search state: The algorithm maintains the current best solution (best_x, best_y).
# Candidate generation: Candidates are generated randomly within the search domain using a uniform distribution.
# Selection and replacement: The current best solution is replaced if a better candidate is found.
# Adaptation: The algorithm does not adapt during the search.
# Exploration mechanisms: The random candidate generation provides exploration throughout the domain.
# Exploitation mechanisms: The algorithm exploits by selecting the best solution seen so far.
# Boundary handling: Candidates outside the search domain are rejected.
# Budget strategy: The algorithm uses the full evaluation budget, generating a new candidate on each iteration.
# Closest known influences: This algorithm is a basic random search method, similar to a Monte Carlo approach.
# Novelty or unusual aspects: The algorithm is simple and straightforward, without any complex mechanisms.
# Failure modes: The algorithm may struggle on highly multimodal or deceptive functions, as it does not have any advanced exploration or exploitation strategies.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Generate a new candidate solution randomly within the search domain
            x = np.random.uniform(func.lower, func.upper, self.dim)

            # Evaluate the candidate and update the best solution if necessary
            y = func(x)
            if y < self.best_y:
                self.best_x = x
                self.best_y = y

        return self.best_x, self.best_y
