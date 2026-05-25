import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a compact implementation of a randomized direct search algorithm
#          for black-box optimization. It uses a simple random sampling approach
#          to explore the search space and track the best solution found so far.
# Search state: The algorithm maintains the current best solution and its objective value.
# Candidate generation: New candidate solutions are sampled randomly from the search space.
# Selection and replacement: The new candidate replaces the current best solution if it improves the objective.
# Adaptation: The algorithm does not adapt during the optimization process.
# Exploration mechanisms: Random sampling provides exploration of the search space.
# Exploitation mechanisms: The algorithm exploits the best solution found so far.
# Boundary handling: Candidate solutions are generated within the provided bounds.
# Budget strategy: The algorithm uses the full evaluation budget and stops when it is exhausted.
# Closest known influences: This implementation is a basic, randomized direct search method.
# Novelty or unusual aspects: The algorithm is compact, self-contained, and does not require any external dependencies.
# Failure modes: The algorithm may struggle on highly multimodal or deceptive objective functions.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        # Read function bounds
        if hasattr(func, 'bounds'):
            lb = func.bounds.lb
            ub = func.bounds.ub
        else:
            lb = func.lower
            ub = func.upper

        # Optimize
        for _ in range(self.budget):
            x = np.random.uniform(lb, ub, size=self.dim)
            y = func(x)
            if y < self.best_y:
                self.best_x = x
                self.best_y = y

        return self.best_x, self.best_y
