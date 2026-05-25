import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm is a compact and robust implementation of a random search method for
#          black-box optimization. It generates candidate solutions uniformly within the
#          provided bounds, evaluates them, and keeps track of the best solution found.
# Search state: The algorithm maintains the current best solution and its objective value.
# Candidate generation: Candidates are generated uniformly within the provided bounds.
# Selection and replacement: The best candidate replaces the current best solution if it
#                            has a lower objective value.
# Adaptation: There is no adaptation in this simple random search algorithm.
# Exploration mechanisms: Uniform random sampling provides exploration.
# Exploitation mechanisms: There are no explicit exploitation mechanisms.
# Boundary handling: Candidate solutions that fall outside the provided bounds are
#                    regenerated until they are within the bounds.
# Budget strategy: The algorithm uses the full evaluation budget at each call.
# Closest known influences: This is a basic random search algorithm, similar to those
#                           used in many optimization benchmarks.
# Novelty or unusual aspects: There are no particularly novel or unusual aspects to this
#                             simple random search implementation.
# Failure modes: The algorithm may struggle with highly multimodal or deceptive
#                objective functions, as it has no mechanism for escaping local optima.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Generate a candidate solution uniformly within the bounds
            x = np.random.uniform(func.lower, func.upper, self.dim)

            # Evaluate the candidate and update the best solution if it's better
            y = func(x)
            if y < self.best_y:
                self.best_x = x
                self.best_y = y

        return self.best_x, self.best_y
