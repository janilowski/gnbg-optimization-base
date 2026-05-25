import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a simple implementation of a random search algorithm for
#          the GNBG black-box minimization benchmark. It generates candidate
#          solutions uniformly within the given bounds, evaluates them, and
#          keeps track of the best solution found so far.
# Search state: The current best solution (best_x, best_y) is maintained.
# Candidate generation: Candidates are generated uniformly within the given
#                       bounds.
# Selection and replacement: The best candidate replaces the current best
#                            solution if it improves the objective.
# Adaptation: There is no adaptation mechanism in this simple algorithm.
# Exploration mechanisms: The random search explores the space uniformly.
# Exploitation mechanisms: The algorithm exploits the current best solution
#                          by using it as a reference point for generating
#                          new candidates.
# Boundary handling: Candidates outside the given bounds are discarded.
# Budget strategy: The algorithm uses the full evaluation budget provided.
# Closest known influences: This is a basic random search algorithm, which
#                          is a common starting point for black-box
#                          optimization problems.
# Novelty or unusual aspects: There are no novel or unusual aspects to this
#                             implementation.
# Failure modes: The algorithm may struggle in high-dimensional problems or
#                problems with complex multi-modal landscapes, as it relies
#                solely on random exploration.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            # Generate a new candidate solution
            candidate_x = np.random.uniform(func.lower, func.upper, size=self.dim)

            # Evaluate the candidate solution
            candidate_y = func(candidate_x)
            evaluations += 1

            # Update the best solution if the candidate is better
            if candidate_y < self.best_y:
                self.best_x = candidate_x
                self.best_y = candidate_y

        return self.best_x, self.best_y
