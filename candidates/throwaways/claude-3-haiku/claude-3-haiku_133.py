import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This implementation uses a random search algorithm to optimize the
#          objective function. It generates candidate solutions randomly within
#          the provided bounds, evaluates them, and keeps track of the best
#          solution found so far.
# Search state: The search state is represented by the current best solution
#               (best_x, best_y).
# Candidate generation: Candidates are generated randomly within the provided
#                      bounds using a uniform distribution.
# Selection and replacement: The best candidate solution is selected and
#                            replaces the current best solution if it improves
#                            the objective value.
# Adaptation: There is no adaptation mechanism in this implementation.
# Exploration mechanisms: The random generation of candidates promotes
#                         exploration of the search space.
# Exploitation mechanisms: The selection of the best candidate solution
#                          promotes exploitation of promising regions.
# Boundary handling: Candidate solutions that fall outside the provided bounds
#                    are regenerated within the bounds.
# Budget strategy: The algorithm simply evaluates candidates until the
#                  evaluation budget is exhausted.
# Closest known influences: This is a basic random search algorithm, which is
#                          a simple and robust optimization technique.
# Novelty or unusual aspects: There are no novel or unusual aspects to this
#                            implementation.
# Failure modes: The algorithm may struggle with highly multimodal or
#                deceptive objective functions, as it does not have
#                sophisticated exploration or exploitation mechanisms.
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
            # Generate a candidate solution randomly within the bounds
            if hasattr(func, 'bounds'):
                lb, ub = func.bounds.lb, func.bounds.ub
            else:
                lb, ub = func.lower, func.upper
            x = np.random.uniform(lb, ub, size=self.dim)
            y = func(x)
            evaluations += 1

            # Update the best solution if the candidate is better
            if y < self.best_y:
                self.best_x = x
                self.best_y = y

        return self.best_x, self.best_y
