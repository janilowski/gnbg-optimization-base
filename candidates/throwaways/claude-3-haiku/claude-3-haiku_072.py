import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a compact implementation of a simple randomized search algorithm
#          for black-box function optimization. It generates random candidate
#          solutions within the function's bounded domain, evaluates them, and
#          keeps track of the best solution found so far.
# Search state: The algorithm maintains the current best solution (best_x, best_y).
# Candidate generation: Candidate solutions are generated randomly within the
#                       function's bounded domain.
# Selection and replacement: The best candidate solution found so far replaces
#                            the current best solution if it improves the
#                            objective function value.
# Adaptation: The algorithm does not adapt or learn from previous iterations.
# Exploration mechanisms: The random candidate generation provides a
#                         broad exploration of the search space.
# Exploitation mechanisms: The algorithm focuses on the current best solution,
#                          gradually improving it.
# Boundary handling: Candidate solutions that fall outside the bounded domain
#                    are discarded and a new candidate is generated.
# Budget strategy: The algorithm stops when the evaluation budget is exhausted.
# Closest known influences: This algorithm is a basic randomized search
#                          approach, similar to random sampling or random
#                          restarts of a local optimizer.
# Novelty or unusual aspects: The algorithm is designed to be compact and
#                             robust across dimensions, without any
#                             specialized techniques.
# Failure modes: The algorithm may struggle with high-dimensional problems
#                or problems with highly multimodal or deceptive objective
#                functions, where random search is inefficient.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            x = self._generate_candidate(func)
            y = func(x)
            self.evaluations += 1
            if y < self.best_y:
                self.best_x = x
                self.best_y = y

        return self.best_x, self.best_y

    def _generate_candidate(self, func):
        if hasattr(func, 'bounds'):
            lb, ub = func.bounds.lb, func.bounds.ub
        else:
            lb, ub = func.lower, func.upper

        candidate = lb + np.random.rand(self.dim) * (ub - lb)
        return candidate
