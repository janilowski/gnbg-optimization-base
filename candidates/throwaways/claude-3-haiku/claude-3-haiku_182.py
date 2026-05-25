import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm is a simple randomized search that iteratively samples
#          new candidate solutions within the given bounds, keeping track of the
#          best solution found so far.
# Search state: The search state consists of the current best solution (best_x,
#               best_y) and the remaining evaluation budget.
# Candidate generation: New candidate solutions are sampled uniformly at random
#                      within the given bounds.
# Selection and replacement: The best candidate solution is compared to the
#                            current best, and replaces it if better.
# Adaptation: There is no adaptation mechanism in this algorithm.
# Exploration mechanisms: The algorithm relies on random sampling to explore the
#                         search space.
# Exploitation mechanisms: The algorithm exploits the best solution found so
#                          far by using it as the basis for new samples.
# Boundary handling: Candidate solutions that fall outside the given bounds are
#                    rejected and not evaluated.
# Budget strategy: The algorithm stops when the evaluation budget is exhausted.
# Closest known influences: This algorithm is a basic random search approach,
#                           similar to many simple optimization techniques.
# Novelty or unusual aspects: There are no novel or unusual aspects to this
#                             algorithm.
# Failure modes: The algorithm may struggle in high-dimensional problems or
#                problems with complex, multi-modal objective functions, as it
#                relies solely on random sampling without any more sophisticated
#                exploration or exploitation mechanisms.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_y = float('inf')

    def __call__(self, func):
        self.budget_remaining = self.budget
        
        # Read bounds from func.lower, func.upper or func.bounds.lb, func.bounds.ub
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            self.lower = np.array(func.lower)
            self.upper = np.array(func.upper)
        elif hasattr(func, 'bounds'):
            self.lower = np.array(func.bounds.lb)
            self.upper = np.array(func.bounds.ub)
        else:
            raise ValueError('Could not find function bounds')
        
        # Main optimization loop
        while self.budget_remaining > 0:
            # Generate a new candidate solution
            x = np.random.uniform(self.lower, self.upper, size=self.dim)
            
            # Evaluate the candidate solution
            y = func(x)
            self.budget_remaining -= 1
            
            # Update the best solution if necessary
            if y < self.best_y:
                self.best_x = x
                self.best_y = y
        
        return self.best_x, self.best_y
