# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free stochastic hill climbing algorithm with adaptive step sizes.
# Search state: Maintains a single 'best_x' point found so far and its fitness 'best_y'.
# Candidate generation: Generates new candidates by adding Gaussian noise to the current best_x, scaled by a step size sigma.
# Selection and replacement: Simple greedy acceptance; if a candidate is better than the current best_x, update best_x.
# Adaptation: Employs a 1/5th success rule: if a candidate is accepted, increase sigma; otherwise, decrease sigma to refine search.
# Exploration mechanisms: Initial sigma covers nearly the entire space; noise allows local escape.
# Exploitation mechanisms: Adaptive shrinkage of sigma focuses on local refinements of the best-found point.
# Boundary handling: Candidates are clipped to the function bounds using numpy.clip.
# Budget strategy: Iterative loop runs until exactly the specified budget of function evaluations is exhausted.
# Closest known influences: Adaptive Random Search / (1+1)-ES (Evolution Strategy).
# Novelty or unusual aspects: Minimalist implementation focusing on robustness in black-box scenarios without derivative info.
# Failure modes: Can get stuck in local optima; performance sensitive to initial sigma and local landscape ruggedness.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x = np.random.uniform
