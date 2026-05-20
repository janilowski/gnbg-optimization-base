# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An extremely robust, compact, and fully adaptive implementation of Success-History Adaptive Differential Evolution with Linear Population Size Reduction (L-SHADE).
# Search state: Population of vectors, corresponding fitness values, a success-history memory of mutation scale factors (F) and crossover rates (Cr), and an archive of historical sub-optimal solutions.
# Candidate generation: Generates candidates using the "current-to-pbest/1" mutation strategy, leveraging historical archive vectors for diversity, followed by binomial crossover.
# Selection and replacement: Standard greedy DE selection where offspring replace parents only if they achieve a better or equal fitness value.
# Adaptation: Adapts mutation parameters (F, Cr) via successful parameter memories updated with a weighted Lehmer mean. Population size linearly decreases across the budget to focus search.
# Exploration mechanisms: High population diversity in early stages, archive-based mutations to maintain historical direction vectors, and Cauchy-distributed mutation steps.
# Exploitation mechanisms: Directional mutation towards the top-p best performing individuals, binomial crossover, and linear population reduction targeting convergence.
# Boundary handling: Points violating bounds are repaired using a midpoint contraction towards the parent vector to ensure feasible and high-quality solutions.
# Budget strategy: Monotonically tracks evaluations and caps population initialization / generation loops dynamically based on exact remaining budget.
# Closest known influences: L-SHADE (Tanabe and Fukunaga, 2014) and SHADE (Tanabe and Fukunaga, 2013).
# Novelty or unusual aspects: Simplified historical parameter memory and self-correcting selection fallback mechanism for extreme low-budget setups.
# Failure modes: Extremely low budgets (< 10 evaluations) limit the ability of the evolutionary process to adapt, defaulting to a center-biased random search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # 1. Retrieve bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            lb = np.zeros(self.dim) - 100.0
            ub = np.ones(self.dim) * 100.0
