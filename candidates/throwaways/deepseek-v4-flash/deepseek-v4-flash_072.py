import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: (1+1)-Evolution Strategy with self-adaptive step size for black-box minimization.
# Search state: One parent solution vector and its associated step size sigma.
# Candidate generation: Sample a new solution from isotropic Gaussian centered at parent with standard deviation sigma; simultaneously mutate sigma via log-normal distribution.
# Selection and replacement: The candidate replaces both the parent and sigma if its fitness is better (minimization); otherwise the parent is retained unchanged.
# Adaptation: Step size evolves through self-adaptation; no explicit covariance or learning rate adaptation.
# Exploration mechanisms: Large step sizes allow global exploration; random direction provides isotropic search.
# Exploitation mechanisms: Small step sizes refine the current best region; only accepted improvements move the search.
# Boundary handling: Candidate variables are clipped to the feasible bounds.
# Budget strategy: One function evaluation per iteration; stops exactly when the budget is reached.
# Closest known influences: Rechenberg's (1+1)-ES with self-adaptive mutation step size.
# Novelty or unusual aspects: None; straightforward implementation of a classic method.
# Failure modes: May converge slowly on ill‑conditioned or multimodal problems; initial sigma and adaptation rate affect robustness; can stagnate in high‑dimensional landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        else:
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)

        # Parameter settings
        tau = 1.0 / np.sqrt(self.dim)          # learning rate for sigma
        init_sigma = (ub - lb).mean() / 5.0    # initial step size

        # Initialize parent
        parent_x = lb + (ub - lb) * np.random.uniform(0, 1, self.dim)
        parent_sigma = init_sigma
        evals = 0

        # Evaluate initial parent
        parent_y = func(parent_x)
        evals += 1
        best_x = parent_x.copy()
        best_y = parent_y

        # Main loop
        while evals < self.budget:
            # Mutate sigma
            candidate_sigma = parent_sigma * np.exp(tau * np.random.randn())
            # Mutate x
            candidate_x = parent_x + candidate_sigma * np.random.randn(self.dim)
            # Clip to bounds
            candidate_x = np.clip(candidate_x, lb, ub)

            # Evaluate candidate
            candidate_y = func(candidate_x)
            evals += 1

            # Selection (minimization)
            if candidate_y < parent_y:
                parent_x, parent_sigma, parent_y = candidate_x, candidate_sigma, candidate_y
                if candidate_y < best_y:
                    best_x = candidate_x.copy()
                    best_y = candidate_y

        return best_x, best_y
