import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1)-Evolution Strategy with step-size adaptation using Rechenberg's 1/5th rule.
#          Designed for black-box minimization within a fixed evaluation budget.
# Search state: Single current solution vector x and its objective value f(x),
#               plus a global step size sigma that adapts during the run.
# Candidate generation: Isotropic Gaussian mutation: y = x + sigma * N(0,I).
#                       Candidate is projected into the feasible domain if out of bounds.
# Selection and replacement: Deterministic (1+1) replacement: if f(y) < f(x), replace.
# Adaptation: Step size sigma is updated every generation using a success-based rule:
#             sigma *= exp((success - 1/5) / (dim+1)), where success=1 if improvement occurred,
#             else 0. This implements the 1/5th rule in a smooth exponential manner.
# Exploration mechanisms: Isotropic Gaussian mutation with adapting sigma provides exploration.
#                         When sigma is large, jumps are big; when it is small, local search.
# Exploitation mechanisms: The selection accepts only improvements, thus converges towards local
#                          optima. Step size adaptation tightens the search as progress stalls.
# Boundary handling: After mutation, candidate coordinates are clipped to [lower, upper].
# Budget strategy: The total number of objective evaluations is strictly limited to `budget`.
#                  The initial evaluation counts, and each trial candidate counts one evaluation.
# Closest known influences: Classic (1+1)-ES with Rechenberg's step-size control, as described in
#                           "Evolutionary Strategies" by Schwefel and Bäck.
# Novelty or unusual aspects: None; this is a standard, minimal ES implementation. No restart or
#                             population mechanisms. It relies on the harness to set the seed.
# Failure modes: Can get trapped in local optima. Performance degrades in high-dimensional
#                separable problems where isotropic mutation is inefficient. Step size may shrink
#                prematurely if the function is deceptive. No recovery mechanism (no restarts).
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function does not provide bounds via 'lower'/'upper' or 'bounds.lb'/'bounds.ub'")

        dim = self.dim
        budget = self.budget

        # Initial point uniformly in bounds
        x = lb + (ub - lb) * np.random.rand(dim)
        fx = func(x)
        evals = 1

        best_x = x.copy()
        best_y = fx

        # Initial step size: 20% of the range
        range_len = ub - lb
        sigma = 0.2 * np.min(range_len)  # isotropic step based on smallest range
        sigma = max(sigma, 1e-12)       # avoid zero sigma

        # Adaptation parameters
        target_success = 0.2            # 1/5
        damping = dim + 1.0

        while evals < budget:
            # Generate candidate with isotropic Gaussian mutation
            y = x + sigma * np.random.randn(dim)
            # Project onto feasible domain
            y = np.clip(y, lb, ub)
            fy = func(y)
            evals += 1

            # (1+1) selection: accept only improvement
            if fy < fx:
                x = y
                fx = fy
                success = 1.0
            else:
                success = 0.0

            # Update step size via smooth 1/5th rule
            sigma *= np.exp((success - target_success) / damping)

            # Clamp sigma to avoid extreme values
            sigma = np.clip(sigma, 1e-12, 0.5 * np.min(range_len))

            # Update global best
            if fx < best_y:
                best_y = fx
                best_x = x.copy()

        return best_x, best_y
