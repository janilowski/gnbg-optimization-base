# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a simple adaptive random‑search algorithm for black‑box minimization. It starts from a uniformly sampled point, then iteratively proposes new candidates by adding an isotropic Gaussian jitter to the current best solution. The step size (σ) is increased after successful improvements and decreased after failures, providing a lightweight exploration‑exploitation balance.
# Search state: The algorithm maintains the best solution found so far (best_x, best_y) and a scalar step size σ that controls the spread of Gaussian proposals.
# Candidate generation: Each iteration draws a random direction from a standard normal distribution, scales it by σ, and adds it to the current best point. The proposal is clipped into the feasible region defined by the bounds.
# Selection and replacement: If a candidate yields a lower objective value it replaces the stored best point; otherwise the best point remains unchanged.
# Adaptation: σ is multiplied by a factor >1 (default 1.2) after a successful move and by a factor <1 (default 0.8) after a failure, mimicking a simple (1+1)‑ES step‑size rule.
# Exploration mechanisms: Large σ encourages exploring distant regions; σ is bounded above by half the average bound range.
# Exploitation mechanisms: Small σ focuses sampling near the current best; σ is bounded below by a tiny value (1e‑8) to avoid complete stagnation.
# Boundary handling: Proposals are clipped element‑wise to the lower/upper bounds retrieved from the function object.
# Budget strategy: The algorithm counts each func(...) call and stops once the supplied evaluation budget is exhausted, guaranteeing no overrun.
# Closest known influences: The method is a simplified (1+1)‑ES with mutative step size, related to classic random search and simple evolution strategies.
# Novelty or unusual aspects: The step‑size adaptation is driven purely by immediate success/failure, without maintaining a population or covariance matrix, making the implementation compact and dimension‑agnostic.
# Failure modes: If σ shrinks to its minimum too quickly the algorithm may become trapped in a local optimum; the lower bound mitigates this but cannot guarantee global optimality.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Simple adaptive random search for black‑box minimization."""
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Retrieve problem bounds (supports func.lower/upper or func.bounds.lb/ub)
        lb, ub = self._extract_bounds(func, self.dim)

        # Initial step size σ (scaled to the problem range)
        sigma = np.mean(ub - lb) / 6.0

        # Sample initial point uniformly within bounds
        x = np.random.uniform(lb, ub)
        y = func(x)
        evals = 1
        best_x = x.copy()
        best_y = y

        # Adaptation parameters
        sigma_increase = 1.2
        sigma_decrease = 0.8
        max_sigma = np.mean(ub - lb) / 2.0
        min_sigma = 1e-8

        # Main search loop
        while evals < self.budget:
            # Generate candidate by isotropic Gaussian jitter around current best
            proposal = best_x + sigma * np.random.randn(self.dim)
            # Ensure proposal stays within bounds
            proposal = np.clip(proposal, lb, ub)

            # Evaluate candidate
            y_proposal = func(proposal)
            evals += 1

            # Update best solution and adapt σ
            if y_proposal < best_y:
                best_x = proposal.copy()
                best_y = y_proposal
                sigma = min(sigma * sigma_increase, max_sigma)
            else:
                sigma = max(sigma * sigma_decrease, min_sigma)

        return best_x, best_y

    @staticmethod
    def _extract_bounds(func, dim):
        # Try func.lower / func.upper
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        # Try func.bounds.lb / func.bounds.ub
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback default bounds [-10, 10] for all dimensions
            lb = np.full(dim, -10.0)
            ub = np.full(dim, 10.0)

        # Ensure 1‑D arrays of length dim
        if lb.ndim == 0:
            lb = np.full(dim, lb)
        if ub.ndim == 0:
            ub = np.full(dim, ub)

        return lb, ub
