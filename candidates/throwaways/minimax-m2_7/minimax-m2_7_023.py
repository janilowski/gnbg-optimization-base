# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary:
#   The module implements a lightweight CMA-ES (Covariance Matrix Adaptation Evolution
#   Strategy) optimizer tailored for black-box minimization tasks. It maintains a
#   multivariate normal search distribution over the problem domain, adapts its mean,
#   covariance, and global step-size based on the relative success of sampled candidates,
#   and iteratively refines solutions until the evaluation budget is exhausted.
#
# Search state:
#   • Mean vector (m) – the center of the current search distribution.
#   • Step-size (σ) – overall scaling of the distribution.
#   • Covariance matrix (C) – captures pairwise correlations among variables.
#   • Evolution paths (pσ and pc) – internal accumulators that steer the adaptation of σ
#     and C, respectively.
#
# Candidate generation:
#   At each iteration λ (population size) samples are drawn from N(m, σ²·C). If bounds
#   are available they are enforced by clipping each sample to the feasible region.
#
# Selection and replacement:
#   The λ candidates are evaluated on the true objective. The μ best (lowest function
#   values) are retained and used to update the distribution parameters via weighted
#   recombination.
#
# Adaptation:
#   • Step-size σ is updated using the cumulative step-size adaptation (CSA) rule,
#     which shrinks σ when the evolution path pσ deviates from the expected length
#     under random selection.
#   • Covariance C is updated with both a rank‑one (pc pcᵀ) and a rank‑μ component,
#     employing learning rates c1 and cμ respectively. This blends long‑term
#     exploration directions with recent successful step information.
#
# Exploration mechanisms:
#   • Initial C is set to the identity, giving isotropic exploration.
#   • The population size λ grows logarithmically with dimensionality, providing a
#     broad sampling front.
#
# Exploitation mechanisms:
#   • Weighted recombination of the μ best points shifts the mean toward promising
#     regions.
#   • The covariance matrix gradually focuses the sampling on directions that have
#     proven beneficial, while σ contracts to refine solutions.
#
# Boundary handling:
#   After sampling, each candidate vector is clipped to the provided lower/upper
#   bounds. This prevents the optimizer from proposing solutions outside the
#   admissible domain.
#
# Budget strategy:
#   The algorithm never exceeds the supplied evaluation budget. When the remaining
#   budget is smaller than λ, a reduced batch is evaluated, and the main loop ends
#   as soon as the counter reaches the budget.
#
# Closest known influences:
#   The implementation follows the classic CMA-ES formulation by Hansen & Ostermeier
#   (2001) and borrows parameter settings from the original authors’ recommendations.
#
# Novelty or unusual aspects:
#   • Pure NumPy implementation – no external dependencies beyond the standard library.
#   • Uses a compact, single‑file class interface suitable for direct integration into
#     benchmark harnesses.
#   • Includes a simple diagonal jitter to keep the covariance matrix positive
#     definite in low‑budget scenarios.
#
# Failure modes:
#   • If the budget is too small, the optimizer may not have enough samples to learn
#     a useful covariance, effectively reducing to a random search.
#   • Highly non‑separable or noisy landscapes can impair the covariance adaptation,
#     especially when λ is set to a low value.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple CMA-ES optimizer for bound‑constrained black‑box minimization.

    The class follows the benchmark harness expectations:
        • __init__(self, budget, dim) → configure the optimizer.
        • __call__(self, func) → run optimization, returning (best_x, best_y).

    The optimizer respects the evaluation budget, never exceeding it.
    Bounds are read from either `func.lower` / `func.upper` or
    `func.bounds.lb` / `func.bounds.ub`.
    """

    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # Population size (λ) and number of parents (μ)
        self.lambda_ = int(4 + 3 * np.log(dim))
        self.mu = self.lambda_ // 2

        # Positive recombination weights (normalized)
        self.weights = np.zeros(self.mu)
        sum_w = 0.0
        for i in range(self.mu):
            w = np.log((self.lambda_ + 1) / 2.0) - np.log(i + 1)
            if w < 0.0:
                w = 0.0
            self.weights[i] = w
            sum_w += w
        if sum_w > 0.0:
            self.weights /= sum_w
        else:
            self.weights[:] = 1.0 / self.mu

        # ----- CMA-ES internal parameters -----
        # Cumulative step-size adaptation (CSA)
        self.cs = (self.mu + 2.0) / (self.dim + self.mu + 5.0)
        self.ds = 1.0 + self.cs + 2.0 * max(0.0, np.sqrt(self.mu / (self.dim + 1)) - 1.0)
        self.chiN = (
            np.sqrt(self.dim)
            * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))
        )

        # Covariance matrix adaptation
        self.cc = (4.0 + self.mu / self.dim) / (self.dim + 4.0 + 2.0 * self.mu / self.dim)
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mu)
        self.cmu = min(1.0 - self.c1,
                       2.0 * (self.mu - 2.0 + 1.0 / self.mu)
                       / ((self.dim + 2.0) + 2.0 * self.mu))

        # Internal state (to be set during a call)
        self.mean = None
        self.sigma = None
        self.C = None
        self.ps = None
        self.pc = None
        self.evals = 0
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        """
        Run the CMA-ES optimizer on the given black‑box function.

        Parameters
        ----------
        func : callable
            Objective function to be minimized. Must accept a 1‑D NumPy array of
            length `dim` and return a scalar.

        Returns
        -------
        best_x : np.ndarray
            Best solution found (vector of length `dim`).
        best_y : float
            Objective value at `best_x`.
        """
        # ----- Determine search bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError(
                "Cannot infer bounds: provide func.lower/func.upper or "
                "func.bounds.lb/func.bounds.ub"
            )
        if np.any(lower > upper):
            raise ValueError("Lower bounds must not exceed upper bounds")

        # ----- Initialize CMA-ES distribution -----
        # Mean uniformly inside the hyper‑rectangle
        self.mean = lower + (upper - lower) * np.random.rand(self.dim)

        # Initial step‑size (
