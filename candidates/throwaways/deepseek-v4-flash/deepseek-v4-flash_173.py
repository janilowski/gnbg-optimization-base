# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant.
#          It maintains a Gaussian search distribution (mean and diagonal covariance)
#          and adapts the mean and step sizes based on the best solutions found in each
#          generation. The algorithm is designed for compactness and robustness across
#          dimensions, using standard numpy only.
# Search state: A mean vector (mu) and a per‑dimension standard deviation vector (sigma).
#               Also stores the best solution found so far and its objective value.
# Candidate generation: Each candidate is drawn from a multivariate normal distribution
#                       with mean mu and diagonal covariance diag(sigma**2). Samples are
#                       then clipped to the problem bounds.
# Selection and replacement: In each generation, all lambda candidates are evaluated.
#                            The top half (mu) are selected. The new mean is a weighted
#                            average of the selected candidates (equal weights). The new
#                            per‑dimension std is the standard deviation of the selected
#                            candidates, smoothed with the previous sigma.
# Adaptation: The mean is updated by a convex combination of the sample mean of the best
#             and the previous mean (smoothing factor = 0.8). The per‑dimension std is
#             updated analogously, also using the standard deviation of the best samples.
#             This allows the distribution to contract around promising regions.
# Exploration mechanisms: Large initial std (20% of the variable range) and stochastic
#                         sampling from a Gaussian distribution. The smoothing in the
#                         adaptation keeps the distribution wide enough to explore.
# Exploitation mechanisms: The mean shifts towards the best candidates, and the std
#                          shrinks as the population converges, focusing on the best area.
# Boundary handling: Samples that fall outside [lb, ub] are clipped to the bounds.
#                    This is simple and effective for most problems.
# Budget strategy: The budget is split into generations of size lambda = 30 (or fewer for
#                  the last generation). The distribution is updated after each generation
#                  that has at least 2 evaluations.
# Closest known influences: This algorithm is a simplified and diagonal‑only version of
#                           the Cross‑Entropy Method (CEM) for continuous optimisation.
#                           It also resembles a (mu, lambda)‑ES with per‑dimension step
#                           sizes derived from the selected sample variance.
# Novelty or unusual aspects: The adaptation uses a fixed smoothing factor and equal
#                             weights for selected candidates, which is very simple but
#                             works reasonably across many problems.
# Failure modes: On very high‑dimensional or highly multimodal landscapes, the
#                diagonal covariance may be insufficient and the algorithm may converge
#                prematurely. It also assumes the objective is reasonably smooth.
#                Very small budgets (< lambda) will only perform random search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Simple (mu, lambda)-ES with per-dimension step size adaptation."""

    def __init__(self, budget: int, dim: int):
        """
        Args:
            budget: Maximum number of function evaluations.
            dim: Dimensionality of the problem.
        """
        self.budget = budget
        self.dim = dim

        # Population size.  Use a fixed moderate size; avoid being too large.
        self.lambda_ = max(10, min(100, dim * 2))

        # Fraction of population to select (top half)
        self.mu_ = max(2, self.lambda_ // 2)

        # Smoothing factors for mean and std updates
        self.c_mean = 0.8
        self.c_std = 0.8

        # Minimum standard deviation to prevent collapse
        self.min_std = 1e-6

    def __call__(self, func):
        """
        Minimise func within the evaluation budget.

        Args:
            func: A callable that supports either
                  func.lower / func.upper (array-like) or
                  func.bounds.lb / func.bounds.ub (array-like).

        Returns:
            best_x: The best candidate found (numpy array).
            best_y: Its objective value (scalar).
        """
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds. Expecting func.lower/upper or func.bounds.lb/ub.")

        dim = self.dim

        # Initialise distribution: mean at centre, std = 20% of range
        mu = (lb + ub) / 2.0
        sigma = (ub - lb) * 0.2
        sigma = np.maximum(sigma, self.min_std)  # ensure >0

        # Best ever
        best_x = mu.copy()
        best_y = func(best_x)
        n_evals = 1

        # Main loop: consume budget in generations
        while n_evals < self.budget:
            # Determine how many candidates this generation
            remaining = self.budget - n_evals
            lam = min(self.lambda_, remaining)
            if lam < 1:
                break

            # Sample candidates
            candidates = np.random.normal(loc=mu, scale=sigma, size=(lam, dim))
            # Clip to bounds
            candidates = np.clip(candidates, lb, ub)

            # Evaluate
            values = np.empty(lam)
            for i in range(lam):
                values[i] = func(candidates[i])
            n_evals += lam

            # Update best ever
            idx_min = np.argmin(values)
            if values[idx_min] < best_y:
                best_y = values[idx_min]
                best_x = candidates[idx_min].copy()

            # Selection: keep top mu_ candidates (minimisation)
            if lam >= 2:
                # Partial sort: get indices of the best mu_ values
                # (use argpartition for efficiency, then sort the partition)
                mu_actual = min(self.mu_, lam - 1)  # at least 1 selected?
                # Actually we need at least 2 for std update
                if mu_actual < 2:
                    continue  # not enough points to update distribution meaningfully
                idx_sorted = np.argpartition(values, mu_actual - 1)[:mu_actual]
                # Sort within the selected for consistent mean/std computation
                idx_sorted = idx_sorted[np.argsort(values[idx_sorted])]

                selected = candidates[idx_sorted]

                # Update mean: average of selected
                new_mean = np.mean(selected, axis=0)

                # Update per-dimension std: std of selected along each dimension,
                # then smooth with previous sigma
                new_std = np.std(selected, axis=0, ddof=1)  # unbiased estimator
                # Avoid zero std if all selected are identical
                new_std = np.maximum(new_std, self.min_std)

                # Smoothing
                mu = (1 - self.c_mean) * mu + self.c_mean * new_mean
                sigma = (1 - self.c_std) * sigma + self.c_std * new_std
                sigma = np.maximum(sigma, self.min_std)

        return best_x, best_y
