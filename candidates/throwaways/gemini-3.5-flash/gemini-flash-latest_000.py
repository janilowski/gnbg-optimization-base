# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Separable Natural Evolution Strategies (SNES) with back-projection boundary handling, dynamic population sizing, and periodic restarts. SNES maintains a mean vector and coordinate-wise standard deviations (step sizes), adapting them using natural gradients to minimize the objective function.
# Search state: Mean vector `mu`, coordinate-wise step sizes `sigma`, best found solution `best_x` and its fitness `best_y`, and current evaluation count.
# Candidate generation: Mutation vectors are sampled from a standard normal distribution, scaled by `sigma`, added to `mu`, and then clipped to the problem bounds. The search steps `z` are reconstructed from the clipped candidates to prevent divergence at boundaries.
# Selection and replacement: Candidates are evaluated and sorted by fitness. Recombination weights are computed using a logarithmic utility function that prioritizes top-performing candidates.
# Adaptation: The mean `mu` is updated towards the weighted average of successful steps, and the coordinate-wise step sizes `sigma` are scaled exponentially based on the natural gradient of the fitness distribution.
# Exploration mechanisms: Initial search starts with a wide coverage (step size equal to 1/4 of the bound range). Restarts are triggered when the step sizes converge to a threshold, introducing a new random mean and resetting step sizes to explore other regions of the design space.
# Exploitation mechanisms: Logarithmic weights focus updates on the best candidates in each generation, driving fast local convergence.
# Boundary handling: Candidates are clipped to the lower and upper bounds. The mutation step `z` is recomputed using the clipped candidate to ensure the covariance adaptation respects the boundaries and avoids "out-of-bounds" drift.
# Budget strategy: Strictly tracks evaluations to avoid exceeding the budget. The population size is adapted based on both dimension and budget to ensure sufficient generations can be run.
# Closest known influences: Separable Natural Evolution Strategies (SNES) by Schaul et al., and Covariance Matrix Adaptation Evolution Strategy (CMA-ES) by Hansen.
# Novelty or unusual aspects: Dynamic reconstruction of mutation steps from clipped candidates (back-projection) combined with a highly compact restart loop that operates under strict budget tracking.
# Failure modes: High-dimensional highly non-separable multimodal landscapes where coordinate-wise scaling is insufficient, though restarts mitigate local minima traps.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Retrieve bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Handle infinite or invalid bounds gracefully
        lb = np.where(np.isinf(lb), -1e2, lb)
        ub = np.where(np.isinf(ub), 1e2, ub)

        best_x = None
        best_y = float('inf')
        evals = 0

        # Dynamic population size
        pop_size = int(4 + int(3 * np.log(self.dim)))
        # Ensure we can run at least a few generations
        pop_size = max(4, min(pop_size, self.budget // 4))
        # If budget is extremely small, pop_size can be reduced
        pop_size = min(pop_size, self.budget)

        if pop_size <= 0:
            return None, None

        # Compute utility weights for recombination
        mu_w = pop_size // 2
        weights = np.zeros(pop_size)
        if mu_w > 0:
            raw_weights = np.log(mu_w + 0.5) - np.log(np.arange(1, mu_w + 1))
            weights[:mu_w] = raw_weights / np.sum(raw_weights)

        # Learning rates for SNES
        eta_mu = 1.0
        eta_sigma = (3.0 + np.log(self.dim)) / (5.0 * np.sqrt(self.dim))

        while evals < self.budget:
            # Initialize search state for a run/restart
            mu = lb + np.random.rand(self.dim) * (ub - lb)
            sigma = 0.25 * (ub - lb)
            sigma = np.where(sigma == 0.0, 1.0, sigma)

            # Local optimization loop (until restart triggered or budget exhausted)
            while evals < self.budget:
                # Check if we should restart due to convergence
                if np.max(sigma) < 1e-10 * np.min(ub - lb):
                    break

                # Adjust population size if remaining budget is small
                current_pop = min(pop_size, self.budget - evals)
                if current_pop <= 0:
                    break

                # Sample mutations
                z_raw = np.random.randn(current_pop, self.dim)
                candidates = mu + sigma * z_raw

                # Clip to bounds
                candidates_clipped = np.clip(candidates, lb, ub)

                # Back-project: compute actual mutations used after boundary constraint
                z = (candidates_clipped - mu) / sigma

                # Evaluate candidates
                scores = []
                for i in range(current_pop):
                    val = func(candidates_clipped[i])
                    evals += 1
                    scores.append(val)
                    if val < best_y:
                        best_y = val
                        best_x = np.copy(candidates_clipped[i])

                # Sort by scores
                sorted_idx = np.argsort(scores)

                # Compute gradients
                active_weights = weights[:current_pop]
                if np.sum(active_weights) > 0:
                    active_weights = active_weights / np.sum(active_weights)
                else:
                    active_weights = np.ones(current_pop) / current_pop

                grad_mu = np.zeros(self.dim)
                grad_sigma = np.zeros(self.dim)
                for rank, idx in enumerate(sorted_idx):
                    w = active_weights[rank]
                    grad_mu += w * z[idx]
                    grad_sigma += w * (z[idx]**2 - 1.0)

                # Update state
                mu += eta_mu * sigma * grad_mu
                sigma *= np.exp(0.5 * eta_sigma * grad_sigma)

                # Ensure sigma stays within reasonable bounds
                sigma = np.clip(sigma, 1e-15, 2.0 * (ub - lb))

        return best_x, best_y
