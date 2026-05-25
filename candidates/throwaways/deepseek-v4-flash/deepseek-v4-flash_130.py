# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements (1+1)-Evolution Strategy with Cumulative Step-size Adaptation (CSA).
# Search state: A single candidate point (mean vector) and its objective value, plus step size sigma
# and an evolution path used to adapt sigma.
# Candidate generation: Offspring = mean + sigma * N(0,I) where N(0,I) is a standard normal vector.
# Selection and replacement: Offspring replaces the parent if it yields a lower objective value.
# Adaptation: Step size sigma is adapted based on the evolution path length:
#   - On success (offspring better), the path is updated toward the step direction and sigma may increase.
#   - On failure, the path decays and sigma may decrease.
#   The update uses a cumulative moving average with a learning rate close to 1/dim.
# Exploration mechanisms: Gaussian mutation with evolving step size. Initial step size is set to
# about 20% of the search space range. CSA allows the algorithm to increase sigma in early phases
# and reduce it when many failures occur.
# Exploitation mechanisms: Once a promising region is found, step size shrinks, leading to fine-grained search.
# Boundary handling: Reflective correction for each coordinate that exceeds bounds, which keeps the candidate
# inside the feasible domain without distorting the search direction too much.
# Budget strategy: The total budget includes the initial evaluation; each subsequent iteration uses one evaluation.
# Closest known influences: (1+1)-CMA-ES with CSA (without covariance matrix), often called (1+1)-ES with CSA.
# Novelty or unusual aspects: Very compact implementation, suitable for low to moderate dimensions.
# Failure modes: Can get stuck in early stagnation if step size shrinks too quickly; also may struggle
# on highly multi-modal or noisy functions where single parent is insufficient. Works best when the function
# is locally convex and smooth.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the algorithm with a given budget and dimension.
        Budget is the number of function evaluations allowed.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the algorithm on the provided function object.
        Returns (best_x, best_y) where best_x is a 1-D numpy array
        and best_y is a float (the minimal objective value found).
        """
        # Determine lower and upper bounds from func
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float).flatten()
            ub = np.array(func.upper, dtype=float).flatten()
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.array(func.bounds.lb, dtype=float).flatten()
            ub = np.array(func.bounds.ub, dtype=float).flatten()
        else:
            raise ValueError("Function object must provide bounds via .lower/.upper or .bounds.lb/.bounds.ub")

        # Ensure bounds are 1D arrays of proper dimension
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        if lb.shape[0] != self.dim or ub.shape[0] != self.dim:
            raise ValueError("Dimension mismatch between bounds and dim argument")

        # Initial step size: 20% of the average range
        sigma0 = 0.2 * np.mean(ub - lb)

        # Initialize mean (current best point) uniformly in the box
        mean = lb + (ub - lb) * np.random.uniform(0, 1, self.dim)

        # Evaluate initial point
        best_y = func(mean)
        best_x = mean.copy()
        evaluations = 1

        # CSA parameters (defaults for (1+1)-ES)
        # target success rate is about 0.27, learning rate about 1/dim
        c = 1.0 / self.dim          # learning rate for evolution path
        d = 1.0 + self.dim / 2.0    # damping factor for step size update
        p_success_target = 0.27
        p_threshold = 0.44          # threshold for success probability (used for sign computation)

        # Evolution path (cumulative path length)
        pc = np.zeros(self.dim)

        # Step size
        sigma = sigma0

        # Main loop
        while evaluations < self.budget:
            # Generate offspring
            z = np.random.normal(0, 1, self.dim)   # standard normal step
            offspring = mean + sigma * z

            # Reflective boundary handling
            for i in range(self.dim):
                if offspring[i] < lb[i]:
                    offspring[i] = 2 * lb[i] - offspring[i]
                    # If still out of bounds after reflection, clip
                    if offspring[i] > ub[i]:
                        offspring[i] = ub[i]
                elif offspring[i] > ub[i]:
                    offspring[i] = 2 * ub[i] - offspring[i]
                    if offspring[i] < lb[i]:
                        offspring[i] = lb[i]

            # Evaluate offspring
            offspring_y = func(offspring)
            evaluations += 1

            # Selection: keep the better one
            if offspring_y < best_y:
                # Success: update mean, best, and evolution path
                mean = offspring
                best_x = offspring.copy()
                best_y = offspring_y
                pc = (1 - c) * pc + np.sqrt(c * (2 - c)) * z
                success = True
            else:
                # Failure: mean unchanged, path decays
                pc = (1 - c) * pc
                success = False

            # Step size adaptation using CSA
            # Compute expected length of evolution path under random selection
            # For (1+1)-ES with p_success_target = 0.27, we use a simple rule:
            # sigma *= exp( (success_prob - p_success_target) / (d * p_threshold * (1-p_threshold)) )
            # where success_prob is estimated by a moving average? Not here; we use sign of pc length.
            # Simpler: use the cumulative step length adaptation (CSA) based on path length.
            # The standard update:
            # sigma *= exp( (norm(pc) - chiN) / (d * sqrt(c*(2-c)) ) )
            chiN = np.sqrt(self.dim) * (1 - 1.0/(4*self.dim) + 1.0/(21*self.dim**2))
            sigma *= np.exp((np.linalg.norm(pc) - chiN) /
                            (d * np.sqrt(c*(2-c))))
            # Ensure sigma does not become too small or too large (safety)
            sigma = np.clip(sigma, 1e-10, 10*sigma0)

            # Early termination if converged (optional, but budget is limit anyway)
            if sigma < 1e-12:
                break

        return best_x, best_y
