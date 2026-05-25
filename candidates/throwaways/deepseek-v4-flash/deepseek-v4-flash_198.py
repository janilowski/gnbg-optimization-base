import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1)-Evolution Strategy with Cumulative Step Size Adaptation (CSA-ES)
#          for black-box minimization. The algorithm maintains a single current
#          solution (mean) and an isotropic step size sigma. At each iteration it
#          generates one offspring by adding a multivariate normal perturbation,
#          clips to the domain bounds, evaluates it, and replaces the parent if
#          the offspring is better. The step size is adapted based on the evolution
#          path (cumulative sum of successful steps) to maintain a balance between
#          exploration and exploitation.
# Search state: A single parent candidate (vector x) and a step size scalar sigma.
#              Additionally, an evolution path p_sigma (vector of size dim) is
#              maintained for step size adaptation.
# Candidate generation: Offspring = parent + sigma * standard_normal(dim).
#                       Bounds are enforced by clipping after generation.
# Selection and replacement: Deterministic truncation – the offspring replaces the
#                            parent if its objective value is strictly less than
#                            the parent's current value (minimization).
# Adaptation: At every replacement event, the evolution path is updated according to
#             p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)) * step,
#             where step = (offspring - parent)/sigma. Then the step size is updated
#             as sigma *= exp((c_sigma/d_sigma)*(||p_sigma||/E||N(0,I)|| - 1)),
#             with E||N(0,I)|| approximated by sqrt(dim).
# Exploration mechanisms: Random Gaussian perturbations with amplitude controlled
#                         by the adapted sigma. The CSA mechanism prevents premature
#                         convergence of the step size.
# Exploitation mechanisms: The search is centered around the current best solution;
#                          only one candidate is evaluated per generation, focusing
#                          computational resources on the region of interest.
# Boundary handling: All candidate points are clipped component-wise to the
#                    domain bounds [lower, upper] after perturbation.
# Budget strategy: The algorithm terminates immediately when the evaluation count
#                  reaches the budget allocated in __init__.
# Closest known influences: Classic (1+1)-ES with cumulative step size adaptation
#                           (Hansen & Ostermeier, 2001; Beyer & Sendhoff, 2017).
# Novelty or unusual aspects: The implementation is deliberately compact and uses
#                             only a single parent and a single offspring, making it
#                             lightweight and easy to follow.
# Failure modes: On rugged, multimodal landscapes the algorithm may converge to a
#                local optimum. In very high dimensions, the isotropic step size
#                can become inefficient, and the simple CSA parameter setting
#                may not be optimal. The lack of recombination limits its
#                ability to exploit population diversity.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        """
        Initialize the (1+1)-ES with given evaluation budget and dimensionality.

        Parameters:
        budget: int – maximum number of objective function evaluations.
        dim: int – search space dimension.
        """
        self.budget = budget
        self.dim = dim
        # Random generator (the harness sets the global seed before each run)
        self.rng = np.random.default_rng()

    def __call__(self, func):
        """
        Run the algorithm on the given objective function.

        Parameters:
        func – an object that implements:
               - lower and upper attributes (either directly or via func.bounds)
               - __call__(x) returning a scalar (minimization)

        Returns:
        (best_x, best_y) – tuple of the best solution found and its value.
        """
        # --- Extract bounds -------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
            lower, upper = lb, ub
        else:
            raise AttributeError("The provided function does not expose lower/upper bounds.")

        # Ensure arrays are 1D of correct dimension
        if lower.ndim == 0:
            lower = np.full(self.dim, lower)
        if upper.ndim == 0:
            upper = np.full(self.dim, upper)
        lower = lower.astype(float).ravel()
        upper = upper.astype(float).ravel()
        assert lower.shape == (self.dim,) and upper.shape == (self.dim,)

        # --- Parameter settings ---------------------------------------------
        # Initial step size: about 20% of the mean range width
        range_width = upper - lower
        sigma0 = 0.2 * np.mean(range_width)  # scalar isotropic step size
        # Learning rates for cumulative step size adaptation (CSA)
        # c_sigma ~ 1/sqrt(dim) is a common heuristic
        c_sigma = 1.0 / np.sqrt(self.dim)
        # d_sigma controls the damping; typical value ~ 1 + sqrt(dim)
        d_sigma = 1.0 + np.sqrt(self.dim)
        expected_norm = np.sqrt(self.dim)  # E[||N(0,I)||] approximation

        # --- Initialization -------------------------------------------------
        # Start at a uniformly random point within bounds
        x = lower + self.rng.uniform(0, 1, size=self.dim) * (upper - lower)
        # Evaluate initial point
        y = func(x)
        evaluations = 1

        # Store best solution found so far
        best_x = x.copy()
        best_y = y

        # Evolution path (initially zero)
        p_sigma = np.zeros(self.dim)

        # --- Main loop ------------------------------------------------------
        while evaluations < self.budget:
            # Generate offspring: isotropic Gaussian perturbation
            z = self.rng.standard_normal(self.dim)  # standard normal vector
            offspring = x + sigma0 * z
            # Clip to bounds
            offspring = np.clip(offspring, lower, upper)

            # Evaluate offspring
            y_off = func(offspring)
            evaluations += 1

            # Update best overall
            if y_off < best_y:
                best_x = offspring.copy()
                best_y = y_off

            # Selection: replace parent if offspring is strictly better
            if y_off < y:
                # Successful step: update evolution path
                # The 'step' is (offspring - x) / sigma0
                step = z  # because offspring - x = sigma0 * z
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * step

                # Update step size using cumulative path length
                norm_p = np.linalg.norm(p_sigma)
                sigma0 = sigma0 * np.exp((c_sigma / d_sigma) * (norm_p / expected_norm - 1))

                # Replace parent
                x = offspring
                y = y_off
            else:
                # Unsuccessful step: update path only (no replacement)
                # In standard (1+1)-CSA, the path is updated only on success.
                # No change to x, y, but sigma may still be adapted later? 
                # Actually typical (1+1)-CSA updates sigma only on success,
                # but some variants also update on failure with a damping factor.
                # For simplicity we keep the standard rule: no path update on failure.
                pass

        return best_x, best_y
