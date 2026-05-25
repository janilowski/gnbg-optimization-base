# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a basic Particle Swarm Optimization (PSO) algorithm for
#          minimizing a black‑box function under a fixed evaluation budget.
# Search state: The optimizer maintains a swarm of particles. Each particle stores its
#               current position, its personal‑best position and fitness, and its velocity.
#               A global best position and its fitness are tracked across the entire swarm.
# Candidate generation: New candidate positions are produced by updating each particle’s
#                      velocity using the inertia weight, the cognitive component (pull
#                      toward its personal best), and the social component (pull toward the
#                      global best). The updated velocity is added to the current position.
# Selection and replacement: After evaluating the fitness of a particle, its personal best
#                           is updated if the new position is better. The global best is
#                           updated whenever any personal best improves it. Particles retain
#                           their positions across iterations.
# Adaptation: The algorithm uses fixed hyper‑parameters (inertia weight, cognitive and
#             social coefficients). There is no dynamic adaptation of these parameters.
# Exploration mechanisms: Random components in the velocity update (uniform random numbers)
#                         and the diverse initial placement of particles provide exploration
#                         of the search space.
# Exploitation mechanisms: The social component pulls particles toward the best solution
#                          found so far, focusing search around promising regions.
# Boundary handling: After each position update, the particle’s coordinates are clipped to
#                    the problem’s lower and upper bounds, ensuring feasibility.
# Budget strategy: The swarm size is chosen proportionally to the dimensionality (with a
#                  lower bound) but capped by the remaining budget. The optimizer runs
#                  iteratively, evaluating a batch of particles each iteration until the
#                  total number of function evaluations reaches the budget, never exceeding it.
# Closest known influences: Standard PSO as introduced by Kennedy and Eberhart (2001).
# Novelty or unusual aspects: The implementation is deliberately simple, using minimal
#                             hyper‑parameters and a fixed inertia weight, making it easy to
#                             understand and adapt.
# Failure modes: If the budget is extremely low (e.g., 1–2 evaluations), the algorithm may
#                only sample a handful of random points, preventing a meaningful search. Fixed
#                hyper‑parameters may be suboptimal for highly multi‑modal or deceptive
#                landscapes, leading to slow convergence or entrapment in local minima.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple Particle Swarm Optimization (PSO) for black‑box minimization.

    The optimizer keeps a swarm of particles that explore the search space using
    velocity updates influenced by personal and global best positions. It respects
    the supplied evaluation budget and never exceeds it.
    """

    def __init__(self, budget, dim):
        """
        Initialize the optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        # Determine swarm size: at least 10 particles, scaled by dimension,
        # but never larger than the available budget (leaving at least one
        # evaluation for the final iteration).
        if self.budget <= 0:
            # No evaluations allowed – size is irrelevant.
            self.swarm_size = 1
        else:
            # Ensure at least one particle; cap by budget.
            self.swarm_size = max(1, min(self.budget, max(10, dim + 5)))

    def __call__(self, func):
        """
        Run PSO to minimize `func` within the evaluation budget.

        Parameters
        ----------
        func : callable
            Black‑box objective function. It must accept a 1‑D NumPy array of length `dim`
            and return a scalar (the function value).

        Returns
        -------
        best_x : ndarray
            Best solution found (vector of length `dim`).
        best_y : float
            Objective value at `best_x`.
        """
        budget = self.budget
        dim = self.dim

        # Handle zero or negative budget gracefully.
        if budget <= 0:
            # Return a dummy solution; no evaluation performed.
            return np.zeros(dim), np.inf

        # ------------------------------------------------------------------
        # Determine problem bounds (support two common interfaces).
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func, 'bounds.lb') and hasattr(func, 'bounds.ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: assume a typical search range of [-10, 10] for all variables.
            lb = np.full(dim, -10.0, dtype=float)
            ub = np.full(dim, 10.0, dtype=float)

        # Ensure bounds are arrays of length `dim` (scalar bounds are broadcast).
        if lb.ndim == 0:
            lb = np.full(dim, lb)
        if ub.ndim == 0:
            ub = np.full(dim, ub)

        # ------------------------------------------------------------------
        # Initialize swarm.
        # ------------------------------------------------------------------
        n = self.swarm_size
        # Initial positions – uniformly random within the bounds.
        x = np.random.uniform(lb, ub, size=(n, dim))
        # Initial velocities – zero (optional: could be small random values).
        v = np.zeros((n, dim))

        # Evaluate initial particles.
        y = np.empty(n, dtype=float)
        for i in range(n):
            y[i] = func(x[i])
        used = n

        # Personal bests.
        pbest_x = x.copy()
        pbest_y = y.copy()

        # Global best.
        idx_best = np.argmin(y)
        gbest_x = x[idx_best].copy()
        gbest_y = y[idx_best]

        # ------------------------------------------------------------------
        # PSO hyper‑parameters (fixed during the run).
        # ------------------------------------------------------------------
        w = 0.7       # inertia weight
        c1 = 1.5      # cognitive coefficient
        c2 = 1.5      # social coefficient

        # ------------------------------------------------------------------
        # Main optimization loop.
        # ------------------------------------------------------------------
        while used < budget:
            # Determine how many particles we can evaluate in this iteration.
            remaining = budget - used
            use_n = min(n, remaining)

            # Update velocities and positions for the first `use_n` particles.
            r1 = np.random.uniform(0.0, 1.0, size=(use_n, dim))
            r2 = np.random.uniform(0.0, 1.0, size=(use_n, dim))
            v[:use_n] = (w * v[:use_n]
                         + c1 * r1 * (pbest_x[:use_n] - x[:use_n])
                         + c2 * r2 * (gbest_x - x[:use_n]))
            x[:use_n] = x[:use_n] + v[:use_n]

            # Clip to feasible region.
            x[:use_n] = np.clip(x[:use_n], lb, ub)

            # Evaluate the chosen particles.
            for i in range(use_n):
                y_i = func(x[i])
                # Update personal best if improved (including equal for stability).
                if y_i <= pbest_y[i]:
                    pbest_x[i] = x[i].copy()
                    pbest_y[i] = y_i
                # Update global best if improved.
                if y_i <= gbest_y:
                    gbest_x = x[i].copy()
                    gbest_y = y_i

            used += use_n

        return gbest_x, gbest_y
