# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Particle Swarm Optimization (PSO) implementation for continuous black‑box minimization.
# Search state: The algorithm maintains a swarm of particles, each with a position, velocity, and personal best.
# Candidate generation: New candidates are produced by updating velocities based on personal and global best attractors, then adding the velocity to the current position, with clipping to the problem bounds.
# Selection and replacement: After evaluating a particle, we update its personal best if an improvement is found; the global best is also updated if the new fitness is the best seen so far.
# Adaptation: Fixed inertia weight (0.7) and acceleration coefficients (1.5) balance exploration and exploitation. Velocity is clamped to avoid excessive movement.
# Exploration mechanisms: Randomness in velocity updates and an initial swarm spread across the domain encourage exploration. Swarm size grows with dimension to cover the space.
# Exploitation mechanisms: Cognitive (personal best) and social (global best) terms steer particles toward promising regions.
# Boundary handling: Positions are clipped to the provided lower/upper limits after each update; velocities are limited to a maximum magnitude relative to the search range.
# Budget strategy: The budget is divided by the swarm size to run full PSO iterations; any leftover evaluations are spent on random samples, guaranteeing the budget is never exceeded.
# Closest known influences: Canonical PSO (Kennedy & Eberhart, 1995) with a global neighbourhood (gbest) topology and common default parameters.
# Novelty or unusual aspects: Swarm size scales with dimension but is capped to avoid excessive evaluations; residual budget is consumed by random sampling rather than early termination.
# Failure modes: With a very small budget (fewer evaluations than the swarm size), the algorithm may only explore a handful of random points and fail to converge. Fixed parameters may be suboptimal for highly deceptive landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple Particle Swarm Optimization (PSO) for black‑box minimization.
    Implements the required interface:
        __init__(budget, dim)
        __call__(func) -> (best_x, best_y)
    """
    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func) -> tuple[np.ndarray, float]:
        """
        Run PSO within the evaluation budget and return the best found solution.

        Parameters
        ----------
        func : callable
            Black‑box objective function. It must accept a 1‑D array_like of length `dim`
            and return a scalar (the objective value). Bounds are read from
            `func.lower` / `func.upper` or `func.bounds.lb` / `func.bounds.ub`.

        Returns
        -------
        best_x : np.ndarray
            Best candidate solution found (1‑D array of shape (dim,)).
        best_y : float
            Corresponding objective value.
        """
        # ------------------------------------------------------------------
        # Determine search space bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower)
            upper = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb)
            upper = np.asarray(func.bounds.ub)
        else:
            raise ValueError("func must expose 'lower'/'upper' or 'bounds.lb'/'bounds.ub'")

        # Ensure bounds are the correct length
        if lower.shape != (self.dim,) or upper.shape != (self.dim,):
            raise ValueError(
                f"Bounds dimension mismatch: expected ({self.dim},), "
                f"got lower={lower.shape}, upper={upper.shape}"
            )

        # ------------------------------------------------------------------
        # PSO hyper‑parameters (fixed for simplicity)
        # ------------------------------------------------------------------
        # Swarm size scales with dimension but is capped to avoid excessive evaluations.
        pop_size = min(20 + 2 * self.dim, 100)

        # Inertia, cognitive and social coefficients
        w = 0.7   # inertia weight
        c1 = 1.5  # cognitive coefficient
        c2 = 1.5  # social coefficient

        # Velocity limits (20% of the search range)
        search_range = upper - lower
        max_vel = search_range * 0.2

        # ------------------------------------------------------------------
        # Budget allocation
        # ------------------------------------------------------------------
        budget = self.budget
        max_full_iters = budget // pop_size          # number of full PSO cycles
        remaining_evals = budget - max_full_iters * pop_size  # leftover evaluations

        # ------------------------------------------------------------------
        # Initialize swarm
        # ------------------------------------------------------------------
        # Positions: uniform random in [lower, upper]
        x = np.random.uniform(lower, upper, size=(pop_size, self.dim))
        # Velocities: uniform random in [-max_vel, max_vel]
        v = np.random.uniform(-max_vel, max_vel, size=(pop_size, self.dim))

        # Personal best positions and fitness
        pbest_x = x.copy()
        pbest_y = np.full(pop_size, np.inf)

        # Global best
        gbest_x = None
        gbest_y = np.inf

        # ------------------------------------------------------------------
        # Initial evaluation
        # ------------------------------------------------------------------
        for i in range(pop_size):
            y = func(x[i])
            pbest_y[i] = y
            if y < gbest_y:
                gbest_y = y
                gbest_x = x[i].copy()

        # ------------------------------------------------------------------
        # Full PSO iterations
        # ------------------------------------------------------------------
        for _ in range(max_full_iters):
            for i in range(pop_size):
                # Random coefficients for velocity update
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Update velocity
                v[i] = (w * v[i]
                        + c1 * r1 * (pbest_x[i] - x[i])
                        + c2 * r2 * (gbest_x - x[i]))

                # Clip velocity
                v[i] = np.clip(v[i], -max_vel, max_vel)

                # Update position
                x[i] = np.clip(x[i] + v[i], lower, upper)

                # Evaluate fitness
                y = func(x[i])

                # Update personal best
                if y < pbest_y[i]:
                    pbest_y[i] = y
                    pbest_x[i] = x[i].copy()

                # Update global best
                if y < gbest_y:
                    gbest_y = y
                    gbest_x = x[i].copy()

        # ------------------------------------------------------------------
        # Consume any remaining budget with random sampling
        # ------------------------------------------------------------------
        for _ in range(remaining_evals):
            x_rand = np.random.uniform(lower, upper, size=self.dim)
            y = func(x_rand)
            if y < gbest_y:
                gbest_y = y
                gbest_x = x_rand.copy()

        # Ensure best_x is a proper numpy array
        if gbest_x is None:
            # Should never happen because budget >= 1
            gbest_x = np.empty(self.dim)

        return gbest_x, float(gbest_y)
