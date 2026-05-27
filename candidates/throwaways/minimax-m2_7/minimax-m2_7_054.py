# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a lightweight Particle Swarm Optimization (PSO)
# algorithm for bound‑constrained black‑box minimization.  The optimizer keeps a
# swarm of particles that iteratively update their positions and velocities,
# sharing information about the best solution discovered so far.
# Search state: Each particle holds its current position, velocity, personal best
# position and the associated objective value.  A global best position/value is
# maintained across the whole swarm.
# Candidate generation: A new candidate is produced by first updating a particle’s
# velocity as a weighted sum of its previous velocity, a cognitive term that pulls
# it toward its own personal best, and a social term that pulls it toward the
# current global best.  The position is then advanced by the new velocity.
# Selection and replacement: After evaluating the objective for every particle,
# any new personal best (lower objective) replaces the stored personal best.  If
# the new value is lower than the global best, the global best is also replaced.
# Adaptation: The inertia weight (w) and cognitive/social coefficients (c1, c2)
# are fixed (w = 0.7, c1 = 1.4, c2 = 1.4).  Velocity is clamped to a fraction of
# the bound range to avoid excessive jumps.  Swarm size scales with dimension
# (default = 4·dim, minimum 2) to keep the method robust across different
# problem sizes.
# Exploration mechanisms: Random components (r1, r2) in the velocity update,
# combined with velocity clamping, give a balanced exploration‑exploitation
# trade‑off.  Initial particles are sampled uniformly over the whole bound
# interval, further supporting exploration.
# Exploitation mechanisms: The social component attracts particles toward the
# current global best, focusing the search around promising regions.  Personal
# best memory retains each particle’s best known point, allowing the swarm to
# exploit good directions while still exploring.
# Boundary handling: After each position update, coordinates are clipped to the
# provided lower/upper bounds.  If no bounds are supplied, the search operates
# over the entire real line (subject to velocity clamping).
# Budget strategy: The algorithm counts each function evaluation and stops as
# soon as the budget is exhausted.  If the budget is smaller than the default
# swarm size, the swarm is reduced to the budget so that at least one evaluation
# can be performed.
# Closest known influences: The implementation follows the canonical PSO scheme
# introduced by Kennedy & Eberhart (1995) with the inertia weight term added by
# Shi & Eberhart (1998).
# Novelty or unusual aspects: The code is deliberately minimal, using only the
# Python standard library and NumPy, making it easy to embed in any benchmarking
# framework without external dependencies.
# Failure modes: With a very small budget (e.g., 1 evaluation) the algorithm
# returns only a single random point and cannot converge.  PSO may also get
# trapped in local optima on highly multi‑modal or deceptive landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Lightweight Particle Swarm Optimizer for bound‑constrained black‑box minimization.

    Parameters
    ----------
    budget : int
        Maximum number of objective function evaluations allowed.
    dim : int
        Dimensionality of the problem (number of variables).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Swarm size scales with dimension; at least 2 particles.
        self.swarm_size = max(2, 4 * dim)
        # Standard PSO coefficients
        self.w = 0.7    # inertia weight
        self.c1 = 1.4  # cognitive coefficient
        self.c2 = 1.4  # social coefficient

    def __call__(self, func):
        """
        Run PSO on the given objective function.

        Parameters
        ----------
        func : callable
            A function that accepts a 1‑D NumPy array of length ``dim`` and returns
            a scalar (the objective value).

        Returns
        -------
        best_x : np.ndarray
            The best (lowest) solution found.
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------------
        # 1. Determine search bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'lower'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # No bounds provided → assume unbounded space
            lb = np.full(self.dim, -np.inf)
            ub = np.full(self.dim, np.inf)

        # Ensure lb/ub are 1‑D arrays of length dim
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        # ------------------------------------------------------------------
        # 2. Initialise swarm
        # ------------------------------------------------------------------
        # Reduce swarm size if budget is smaller than the default swarm
        n_particles = min(self.swarm_size, self.budget)

        # Initialise particle positions uniformly within the bounds
        positions = np.random.uniform(lb, ub, size=(n_particles, self.dim))

        # Initialise velocities (random in a small fraction of the bound range)
        range_ = ub - lb
        # Replace infinite ranges with a finite fallback for velocity scaling
        range_[np.isinf(range_)] = 1.0
        velocities = np.random.uniform(-0.1 * range_, 0.1 * range_,
                                       size=(n_particles, self.dim))

        # ------------------------------------------------------------------
        # 3. Initial evaluation and bookkeeping
        # ------------------------------------------------------------------
        evals = 0
        personal_best_pos = positions.copy()
        personal_best_val = np.empty(n_particles)

        # Global best initialisation
        best_x = None
        best_y = np.inf

        for i in range(n_particles):
            y = func(positions[i])
            evals += 1
            personal_best_val[i] = y

            if y < best_y:
                best_x = positions[i].copy()
                best_y = y

            if evals >= self.budget:
                return best_x, best_y

        # ------------------------------------------------------------------
        # 4. PSO main loop
        # ------------------------------------------------------------------
        # Maximum number of full iterations we can afford
        remaining = self.budget - evals
        max_iter = remaining // n_particles

        for _ in range(max_iter):
            # Generate random matrices for velocity update
            r1 = np.random.uniform(0, 1, size=(n_particles, self.dim))
            r2 = np.random.uniform(0, 1, size=(n_particles, self.dim))

            # Update velocities
            velocities = (self.w * velocities
                          + self.c1 * r1 * (personal_best_pos - positions)
                          + self.c2 * r2 * (best_x - positions))

            # Clamp velocities to avoid excessive movement
            vel_limit = 0.2 * range_
            velocities = np.clip(velocities, -vel_limit, vel_limit)

            # Update positions
            positions = positions + velocities
            # Keep particles inside the bounds
            positions = np.clip(positions, lb, ub)

            # Evaluate each particle
            for i in range(n_particles):
                y = func(positions[i])
                evals += 1

                # Update personal best if improvement
                if y < personal_best_val[i]:
                    personal_best_val[i] = y
                    personal_best_pos[i] = positions[i].copy()

                    # Update global best if improvement
                    if y < best_y:
                        best_x = positions[i].copy()
                        best_y = y

                # Respect the budget exactly
                if evals >= self.budget:
                    return best_x, best_y

        # Return the best solution found
        return best_x, best_y
