import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a Particle Swarm Optimization (PSO) algorithm for black-box minimization.
# Search state: Positions, velocities, personal best positions and values, global best position and value, and iteration counter.
# Candidate generation: Each particle's new position is current position plus velocity, where velocity is updated using inertia, cognitive (personal best), and social (global best) components.
# Selection and replacement: After evaluating a new position, if its fitness is better than the particle's personal best, the personal best is replaced; if it is better than the global best, the global best is updated.
# Adaptation: Inertia weight linearly decreases from 0.9 to 0.4 over the run to transition from exploration to exploitation. Acceleration coefficients are fixed at 2.0.
# Exploration mechanisms: Random initialization of positions and velocities, stochastic velocity update components, and high initial inertia encourage exploration.
# Exploitation mechanisms: Attraction to personal and global bests promotes convergence to promising regions; decreasing inertia shifts focus to exploitation later in the run.
# Boundary handling: Positions are clamped to the specified bounds after each update.
# Budget strategy: A swarm size is chosen based on the evaluation budget (max 50, min 2). The algorithm runs full swarm iterations until the budget is exhausted; any leftover evaluations (less than a full swarm) are ignored.
# Closest known influences: Standard Particle Swarm Optimization (PSO) with linearly decreasing inertia weight, as popularized by Shi and Eberhart.
# Novelty or unusual aspects: None; it is a straightforward, classical implementation.
# Failure modes: May stagnate in multimodal landscapes, especially with fixed acceleration coefficients; performance may degrade in high dimensions without adaptive parameters; clamping may cause particles to stick to boundaries.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the PSO algorithm.

        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState()  # will use seed set by harness

    def __call__(self, func):
        """
        Run the PSO algorithm on a given objective function.

        Parameters
        ----------
        func : callable
            The objective function to minimize. Must provide bounds via
            func.lower / func.upper or func.bounds.lb / func.bounds.ub.

        Returns
        -------
        tuple
            (best_x, best_y) where best_x is the best found solution (numpy array)
            and best_y is its corresponding function value (float).
        """
        # --- read bounds ---
        try:
            lb = func.lower
            ub = func.upper
        except AttributeError:
            try:
                lb = func.bounds.lb
                ub = func.bounds.ub
            except AttributeError:
                raise RuntimeError("Cannot read bounds from func.")
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)
        dim = self.dim
        range_vec = ub - lb

        # --- determine swarm size ---
        # S = max(2, min(50, budget // 20)). Ensures at least 2 particles.
        s = max(2, min(50, self.budget // 20))

        # --- initialize particles ---
        # positions uniformly random in the box
        pos = self.rng.uniform(lb, ub, size=(s, dim))
        # velocities small random values scaled by 1/4 of the range
        vel = (self.rng.uniform(-1.0, 1.0, size=(s, dim)) * 0.25) * range_vec

        # personal bests
        pbest_pos = pos.copy()
        pbest_val = np.full(s, np.inf)
        # global best
        gbest_pos = np.empty(dim)
        gbest_val = np.inf

        # --- evaluate initial swarm ---
        evals = 0
        for i in range(s):
            y = func(pos[i])
            evals += 1
            if y < pbest_val[i]:
                pbest_val[i] = y
                pbest_pos[i] = pos[i].copy()
            if y < gbest_val:
                gbest_val = y
                gbest_pos = pos[i].copy()

        # --- main PSO loop ---
        # compute maximum number of full swarm iterations we can do
        remaining = self.budget - evals
        max_iter = remaining // s if remaining > 0 else 0
        if max_iter > 0:
            # inertia linearly decreases from 0.9 to 0.4 over iterations
            w_start = 0.9
            w_end = 0.4
            c1 = 2.0
            c2 = 2.0

            for it in range(max_iter):
                w = w_start - (w_start - w_end) * (it / max_iter)
                # update each particle
                for i in range(s):
                    # random coefficients
                    r1 = self.rng.rand(dim)
                    r2 = self.rng.rand(dim)
                    # velocity update
                    vel[i] = (w * vel[i] +
                              c1 * r1 * (pbest_pos[i] - pos[i]) +
                              c2 * r2 * (gbest_pos - pos[i]))
                    # position update
                    pos[i] = pos[i] + vel[i]
                    # boundary clamp
                    pos[i] = np.clip(pos[i], lb, ub)
                    # evaluate
                    y = func(pos[i])
                    evals += 1
                    # update personal best
                    if y < pbest_val[i]:
                        pbest_val[i] = y
                        pbest_pos[i] = pos[i].copy()
                    # update global best
                    if y < gbest_val:
                        gbest_val = y
                        gbest_pos = pos[i].copy()

        # (any leftover evaluations are not used)
        return gbest_pos, gbest_val
