import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a standard particle swarm optimization (PSO) algorithm 
# with constriction coefficients for black-box minimization.
# Search state: A swarm of particles, each with position, velocity, personal 
# best position and its value, and a global best position and value.
# Candidate generation: Each particle updates its velocity by combining its 
# previous velocity (with constriction), personal best direction, and global 
# best direction, then adds this velocity to its current position.
# Selection and replacement: Each particle replaces its personal best if the 
# new position yields a lower function value. Global best is updated whenever 
# any particle finds a new best.
# Adaptation: The constriction factor (chi=0.7298) and acceleration constants 
# (c1=c2=2.05) are fixed, ensuring convergence without additional adaptation.
# Exploration mechanisms: Random numbers (r1, r2) in velocity update provide 
# stochastic exploration. Initial positions and velocities are uniformly 
# random in the feasible region.
# Exploitation mechanisms: Particles are attracted towards their own personal 
# best and the global best, focusing search in promising areas.
# Boundary handling: Positions are clipped to the variable bounds after 
# updating. Velocities are also clamped to a fraction (0.2) of the variable 
# range to prevent excessive oscillations.
# Budget strategy: The swarm size is capped by the budget to avoid overshoot. 
# The algorithm stops immediately when the evaluation count reaches the budget.
# Closest known influences: Standard PSO with constriction factor, as 
# described by Clerc and Kennedy (2002).
# Novelty or unusual aspects: None – straightforward implementation.
# Failure modes: May converge prematurely to local optima, especially in 
# high‑dimensional multimodal problems given limited budget. No restart 
# mechanism.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine variable bounds from the provided function object.
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.array(func.lower, dtype=float)
            upper = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lower = np.array(func.bounds.lb, dtype=float)
            upper = np.array(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot read bounds from func; "
                             "expected func.lower/upper or func.bounds.lb/ub")

        # ----- PSO parameter settings (constriction factor) -----
        chi = 0.7298               # constriction factor (Clerc & Kennedy)
        c1 = 2.05                  # cognitive acceleration
        c2 = 2.05                  # social acceleration

        # Swarm size: at least 1, at most 2*dim, but never exceed the budget.
        swarm_size = max(1, min(2 * self.dim, self.budget))

        # Velocity bound: 20 % of each dimension's range.
        v_max = 0.2 * (upper - lower)

        # ----- Initialise population -----
        # Positions: uniformly random inside bounds.
        pos = lower + (upper - lower) * np.random.rand(swarm_size, self.dim)
        # Velocities: uniformly random in [-v_max, v_max].
        vel = -v_max + 2 * v_max * np.random.rand(swarm_size, self.dim)

        # Evaluate initial population.
        pbest = pos.copy()
        pbest_val = np.array([func(pos[i]) for i in range(swarm_size)])
        evals = swarm_size

        # Identify global best.
        gbest_idx = np.argmin(pbest_val)
        gbest = pbest[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]

        # If budget is already exhausted, return immediately.
        if evals >= self.budget:
            return gbest, gbest_val

        # ----- PSO iterations -----
        # We will iterate over particles one by one, checking the budget
        # before every single evaluation.
        while evals < self.budget:
            for i in range(swarm_size):
                # Generate two independent random vectors.
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Velocity update with constriction and clamping.
                vel[i] = chi * (vel[i] + c1 * r1 * (pbest[i] - pos[i])
                                       + c2 * r2 * (gbest - pos[i]))
                vel[i] = np.clip(vel[i], -v_max, v_max)

                # Position update.
                pos[i] = pos[i] + vel[i]
                pos[i] = np.clip(pos[i], lower, upper)

                # Evaluate the new position.
                val = func(pos[i])
                evals += 1

                # Update personal best.
                if val < pbest_val[i]:
                    pbest[i] = pos[i].copy()
                    pbest_val[i] = val

                # Update global best.
                if val < gbest_val:
                    gbest = pos[i].copy()
                    gbest_val = val

                # Stop if we have exhausted the budget.
                if evals >= self.budget:
                    break

        return gbest, gbest_val
