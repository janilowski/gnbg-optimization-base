# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Particle Swarm Optimization (PSO) with linearly decreasing inertia weight.
# Search state: population of particles, each with position, velocity, personal best position and fitness.
# Candidate generation: new positions are computed by updating velocity using cognitive (personal best) and social (global best) components, then applying clamped bounds.
# Selection and replacement: after evaluation, each particle updates its personal best if the new fitness is better; the global best is updated after evaluating the whole population.
# Adaptation: inertia weight linearly decreases from 0.9 to 0.4 over the course of the run.
# Exploration mechanisms: random cognitive and social coefficients (each uniformly sampled per dimension) combined with the decreasing inertia encourage initial exploration.
# Exploitation mechanisms: the global best attracts all particles, and the decreasing inertia weight shifts focus toward exploitation in later iterations.
# Boundary handling: positions are clipped to the search domain; if a particle is clipped, its velocity is set to zero to prevent immediate re‑exit.
# Budget strategy: one generation (all particles evaluated) per iteration until the evaluation budget is reached; the loop checks remaining budget before evaluating each particle.
# Closest known influences: canonical PSO (Kennedy & Eberhart, 1995) with inertia weight (Shi & Eberhart, 1998).
# Novelty or unusual aspects: none – the implementation is a straightforward, compact PSO tailored for a black‑box benchmark.
# Failure modes: may struggle with highly multimodal or deceptive landscapes due to premature convergence; fixed population size may be suboptimal for very low or very high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from the function object (flexible naming)
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot read bounds from func: need lower/upper or bounds.lb/ub")

        # Ensure bounds are 1‑D arrays
        if lb.ndim == 0:
            lb = np.array([lb])
            ub = np.array([ub])
        lb = lb.ravel()
        ub = ub.ravel()

        d = self.dim
        # Population size: at least 4, at most budget/2, scaling with dimension
        pop_size = max(4, min(self.budget // 2, 10 * d))
        # Clamp to budget (need at least 1 evaluation per particle)
        if pop_size > self.budget:
            pop_size = self.budget
        if pop_size < 1:
            pop_size = 1

        # PSO parameters
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0   # cognitive coefficient
        c2 = 2.0   # social coefficient

        # Initialize population uniformly in bounds
        positions = np.random.uniform(lb, ub, size=(pop_size, d))
        velocities = np.random.uniform(-(ub - lb), (ub - lb), size=(pop_size, d)) * 0.1

        # Evaluate initial population
        evaluations = 0
        personal_best_positions = positions.copy()
        personal_best_fitness = np.full(pop_size, np.inf)
        global_best_pos = None
        global_best_fit = np.inf

        for i in range(pop_size):
            if evaluations >= self.budget:
                break
            # Evaluate
            x = positions[i].copy()
            # Ensure within bounds (just in case, though initialisation should be fine)
            x = np.clip(x, lb, ub)
            y = func(x)
            evaluations += 1
            personal_best_fitness[i] = y
            personal_best_positions[i] = x.copy()
            if y < global_best_fit:
                global_best_fit = y
                global_best_pos = x.copy()

        # Main PSO loop
        generation = 0
        while evaluations < self.budget:
            w = w_start - (w_start - w_end) * (evaluations / self.budget)

            # Generate new positions for all particles (but evaluate one by one with budget check)
            for i in range(pop_size):
                if evaluations >= self.budget:
                    break
                # Random coefficients per dimension (common PSO variant)
                r1 = np.random.rand(d)
                r2 = np.random.rand(d)

                # Velocity update
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (personal_best_positions[i] - positions[i])
                                 + c2 * r2 * (global_best_pos - positions[i]))

                # Position update
                new_pos = positions[i] + velocities[i]

                # Boundary handling: clamp and zero velocity if clamped
                for j in range(d):
                    if new_pos[j] < lb[j]:
                        new_pos[j] = lb[j]
                        velocities[i, j] = 0.0
                    elif new_pos[j] > ub[j]:
                        new_pos[j] = ub[j]
                        velocities[i, j] = 0.0

                positions[i] = new_pos

                # Evaluate
                y = func(new_pos)
                evaluations += 1

                # Update personal best
                if y < personal_best_fitness[i]:
                    personal_best_fitness[i] = y
                    personal_best_positions[i] = new_pos.copy()
                    if y < global_best_fit:
                        global_best_fit = y
                        global_best_pos = new_pos.copy()

            generation += 1

        # Return best found solution
        return global_best_pos, global_best_fit
