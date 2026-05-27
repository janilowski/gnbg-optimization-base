import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Particle Swarm Optimization (PSO) with constriction factor for black‑box minimization.
# Search state: A population of particles, each holding a position, velocity, personal best position,
#   and personal best value. The algorithm maintains a global best (best overall solution found).
# Candidate generation: Each particle’s velocity is updated via the standard PSO formula using
#   constriction factor χ, cognitive attraction to its personal best, and social attraction to the
#   global best. Position is updated by adding the velocity.
# Selection and replacement: After evaluating a new position, the particle’s personal best is updated
#   if the new objective value is lower. The global best is updated if any personal best improves.
# Adaptation: The constriction factor χ ≈ 0.7298 (with φ = 4.1) provides convergence by damping
#   velocities over time. No other adaptive mechanisms are used.
# Exploration mechanisms: Random velocity initialization, random acceleration coefficients (r1, r2)
#   in each dimension per particle, and the stochastic nature of the PSO update promote exploration.
# Exploitation mechanisms: Attraction toward the personal and global best positions, combined with
#   velocity damping, focuses search around promising regions.
# Boundary handling: Positions are clamped component‑wise to the specified lower/upper bounds.
#   Velocities are also clamped to a maximum velocity equal to 0.5× the domain width along each
#   dimension.
# Budget strategy: Population size is set as a function of dimension and budget
#   (NP = min(200, max(5, int(10 * sqrt(dim))), budget // 3}) to allow several generations.
#   The algorithm stops as soon as the evaluation counter reaches the budget.
# Closest known influences: Classical PSO (Kennedy & Eberhart, 1995) with constriction factor
#   (Clerc & Kennedy, 2002).
# Novelty or unusual aspects: None. A straightforward, robust implementation that works across
#   a moderate range of dimensions and budgets.
# Failure modes: Premature convergence on highly multimodal or deceptive landscapes, especially
#   with limited budget. May stagnate if the population collapses too quickly.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Particle Swarm Optimization for black‑box minimization."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from the objective function
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide lower/upper or bounds.lb/bounds.ub")

        # Ensure lb and ub are 1‑D arrays of length dim
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)
        lb = lb.astype(float)
        ub = ub.astype(float)

        D = self.dim
        budget = self.budget

        # Population size – balance dimension scaling and budget
        NP = max(5, int(10 * np.sqrt(D)))
        NP = min(NP, budget // 3)          # ensure at least 3 generations when possible
        NP = min(NP, 200)                  # hard upper bound for very high dimensions
        NP = min(NP, budget)               # never exceed budget (initial eval may equal NP)
        if NP < 5:
            NP = max(2, budget)            # extreme case: budget very small

        # PSO parameters
        c1 = 2.05
        c2 = 2.05
        phi = c1 + c2
        chi = 2.0 / abs(2.0 - phi - np.sqrt(phi*phi - 4.0*phi))  # ≈0.7298

        # Domain width and maximum velocity
        width = ub - lb
        vmax = 0.5 * width

        # Initialize population
        rng = np.random.default_rng()          # harness sets seed externally
        # Positions uniformly in bounds
        x = lb + rng.random((NP, D)) * width
        # Velocities scaled to half of vmax
        v = (rng.random((NP, D)) - 0.5) * vmax * 2.0

        # Personal bests
        pbest = x.copy()
        pbest_val = np.full(NP, np.inf)
        gbest = x[0].copy()
        gbest_val = np.inf

        # Evaluation counter
        evals = 0
        idx = 0
        while evals < budget and idx < NP:
            y = func(x[idx])
            evals += 1
            if y < pbest_val[idx]:
                pbest_val[idx] = y
                pbest[idx] = x[idx].copy()
            if y < gbest_val:
                gbest_val = y
                gbest = x[idx].copy()
            idx += 1

        # Main PSO loop
        while evals < budget:
            for i in range(NP):
                if evals >= budget:
                    break
                # Random vectors
                r1 = rng.random(D)
                r2 = rng.random(D)
                # Velocity update (constriction factor)
                v[i] = chi * (v[i] + c1 * r1 * (pbest[i] - x[i]) + c2 * r2 * (gbest - x[i]))
                # Clamp velocity
                v[i] = np.clip(v[i], -vmax, vmax)
                # Position update
                x[i] = x[i] + v[i]
                # Boundary handling: clip to bounds
                x[i] = np.clip(x[i], lb, ub)
                # Evaluate
                y = func(x[i])
                evals += 1
                # Update personal and global best
                if y < pbest_val[i]:
                    pbest_val[i] = y
                    pbest[i] = x[i].copy()
                if y < gbest_val:
                    gbest_val = y
                    gbest = x[i].copy()

        return (gbest, gbest_val)
