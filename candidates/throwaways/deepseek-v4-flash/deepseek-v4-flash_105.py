import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE/rand/1/bin) minimizer designed for
#   black-box benchmarks like GNBG. It uses a fixed population size, constant
#   differential weight and crossover probability, and a simple reflection
#   boundary handler. No adaptation or noise handling is included, keeping the
#   implementation simple and robust across dimensions.
# Search state: The state consists of a population of candidate vectors stored
#   in a 2D array (size NP x dim) and their corresponding objective values.
#   The best known solution (x, y) is tracked separately.
# Candidate generation: For each population member, a trial vector is created
#   using the DE/rand/1 scheme: base = x_r1 + F * (x_r2 - x_r3), where
#   r1, r2, r3 are distinct random indices different from the current index.
#   Binomial crossover with probability CR mixes the trial and the current
#   vector.
# Selection and replacement: Greedy selection: the trial replaces the current
#   member if its objective value is less than or equal to the current one
#   (minimization). This maintains a non-worsening population.
# Adaptation: None. Parameters F=0.8 and CR=0.9 are fixed.
# Exploration mechanisms: Mutation (differential vector) introduces diversity;
#   crossover (CR≈0.9) heavily recombines with the donor, favouring exploration
#   of new directions.
# Exploitation mechanisms: Selection pressure and the fact that the population
#   always retains the best found solutions promote convergence to promising
#   regions.
# Boundary handling: Clamping with reflection – a component that exceeds a
#   bound is reflected symmetrically back into the domain, preserving
#   dimension-wise feasibility.
# Budget strategy: Initial population size NP is set as
#   min(budget, max(5, min(20, 2*dim))). Then generations iterate until the
#   total number of function evaluations reaches exactly the given budget.
#   Each generation consumes NP evaluations (one per trial).
# Closest known influences: Classic DE (Storn & Price, 1997) with
#   DE/rand/1/bin strategy. No advanced variants like jDE or SHADE.
# Novelty or unusual aspects: Extremely simple, no adaptive components,
#   reflects rather than clamps or wraps boundaries, uses a very small
#   population size for low budgets.
# Failure modes: May stagnate on highly multimodal landscapes due to lack of
#   diversity preservation. Fixed parameters can be suboptimal for different
#   problem types. No noise handling makes it sensitive to stochastic
#   evaluations. Budget may be exhausted before convergence.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot find bounds: func.lower/upper or func.bounds.lb/ub required")
        dim = self.dim
        budget = self.budget

        # Population size: small to handle low budgets, but at least 5
        NP = max(5, min(20, 2 * dim))
        NP = min(NP, budget)  # cannot exceed total budget
        if NP < 5:
            NP = max(1, budget)  # extremely low budget: just use all available

        # Initialize population uniformly in bounds
        pop = np.random.uniform(lb, ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)

        # Evaluate initial population
        evals = 0
        for i in range(NP):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1

        # Track best so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # DE parameters
        F = 0.8
        CR = 0.9

        # Main loop: each generation consumes NP evaluations
        while evals < budget:
            # Shuffle order for randomness (optional, but helps)
            order = np.random.permutation(NP)
            for idx in order:
                if evals >= budget:
                    break

                # Select three distinct random indices, different from idx
                candidates = [j for j in range(NP) if j != idx]
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutation: base + F * (diff)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # Crossover: binomial
                cross_mask = np.random.rand(dim) < CR
                # Ensure at least one component is taken from mutant
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(0, dim)] = True
                trial = np.where(cross_mask, mutant, pop[idx])

                # Boundary handling: reflect
                # Compute reflection for components outside [lb, ub]
                lower_viol = trial < lb
                upper_viol = trial > ub
                trial[lower_viol] = 2 * lb[lower_viol] - trial[lower_viol]
                trial[upper_viol] = 2 * ub[upper_viol] - trial[upper_viol]
                # In case reflection still out of bounds (e.g., very narrow domain), clamp
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # Selection: replace if better or equal (non-greedy to preserve diversity)
                if trial_fitness <= fitness[idx]:
                    pop[idx] = trial
                    fitness[idx] = trial_fitness
                    # Update global best
                    if trial_fitness < best_y:
                        best_x = trial.copy()
                        best_y = trial_fitness

        return best_x, best_y
