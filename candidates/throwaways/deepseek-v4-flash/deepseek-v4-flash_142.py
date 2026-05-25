# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE) with rand/1/bin strategy. The algorithm maintains a population of candidate solutions, iteratively improves them via mutation, crossover, and greedy selection. Population size is chosen adaptively based on dimension and budget. All operations are vectorized with numpy.
# Search state: A population of NP vectors (shape (NP, dim)) and the best solution found so far. The fitness values of the population are stored in an array.
# Candidate generation: For each target vector, three distinct random vectors are selected from the population. A mutant vector is computed as base + F*(diff1-diff2). Binomial crossover then mixes the mutant with the target to produce a trial vector.
# Selection and replacement: The trial vector is evaluated; if it yields a lower objective value (minimization) than the target, it replaces the target in the population.
# Adaptation: Fixed parameters F=0.8 (mutation scaling) and CR=0.9 (crossover probability). No adaptive mechanisms.
# Exploration mechanisms: Mutation uses scaled differences between random population members, providing diversity. Crossover allows exchange of components between solutions.
# Exploitation mechanisms: Greedy selection (replace only if trial is better) drives the population toward better regions. Over generations, the population converges.
# Boundary handling: Each coordinate of the trial vector is clamped to the search bounds using numpy.clip.
# Budget strategy: Population size NP is computed in __init__ as a function of dimension and budget: NP = max(4, min(10*dim, budget//8)). This ensures at least 8 generations if budget allows. The algorithm runs full generations while remaining evaluations permit (each generation uses NP evaluations). Leftover evaluations (if any) are ignored.
# Closest known influences: Classic Differential Evolution (Storn and Price, 1997).
# Novelty or unusual aspects: None. Simple, clean implementation.
# Failure modes: May stagnate on non-separable or highly multimodal functions; fixed F and CR may be suboptimal for some problems; very small budgets may lead to insufficient exploration.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Differential Evolution (rand/1/bin) for black-box minimization.

    Parameters
    ----------
    budget : int
        Maximum number of objective function evaluations.
    dim : int
        Dimensionality of the search space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # Choose population size: ensure at least 4 (required for mutation)
        # and not too large relative to budget to allow several generations.
        # Typical DE rule: NP ~ 5-10*dim, but capped by budget.
        self.NP = max(4, min(10 * dim, budget // 8))
        # Mutation scale factor and crossover probability
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        # Determine search bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide lower/upper or bounds.lb/bounds.ub")

        dim = self.dim
        NP = self.NP
        F = self.F
        CR = self.CR

        # Ensure uniform vector lengths
        lb = lb.ravel()
        ub = ub.ravel()
        if len(lb) != dim or len(ub) != dim:
            raise ValueError("Bounds length does not match dimension")

        # Initialise population uniformly within bounds
        pop = np.random.uniform(lb, ub, size=(NP, dim))
        # Evaluate population
        pop_f = np.array([func(x) for x in pop])
        evals = NP  # initial evaluations

        # Track best solution
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_y = pop_f[best_idx]

        # Main loop: run full generations as long as enough budget remains
        while evals + NP <= self.budget:
            # For each target vector i
            for i in range(NP):
                # Generate three distinct random indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                # Mutation: v = x_a + F * (x_b - x_c)
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]  # ensure at least one component from mutant

                # Boundary handling: clip trial to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_f = func(trial)
                evals += 1

                # Selection
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    # Update best if improved
                    if trial_f < best_y:
                        best_y = trial_f
                        best_x = trial.copy()

        return best_x, best_y
