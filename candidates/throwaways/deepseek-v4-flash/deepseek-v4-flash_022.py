# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implementation of an Adaptive Differential Evolution (jDE) algorithm for black-box minimization.
# Search state: Population of NP candidate solutions, their fitness values, generation counter, and best-found solution. Each individual also maintains adaptive parameters F (scaling factor) and CR (crossover rate).
# Candidate generation: For each parent i, three distinct random indices r1,r2,r3 are chosen. Mutant vector v = x_r1 + F_i * (x_r2 - x_r3). Then binomial crossover with CR_i produces trial vector u.
# Selection and replacement: Greedy selection: trial replaces parent if its fitness is less (minimization) or equal (to avoid stagnation). If replacement occurs, F_i and CR_i are updated with the new candidate values.
# Adaptation: Each individual has its own F_i and CR_i, initialized uniformly in [0.1,1] and [0,1] respectively. At each generation, for each individual, new values F_i' and CR_i' are generated with small probabilities (tau1=0.1, tau2=0.1) from uniform distributions; otherwise they remain unchanged. The new values are used for mutation/crossover, and if the trial is accepted, they become the individual’s new parameters.
# Exploration mechanisms: Differential mutation explores directions between random individuals. High CR and F early promote exploration. The adaptation occasionally resets parameters to random values, maintaining diversity.
# Exploitation mechanisms: Selection pressure (greedy replacement) focuses on better solutions. Over generations, the population converges. Low CR and small F values produce exploitation steps.
# Boundary handling: Trial vector components outside bounds are reflected back into the domain. If reflection fails (unlikely), they are clipped to the bound.
# Budget strategy: Loop until budget exhausted. After each generation, check if budget remaining > 0. Best solution is tracked and returned exactly at budget exhaustion.
# Closest known influences: jDE (Brest et al., 2006), Differential Evolution (Storn & Price).
# Novelty or unusual aspects: Simple, no external dependencies beyond numpy. Uses individual-level parameter adaptation. Compact implementation.
# Failure modes: May converge prematurely on highly multimodal landscapes. Stagnation possible if population loses diversity too quickly. No restart mechanism included.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim

        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.array(b.lb, dtype=float)
            ub = np.array(b.ub, dtype=float)
        else:
            raise AttributeError("Objective function must provide either lower/upper or bounds.lb/bounds.ub")

        # Population size
        NP = max(4, int(4 + 3 * np.log(dim)))
        if NP > 100:
            NP = 100
        # Ensure budget is at least NP initial evaluations
        if budget < NP:
            NP = budget

        # jDE parameters
        tau1 = 0.1
        tau2 = 0.1
        fl = 0.1
        fu = 1.0

        # Initialize population and parameters
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.full(NP, np.inf)
        F = np.random.uniform(fl, fu, NP)
        CR = np.random.rand(NP)

        evals = 0
        best_x = None
        best_y = np.inf

        # Initial evaluation
        for i in range(NP):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()

        while evals < budget:
            # One generation
            for i in range(NP):
                # Choose three distinct random indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                r = np.random.choice(candidates, 3, replace=False)
                r1, r2, r3 = r[0], r[1], r[2]

                # Create new F and CR for this trial
                F_new = F[i]
                CR_new = CR[i]
                if np.random.rand() < tau1:
                    F_new = np.random.uniform(fl, fu)
                if np.random.rand() < tau2:
                    CR_new = np.random.rand()

                # Mutation: DE/rand/1
                v = pop[r1] + F_new * (pop[r2] - pop[r3])

                # Crossover: binomial with CR_new
                j_rand = np.random.randint(dim)
                u = np.empty(dim)
                for j in range(dim):
                    if np.random.rand() < CR_new or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]

                # Boundary handling: reflect, then clamp
                for j in range(dim):
                    if u[j] < lb[j]:
                        u[j] = lb[j] + (lb[j] - u[j])
                    elif u[j] > ub[j]:
                        u[j] = ub[j] - (u[j] - ub[j])
                    # Safety clamp
                    u[j] = max(lb[j], min(ub[j], u[j]))

                # Evaluate
                if evals >= budget:
                    break
                trial_fitness = func(u)
                evals += 1

                # Selection
                if trial_fitness <= fitness[i]:
                    pop[i] = u
                    fitness[i] = trial_fitness
                    F[i] = F_new
                    CR[i] = CR_new
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = u.copy()

            # If best improved, optionally do nothing else

        return best_x, best_y
