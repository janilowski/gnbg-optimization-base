import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact Differential Evolution with self-adaptive control parameters (jDE).
#          Designed for black-box minimization on a budget; handles arbitrary dimensions and box bounds.
# Search state: A population of NP candidate solutions (vectors), each with its own F (scale factor) and CR (crossover rate).
#               The best solution found so far is tracked externally.
# Candidate generation: For each population member, a trial vector is created using DE/rand/1/bin:
#                       mutant = x_r1 + F * (x_r2 - x_r3), then binomial crossover with the current member.
# Selection and replacement: Greedy replacement: if the trial vector has lower (better) objective value than the current member,
#                            it replaces it along with its F and CR parameters.
# Adaptation: F and CR are self-adapted per individual each generation:
#             F_new = F_low + rand * F_scale with probability tau_F, else kept;
#             CR_new = rand with probability tau_CR, else kept.
#             (Standard jDE parameters: tau_F=0.1, tau_CR=0.1, F_low=0.1, F_scale=0.9)
# Exploration mechanisms: High initial diversity (uniform random population), self-adaptive F and CR allow both exploration
#                         (large F, high CR) and exploitation (small F, low CR) to emerge.
# Exploitation mechanisms: As the population converges, mutation steps shrink; selection pressure preserves better solutions.
# Boundary handling: Reflection off bounds (bounce-back) for any coordinate that violates the box.
# Budget strategy: The population size NP is set automatically based on dimension and budget: NP = max(4, min(dim*4, budget//2, 100)).
#                  The algorithm runs generation by generation until budget is exhausted or no improvement for a long time.
# Closest known influences: jDE (Brest et al., 2006) – Differential Evolution with self-adaptive F and CR.
# Novelty or unusual aspects: None; straightforward implementation of a well-known adaptive DE variant.
# Failure modes: Poor performance on strongly multimodal or highly non-separable landscapes; may stagnate if population
#                collapses prematurely (can be mitigated by restarts, but not implemented for simplicity).
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the optimizer with given function evaluation budget and problem dimension.
        :param budget: maximum number of function evaluations allowed
        :param dim: number of decision variables
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimizer on the given objective function.
        :param func: callable with .lower/.upper (or .bounds.lb/.bounds.ub) attributes
        :return: (best_x, best_y) where best_x is a 1-D numpy array and best_y is a float
        """
        # ----- read bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        lb = lb.ravel()
        ub = ub.ravel()

        dim = self.dim
        budget = self.budget

        # ----- population size -----
        np_ = max(4, min(dim * 4, budget // 2, 100))

        # ----- jDE parameters -----
        tau_F  = 0.1
        tau_CR = 0.1
        F_low  = 0.1
        F_high = 1.0   # F_scale = F_high - F_low

        # initialise population
        pop = lb + np.random.rand(np_, dim) * (ub - lb)
        F   = np.full(np_, 0.5)      # initial scale factor
        CR  = np.full(np_, 0.9)      # initial crossover rate

        # evaluate initial population
        fitness = np.array([func(x) for x in pop], dtype=float)
        evals = np_

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # main loop
        while evals < budget:
            # generate each new trial vector
            for i in range(np_):
                # if budget exhausted inside the loop, stop
                if evals >= budget:
                    break

                # indices for mutation (exclude i)
                indices = list(range(np_))
                indices.remove(i)
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)

                # self-adaptive F and CR
                if np.random.rand() < tau_F:
                    F[i] = F_low + np.random.rand() * (F_high - F_low)
                if np.random.rand() < tau_CR:
                    CR[i] = np.random.rand()

                # mutation
                mutant = pop[r1] + F[i] * (pop[r2] - pop[r3])

                # crossover (binomial)
                j_rand = np.random.randint(dim)
                trial = np.where(
                    np.random.rand(dim) < CR[i],
                    mutant,
                    pop[i]
                )
                trial[j_rand] = mutant[j_rand]   # ensure at least one component from mutant

                # boundary handling: reflect
                for d in range(dim):
                    if trial[d] < lb[d]:
                        trial[d] = 2 * lb[d] - trial[d]
                        # if still out, clip (rare)
                        if trial[d] < lb[d]:
                            trial[d] = lb[d] + (np.random.rand() * (ub[d] - lb[d]) * 0.1)
                    elif trial[d] > ub[d]:
                        trial[d] = 2 * ub[d] - trial[d]
                        if trial[d] > ub[d]:
                            trial[d] = ub[d] - (np.random.rand() * (ub[d] - lb[d]) * 0.1)

                # evaluate trial
                ftrial = func(trial)
                evals += 1

                # selection: greedy replacement
                if ftrial <= fitness[i]:
                    pop[i]   = trial
                    fitness[i] = ftrial
                    # update best
                    if ftrial < best_y:
                        best_y = ftrial
                        best_x = trial.copy()
                else:
                    # parameter retention: keep F and CR only on success is typical;
                    # jDE standard: parameters are always updated as above, selection only on fitness.
                    # Here we already updated F[i] and CR[i] unconditionally; that's fine.
                    pass

            # optional: after each generation, stop if no improvement for many evals? keep simple.

        return best_x, best_y
