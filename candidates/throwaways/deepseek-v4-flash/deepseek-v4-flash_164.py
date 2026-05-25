import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a self-adaptive Differential Evolution (jDE) algorithm for continuous black-box minimization.
#          The algorithm adapts its crossover rate (CR) and scaling factor (F) per individual during the run,
#          using a simple probabilistic scheme. It is robust across dimensions and requires no parameter tuning.
# Search state: A population of NP candidate vectors, each with its own F and CR values, plus the global best solution.
# Candidate generation: DE/rand/1/bin mutation: for each target vector, three distinct vectors are chosen at random,
#                       and a mutant vector is created as base + F * (diff). Then binomial crossover with CR produces
#                       a trial vector.
# Selection and replacement: Greedy selection: the trial vector replaces the target if its fitness is better or equal.
# Adaptation: With probability tau (0.1), F is resampled from [0.1, 1.0] and CR from [0.0, 1.0] independently per individual.
#             This allows the algorithm to automatically adjust to the problem landscape.
# Exploration mechanisms: The mutation operator with difference vectors and the random adaptation of F/CR encourage
#                         exploration. The random choice of base and difference vectors keeps diversity.
# Exploitation mechanisms: Selection pressure from greedy replacement and the global best tracking guide the population
#                          toward promising regions. The adaptation can also produce small F/CR for fine-grained search.
# Boundary handling: Reflecting out-of-bounds components back into the domain.
# Budget strategy: An evaluation counter is maintained; the algorithm stops as soon as the budget is reached.
# Closest known influences: jDE (Brest et al., 2006, Zhang & Sanderson, 2009).
# Novelty or unusual aspects: None.
# Failure modes: May converge slowly on highly multimodal or deceptive functions; a fixed population size may be
#                suboptimal for extremely low or high budgets.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the optimizer.

        :param budget: Maximum number of function evaluations.
        :param dim: Dimensionality of the problem.
        """
        self.budget = budget
        self.dim = dim
        # Population size – logarithmically scaled with dimension, clamped to be at least 4.
        self.NP = max(4, int(4 + 3 * np.log(dim)))

    def __call__(self, func):
        """
        Run the optimization on the given function.

        :param func: Black-box function with .lower / .upper or .bounds.lb / .bounds.ub attributes.
        :return: (best_x, best_y) where best_y is the minimum value found.
        """
        # ---- extract bounds ----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
            ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)
        else:
            raise AttributeError("Function must provide bounds as .lower/.upper or .bounds.lb/.bounds.ub")

        dim = self.dim
        NP = self.NP

        # ---- initialise population ----
        pop = np.random.uniform(lb, ub, (NP, dim))
        fit = np.array([func(x) for x in pop])
        evals = NP

        # best so far
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # ---- initial F and CR per individual ----
        F = np.full(NP, 0.5)
        CR = np.full(NP, 0.9)

        # ---- main loop ----
        while evals < self.budget:
            for i in range(NP):
                if evals >= self.budget:
                    break

                # choose three distinct indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # mutation: DE/rand/1
                mutant = pop[a] + F[i] * (pop[b] - pop[c])

                # binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]

                # boundary reflection
                # reflect components that lie outside [lb, ub]
                below = trial < lb
                trial[below] = lb[below] + (lb[below] - trial[below])
                above = trial > ub
                trial[above] = ub[above] - (trial[above] - ub[above])
                # clamp any remaining (should be rare after reflection)
                trial = np.clip(trial, lb, ub)

                # evaluate trial
                trial_fit = func(trial)
                evals += 1

                # greedy selection
                if trial_fit <= fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

                # self-adaptation of F and CR
                if np.random.rand() < 0.1:
                    F[i] = 0.1 + 0.9 * np.random.rand()
                if np.random.rand() < 0.1:
                    CR[i] = np.random.rand()

        return best_x, best_y
