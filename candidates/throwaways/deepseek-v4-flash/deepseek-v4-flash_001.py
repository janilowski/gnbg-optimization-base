import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact self-adaptive differential evolution (jDE) variant for black-box
# minimisation. It evolves a population of candidate solutions with adaptive crossover
# and mutation rates.
# Search state: A population of size N, each with its own F (mutation factor) and CR
# (crossover rate), plus the global best solution found so far.
# Candidate generation: For each parent, a trial point is generated using DE/rand/1/bin
# mutation (difference vector from two random distinct members) and binomial crossover
# with the parent. The mutation factor and crossover rate are themselves adapted per
# individual using a probabilistic update scheme (jDE).
# Selection and replacement: Greedy selection: trial replaces parent if its objective
# value is better (minimisation). The best solution over the run is tracked separately.
# Adaptation: F and CR are updated independently for each individual: with small
# probabilities (tau_F, tau_CR) they are resampled uniformly from [0.1,0.9] and
# [0,1] respectively. Otherwise they are inherited unchanged.
# Exploration mechanisms: Differential mutation with adaptive F fosters exploration;
# the resampling of F and CR occasionally injects fresh exploratory behaviour.
# Exploitation mechanisms: Greedy selection preserves good solutions; as the population
# converges, mutation step sizes become smaller due to reduced diversity, promoting
# local refinement.
# Boundary handling: Reflected boundary correction: components that exceed the bounds
# are reflected back into the feasible domain.
# Budget strategy: Each generation uses exactly N evaluations (one per parent). The
# loop stops when the remaining budget is insufficient for a full generation; remaining
# evaluations are discarded.
# Closest known influences: jDE (Brest et al., 2006), a well-known self-adaptive DE.
# Novelty or unusual aspects: None; this is a straightforward implementation of jDE
# tailored to the constraints of the benchmark harness.
# Failure modes: Very low budgets may not allow the population to converge; high-
# dimensional problems may require many generations. The fixed population size scaling
# (4+3*log(dim)) works reasonably for moderate dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Population size: small, scales with dimension logarithmically
        self.pop_size = max(4, int(4 + 3 * np.log(dim)))
        # jDE parameters
        self.tau_F = 0.1   # probability to update F
        self.tau_CR = 0.1  # probability to update CR
        self.F_l = 0.1
        self.F_u = 0.9
        self.CR_l = 0.0
        self.CR_u = 1.0

    def _reflect(self, x, lb, ub):
        """Reflect out-of-bounds components back into [lb, ub]."""
        x = np.where(x < lb, 2 * lb - x, x)
        x = np.where(x > ub, 2 * ub - x, x)
        return np.clip(x, lb, ub)  # clip in case reflection overshoots

    def __call__(self, func):
        # Obtain bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.array(b.lb, dtype=float)
            ub = np.array(b.ub, dtype=float)
        else:
            raise AttributeError("Cannot read bounds from func")

        N = self.pop_size
        dim = self.dim
        budget = self.budget

        # Initialise population uniformly in bounds
        pop = np.random.uniform(lb, ub, size=(N, dim))
        # Evaluate initial population
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Initialise F and CR arrays
        F = np.random.uniform(self.F_l, self.F_u, size=N)
        CR = np.random.uniform(self.CR_l, self.CR_u, size=N)

        # Main loop: stop when less than one full generation left
        while budget >= N:
            new_pop = np.empty_like(pop)
            new_fitness = np.empty(N)

            for i in range(N):
                # Self-adapt F and CR
                if np.random.rand() < self.tau_F:
                    F[i] = self.F_l + np.random.rand() * (self.F_u - self.F_l)
                if np.random.rand() < self.tau_CR:
                    CR[i] = self.CR_l + np.random.rand() * (self.CR_u - self.CR_l)

                # Select three distinct random indices different from i
                candidates = list(range(N))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutant vector (DE/rand/1)
                mutant = pop[r1] + F[i] * (pop[r2] - pop[r3])

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.array([mutant[j] if np.random.rand() < CR[i] or j == j_rand
                                  else pop[i][j] for j in range(dim)])

                # Boundary handling: reflect
                trial = self._reflect(trial, lb, ub)

                # Evaluate trial
                trial_y = func(trial)
                budget -= 1

                # Greedy selection
                if trial_y <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]

            # Replace population
            pop = new_pop
            fitness = new_fitness

        return best_x, best_y
