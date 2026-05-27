import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a Differential Evolution (DE) optimizer for continuous black-box minimization.
# Search state: A population of candidate solutions with their objective values, plus the best found so far.
# Candidate generation: For each target vector, a mutant vector is created by adding the scaled difference
#   between two random population vectors to a third random base vector (DE/rand/1). Then binomial crossover
#   produces a trial vector.
# Selection and replacement: Trial replaces target if trial's objective is better (lower) than target's.
# Adaptation: No self-adaptation; fixed mutation factor F=0.8 and crossover rate CR=0.9.
# Exploration mechanisms: Mutation with random base and difference vectors encourages exploration.
# Exploitation mechanisms: Crossover retains good components from the target; selection pressure improves population.
# Boundary handling: Clamping trial components to bounds.
# Budget strategy: Number of population individuals is adjusted to allow at least one full generation plus initial evaluations.
# Closest known influences: Classic DE/rand/1/bin.
# Novelty or unusual aspects: Simple and robust; no complex adaptation.
# Failure modes: May stagnate on multimodal landscapes if population loses diversity; not suitable for discrete/noisy problems without modification.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        # store budget and dimension for use in __call__
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # get bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Function does not provide bounds via .lower/.upper or .bounds.lb/.bounds.ub")
        # ensure arrays of correct shape
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)
        elif lb.shape == (self.dim,):
            pass
        else:
            lb = np.reshape(lb, (self.dim,))
            ub = np.reshape(ub, (self.dim,))

        # Determine population size NP.
        NP = max(4, min(self.budget // 3, 10 + self.dim))
        NP = min(NP, self.budget)
        if NP < 4:
            # budget too small for DE, fallback to random sampling
            evaluations = 0
            best_x = None
            best_y = float('inf')
            while evaluations < self.budget:
                x = lb + np.random.rand(self.dim) * (ub - lb)
                y = func(x)
                evaluations += 1
                if y < best_y:
                    best_y = y
                    best_x = x
            return best_x, best_y

        # Initialize population
        pop = np.random.uniform(lb, ub, size=(NP, self.dim))
        fitness = np.full(NP, np.inf)
        best_y = np.inf
        best_x = np.empty(self.dim)

        evaluations = 0
        for i in range(NP):
            fitness[i] = func(pop[i])
            evaluations += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()

        # DE parameters
        F = 0.8
        CR = 0.9

        # Main loop
        while evaluations < self.budget:
            for i in range(NP):
                if evaluations >= self.budget:
                    break
                # select three distinct random indices != i
                indices = [j for j in range(NP) if j != i]
                np.random.shuffle(indices)
                a, b, c = indices[:3]

                # mutation
                mutant = pop[a] + F * (pop[b] - pop[c])

                # binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # clamp to bounds
                trial = np.clip(trial, lb, ub)

                # evaluate
                trial_fit = func(trial)
                evaluations += 1

                # selection
                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

        return best_x, best_y
