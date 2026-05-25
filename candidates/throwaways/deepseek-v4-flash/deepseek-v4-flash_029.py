import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a self-adaptive Differential Evolution (jDE) variant for black-box minimization. It adapts the mutation scaling factor (F) and crossover rate (CR) per individual based on historical success.
# Search state: A population of NP candidate solutions in the continuous domain, each with associated F and CR parameters.
# Candidate generation: For each target individual, a mutant is generated using the DE/rand/1 scheme (difference of two random population members). Then binomial crossover with the target creates a trial vector.
# Selection and replacement: Greedy selection: if the trial vector's fitness is better or equal to the target's, it replaces the target. The individual's F and CR are also updated if the trial succeeds.
# Adaptation: F and CR are updated before each trial generation: with 10% probability, new F and CR are randomly sampled from uniform distributions; otherwise, they remain unchanged. Successful updates overwrite the individual's parameters; otherwise the old parameters are retained.
# Exploration mechanisms: Random mutation and crossover maintain diversity. The DE/rand/1 scheme encourages exploration.
# Exploitation mechanisms: Greedy selection pushes population toward better regions; self-adaptive parameters tune to the landscape.
# Boundary handling: Trial coordinates are clipped to the specified bounds.
# Budget strategy: The algorithm stops as soon as the evaluation budget is exhausted. Population size is chosen to balance budget and dimension: NP = max(5, min(10*dim, budget//5)), allowing multiple generations.
# Closest known influences: jDE (Brest et al., 2006), DE/rand/1/bin.
# Novelty or unusual aspects: None; straightforward implementation of jDE.
# Failure modes: May struggle on highly multimodal or deceptive landscapes due to limited population diversity. The fixed mutation scheme (rand/1) can be slow to converge on separable functions.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # population size: trade-off between diversity and generations
        self.NP = max(5, min(10 * dim, budget // 5))
        # initial F and CR for each individual (will be adapted)
        self.F = np.full(self.NP, 0.5)
        self.CR = np.full(self.NP, 0.9)
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        # ---------- read bounds ----------
        try:
            lb = func.lower
            ub = func.upper
        except AttributeError:
            lb = func.bounds.lb
            ub = func.bounds.ub
        lb = np.array(lb, dtype=np.float64)
        ub = np.array(ub, dtype=np.float64)
        if np.any(lb >= ub):
            raise ValueError("Invalid bounds")

        dim = self.dim
        NP = self.NP
        budget = self.budget
        evaluations = 0

        # ---------- initial population ----------
        pop = np.random.uniform(lb, ub, size=(NP, dim))
        fit = np.full(NP, np.inf)
        for i in range(NP):
            fit[i] = func(pop[i])
            evaluations += 1
            if fit[i] < self.best_y:
                self.best_y = fit[i]
                self.best_x = pop[i].copy()
            if evaluations >= budget:
                break

        # ---------- main jDE loop ----------
        while evaluations < budget:
            # iterate over all individuals (one generation)
            for i in range(NP):
                if evaluations >= budget:
                    break

                # --- self-adaptive F and CR (jDE) ---
                if np.random.rand() < 0.1:
                    F_i = np.random.uniform(0.1, 1.0)
                else:
                    F_i = self.F[i]
                if np.random.rand() < 0.1:
                    CR_i = np.random.rand()
                else:
                    CR_i = self.CR[i]

                # --- mutation: DE/rand/1 ---
                # select three distinct indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                r = np.random.choice(candidates, size=3, replace=False)
                r1, r2, r3 = r

                mutant = pop[r1] + F_i * (pop[r2] - pop[r3])

                # --- crossover: binomial ---
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                # --- boundary handling: clip ---
                trial = np.clip(trial, lb, ub)

                # --- evaluation ---
                trial_fit = func(trial)
                evaluations += 1
                if trial_fit < self.best_y:
                    self.best_y = trial_fit
                    self.best_x = trial.copy()

                # --- selection ---
                if trial_fit <= fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fit
                    # keep the successful F and CR
                    self.F[i] = F_i
                    self.CR[i] = CR_i

            # after each generation (or after loop exit)
            # optional: could add population reduction, but not needed

        return self.best_x, self.best_y
