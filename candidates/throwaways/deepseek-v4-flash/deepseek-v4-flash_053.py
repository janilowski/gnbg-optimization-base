import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE) with adaptive parameters for black-box minimization.
# Search state: Population of candidate solutions (vectors) and their fitness values. Best solution tracked.
# Candidate generation: For each target vector, a mutant vector is created using current-to-best/1 strategy:
#   mutant = best + F*(p1 - p2) + F*(p3 - p4) where best is the current best, p1,p2,p3,p4 are distinct individuals.
#   Scale factor F is drawn from a truncated Cauchy distribution per individual.
# Selection and replacement: Binomial crossover with probability Cr drawn from a normal distribution (truncated)
#   yields trial vector. If trial's fitness is better, it replaces the target in population.
# Adaptation: F and Cr are dynamically adapted based on success of generations. After each generation,
#   successful F and Cr values are stored and used to update the distribution parameters (location and scale)
#   for next generation via weighted averages.
# Exploration mechanisms: High initial diversity; mutation with random components; adaptive parameters ensure exploration when needed.
# Exploitation mechanisms: Attraction to the best solution via current-to-best; reduction in parameter variations as algorithm converges.
# Boundary handling: Reflecting mutated components that go out of bounds: if outside, reflect back inward to stay within bounds.
# Budget strategy: Evaluate only up to budget; stop when budget exhausted. Use a generation-based loop, but break if only few evaluations left.
# Closest known influences: Standard Differential Evolution (Storn & Price), JADE (Zhang & Sanderson) with optional adaptation.
# Novelty or unusual aspects: Use of Cauchy distribution for F scale, normal for Cr; simple yet robust adaptation.
# Failure modes: May stagnate on highly multimodal functions if population size too small; may prematurely converge if best solution gets stuck.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Population size: moderate, dimension-dependent
        self.np = max(10, 2 * dim)
        # Adaptation memory lengths
        self.memory_arc = self.np  # for storing successful parameter values
        self.c = 0.1  # adaptation rate

    def __call__(self, func):
        # Retrieve bounds
        try:
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        dim = self.dim
        np_ = self.np
        budget = self.budget
        evals = 0

        # Initialize population uniformly within bounds
        pop = lower + (upper - lower) * np.random.rand(np_, dim)
        fitness = np.full(np_, np.inf)
        for i in range(np_):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1

        # Sort and track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Parameter adaptation memories
        mu_F = 0.5
        mu_Cr = 0.5
        archive_F = []
        archive_Cr = []

        # Main loop
        while evals < budget:
            # Compute generation fitness to detect stagnation (optional, not used for adaptation)
            pop_new = np.empty_like(pop)
            fitness_new = np.empty_like(fitness)
            success_F = []
            success_Cr = []
            success_count = 0

            # For each target index, create trial
            for i in range(np_):
                if evals >= budget:
                    break

                # Choose distinct random indices for mutation (excluding i)
                candidates = list(range(np_))
                candidates.remove(i)
                if len(candidates) < 4:
                    # fallback: use random permutation with replacement
                    r = np.random.choice(np_, 4, replace=False)
                    while i in r:
                        r = np.random.choice(np_, 4, replace=False)
                    p1, p2, p3, p4 = r
                else:
                    r = np.random.choice(candidates, 4, replace=False)
                    p1, p2, p3, p4 = r

                # Generate F from truncated Cauchy (loc=mu_F, scale=0.1)
                F = np.random.standard_cauchy()
                F = mu_F + 0.1 * F
                F = np.clip(F, 0, 1)  # keep in [0,1]

                # Generate Cr from truncated normal (loc=mu_Cr, scale=0.1)
                Cr = np.random.randn() * 0.1 + mu_Cr
                Cr = np.clip(Cr, 0, 1)

                # Mutation: current-to-best/1
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[p1] - pop[p2]) + F * (pop[p3] - pop[p4])

                # Boundary handling by reflection
                for j in range(dim):
                    if mutant[j] < lower[j]:
                        mutant[j] = lower[j] + (lower[j] - mutant[j])
                        if mutant[j] > upper[j]:
                            mutant[j] = lower[j] + np.random.rand() * (upper[j] - lower[j])
                    elif mutant[j] > upper[j]:
                        mutant[j] = upper[j] - (mutant[j] - upper[j])
                        if mutant[j] < lower[j]:
                            mutant[j] = lower[j] + np.random.rand() * (upper[j] - lower[j])

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if np.random.rand() < Cr or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluate trial
                if evals >= budget:
                    break
                trial_fit = func(trial)
                evals += 1

                # Selection
                if trial_fit <= fitness[i]:  # allow equal to avoid stagnation
                    pop_new[success_count] = trial
                    fitness_new[success_count] = trial_fit
                    success_F.append(F)
                    success_Cr.append(Cr)
                    success_count += 1
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()
                else:
                    pop_new[success_count] = pop[i]
                    fitness_new[success_count] = fitness[i]
                    success_count += 1

            # Update population
            if success_count == np_:
                pop = pop_new
                fitness = fitness_new
            else:
                # In case of early termination within generation, leave rest unchanged
                pop[:success_count] = pop_new[:success_count]
                fitness[:success_count] = fitness_new[:success_count]

            # Parameter adaptation using successful F and Cr
            if len(success_F) > 0:
                # Weighted average of successful parameters
                successes = len(success_F)
                # Use arithmetic mean for this generation
                mean_F = np.mean(success_F)
                mean_Cr = np.mean(success_Cr)
                mu_F = (1 - self.c) * mu_F + self.c * mean_F
                mu_Cr = (1 - self.c) * mu_Cr + self.c * mean_Cr
                # Store in archive for potential Lehmer mean (simplified: just one generation)
                # We don't accumulate across generations; we use the moving average above.

        return best_x, best_y
