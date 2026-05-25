import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (mu+lambda) Evolution Strategy with isotropic step-size adaptation using Rechenberg's 1/5 rule.
# Search state: Population of mu candidate solutions and a global step-size sigma.
# Candidate generation: Each offspring is created by copying a randomly selected parent from the current population and adding Gaussian noise with standard deviation sigma.
# Selection and replacement: (mu+lambda) truncation – combine parents and offspring, then keep the mu individuals with the smallest function values.
# Adaptation: After each generation, the step-size sigma is adapted based on the fraction of offspring that improved over the median parent fitness of that generation. sigma = sigma * exp( (success_rate - 0.2) * damping ).
# Exploration mechanisms: Gaussian mutation with step-size dynamically controlled; population diversity helps exploration.
# Exploitation mechanisms: Selection pressure from keeping only the best mu individuals; step-size reduction when success rate < 1/5.
# Boundary handling: Candidate solutions are clipped to the search bounds.
# Budget strategy: Budget is consumed in two phases: mu evaluations for initial population, then lambda evaluations per generation. The algorithm stops as soon as the remaining budget is insufficient for a full generation (but does not waste leftover evaluations).
# Closest known influences: Classic (mu+lambda)-ES with 1/5-rule, as commonly used in BBOB/COCO benchmarks.
# Novelty or unusual aspects: Uses a simple isotropic step-size adaptation (1/5 rule) rather than per-dimension or full CMA; very compact and robust across dimensions.
# Failure modes: May struggle with ill-conditioned or strongly non-separable problems due to isotropic mutations; can converge prematurely on multi-modal landscapes if step-size shrinks too fast.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # algorithm parameters
        self.mu = 10                      # population size (parents)
        self.lambda_ = 20                 # offspring per generation
        self.damping = 1.0                # adaptation damping
        self.target_success = 0.2         # Rechenberg target

    def __call__(self, func):
        # --- read bounds ---
        if hasattr(func, 'lower') and func.lower is not None:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and func.bounds is not None:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot find bounds from function object")

        dim = self.dim
        budget_remaining = self.budget
        mu = self.mu
        lambda_ = self.lambda_

        # initial step-size as fraction of problem range
        sigma = 0.2 * np.mean(ub - lb)

        # --- initialisation ---
        pop = np.random.uniform(lb, ub, size=(mu, dim))
        pop_f = np.full(mu, np.inf)
        for i in range(mu):
            pop_f[i] = func(pop[i])
            budget_remaining -= 1
            if budget_remaining <= 0:
                # budget exhausted immediately – unlikely but handle
                best_idx = np.argmin(pop_f)
                return pop[best_idx].copy(), pop_f[best_idx]

        best_x = pop[np.argmin(pop_f)].copy()
        best_y = np.min(pop_f)

        # --- evolution loop ---
        while budget_remaining >= lambda_:
            # generate offspring
            offspring = np.empty((lambda_, dim))
            offspring_f = np.full(lambda_, np.inf)

            for i in range(lambda_):
                # select a random parent
                parent = pop[np.random.randint(mu)]
                # mutation
                mutant = parent + np.random.normal(0, sigma, size=dim)
                # boundary clipping
                mutant = np.clip(mutant, lb, ub)
                offspring[i] = mutant

            # evaluate offspring, one by one
            for i in range(lambda_):
                offspring_f[i] = func(offspring[i])
                budget_remaining -= 1

            # --- selection: (mu+lambda) truncation ---
            combined = np.vstack((pop, offspring))
            combined_f = np.concatenate((pop_f, offspring_f))
            sorted_idx = np.argsort(combined_f)
            pop = combined[sorted_idx[:mu]].copy()
            pop_f = combined_f[sorted_idx[:mu]].copy()

            # update best
            if pop_f[0] < best_y:
                best_x = pop[0].copy()
                best_y = pop_f[0]

            # --- step-size adaptation (1/5 rule) ---
            # compute success rate as fraction of offspring that improved over the median parent fitness of the previous generation
            # we use the median of the parent population before selection (pop_f before update was already replaced, we need previous generation's parent median)
            # Instead, use the median of the offspring's own parent's (randomly chosen) – simpler: success = fraction of offspring better than the worst parent after selection? 
            # We'll use: fraction of offspring that are better than the median parent fitness of the combined set? 
            # Use median of previous parent population (before this generation's selection). We saved old pop_f before selection? We'll compute before selecting.
            # But after selection we lost old parents. Let's compute success rate as the proportion of offspring that have fitness less than the median of the parent population before selection.
            # We don't have old pop_f after we updated pop_f. So we need to capture old median before update.
            # Alternative: use success fraction of offspring that made it into the new population (i.e., number of offspring among the top mu). That's simpler.
            # Number of offspring in new pop:
            num_offspring_selected = sum(1 for idx in sorted_idx if idx >= mu and idx < mu+lambda_)  # because combined[:mu] is parents, combined[mu:mu+lambda] is offspring
            success_rate = num_offspring_selected / lambda_

            # apply 1/5 rule
            sigma = sigma * np.exp((success_rate - self.target_success) * self.damping)

        return best_x, best_y
