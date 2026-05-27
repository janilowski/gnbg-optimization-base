import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a variant of the JADE (Adaptive Differential Evolution) algorithm
# for black-box minimization. JADE is a population‑based evolutionary algorithm that uses a
# current‑to‑pbest mutation scheme with an optional archive of inferior solutions, and adapts
# the mutation factor F and crossover rate CR during the run.
# Search state: The algorithm maintains a population of candidate solutions (array of shape (NP, dim))
# and their objective values, an external archive of discarded parents (also of size up to NP),
# and two running mean parameters mu_F and mu_CR for F and CR generation.
# Candidate generation: For each population member, a trial vector is created via
#   v = x_pbest + F * (x_r1 - x_r2_archive) + F * (x_r2 - x_r3)??? Wait, correct JADE uses
#   v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2) where x_r1 is from population, x_r2 is
#   from the union of population and archive. Here we use: v = x_pbest + F * (x_r1 - x_r2).
#   Then binomial crossover with rate CR (from a normal distribution with mean mu_CR) produces
#   the final trial.
# Selection and replacement: Greedy selection: if the trial is better (lower objective), it
#   replaces the parent, and the parent is moved to the archive. Otherwise the trial is discarded.
# Adaptation: The successful F and CR values from a generation are recorded. At the end of
#   each generation, mu_F is updated using the Lehmer mean of the successful F values, and
#   mu_CR is updated using the arithmetic mean. These means are then used to generate F and CR
#   for the next generation (F from a Cauchy distribution, CR from a normal distribution).
# Exploration mechanisms: The use of an archive of inferior solutions provides additional
#   diversity by allowing mutation to use a discarded parent as a donor. The scale factor F
#   (Cauchy distribution) can occasionally produce large steps, encouraging exploration.
# Exploitation mechanisms: The pbest selection (from the best p*100% of the population) biases
#   mutation toward promising regions. Additionally, CR adaptation learns which crossover
#   rates are successful, allowing more exploitation when beneficial.
# Boundary handling: After mutation, each coordinate is clipped to the lower/upper bounds.
#   Similarly, during crossover, if a component from the base vector is used and is outside
#   bounds, it is also clipped (though the base vector is always in bounds).
# Budget strategy: The initial population evaluation consumes NP function calls. Then the
#   main loop continues until the remaining budget is exhausted. Within each generation, trial
#   vectors are created one by one until the budget runs out (precise budget usage).
# Closest known influences: JADE (Zhang & Sanderson, 2009) – the original formulation with
#   pbest, archive, and parameter adaptation.
# Novelty or unusual aspects: None; this is a straightforward implementation of a well‑known
#   algorithm. The only minor choice is setting population size NP adaptively based on the
#   budget (NP = max(4, min(50, budget // 2))) to keep the implementation compact.
# Failure modes: High‑dimensional problems (e.g., 1000D) with very small budgets may not
#   allow the population to converge. The algorithm may also struggle on highly multimodal
#   landscapes if the population size is too small relative to the problem complexity.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """JADE optimizer for black‑box minimization.

    Parameters
    ----------
    budget : int
        Maximum number of function evaluations allowed.
    dim : int
        Dimensionality of the problem.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

        # Population size – scaled to budget for compactness, at least 4.
        self.NP = max(4, min(50, budget // 2))
        # Archive size equal to population size.
        self.archive_size = self.NP
        # pbest proportion
        self.p_best = 0.1
        # Learning rate for parameter adaptation
        self.c = 0.1

        # Parameter means
        self.mu_F = 0.5
        self.mu_CR = 0.5

        # RNG (numpy global random state, set by harness)
        self.rng = np.random

    def __call__(self, func):
        # Determine lower and upper bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)
        else:
            raise AttributeError("Cannot find problem bounds from func.lower/upper or func.bounds.lb/ub")

        dim = self.dim
        NP = self.NP
        budget = self.budget

        # Initialise population uniformly in bounds
        pop = self.rng.uniform(lb, ub, size=(NP, dim))
        f_pop = np.empty(NP)
        evals = 0
        for i in range(NP):
            f_pop[i] = func(pop[i])
            evals += 1

        # Best so far
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_y = f_pop[best_idx]

        # Archive (stores discarded parents)
        archive = np.empty((0, dim))

        # Main loop
        while evals < budget:
            # Number of trials we can still perform in this generation
            max_trials = min(NP, budget - evals)
            if max_trials <= 0:
                break

            # Lists to store successful parameters for adaptation
            success_F = []
            success_CR = []

            # Sort population indices by fitness (for pbest selection)
            order = np.argsort(f_pop)
            p_num = max(1, int(NP * self.p_best))  # number of best individuals considered

            for i in range(NP):
                if evals >= budget:
                    break

                # Current target vector
                x = pop[i]

                # Select pbest index among the best p_num individuals
                pbest_idx = order[self.rng.randint(p_num)]
                x_pbest = pop[pbest_idx]

                # Select distinct indices r1, r2 (different from i)
                # r1 from population
                r1 = i
                while r1 == i:
                    r1 = self.rng.randint(NP)
                # r2 from population ∪ archive (with equal probability)
                combined = np.vstack((pop, archive)) if archive.size > 0 else pop
                combined_size = combined.shape[0]
                r2_idx = self.rng.randint(combined_size)
                x_r2 = combined[r2_idx]
                # Ensure r2 is not the same as x_pbest or x? Not strictly required, but for diversity
                # We'll allow it – JADE does not forbid it.

                # Generate F and CR for this individual
                # F from Cauchy distribution with location mu_F, scale 0.1, then clip
                F = self.rng.standard_cauchy() * 0.1 + self.mu_F
                F = np.clip(F, 0.0, 1.0)
                # CR from normal distribution with mean mu_CR, std 0.1, clip to [0,1]
                CR = self.rng.normal(self.mu_CR, 0.1)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: v = x_pbest + F * (x_r1 - x_r2)
                v = x_pbest + F * (pop[r1] - x_r2)
                # Clip to bounds
                v = np.clip(v, lb, ub)

                # Binomial crossover
                j_rand = self.rng.randint(dim)
                trial = np.where(self.rng.rand(dim) < CR, v, x)
                # Ensure at least one component from v (if CR=0)
                trial[j_rand] = v[j_rand]

                # Evaluate trial
                f_trial = func(trial)
                evals += 1

                # Selection
                if f_trial < f_pop[i]:
                    # Success: replace parent, archive old parent
                    # Add parent to archive (if archive not full)
                    if archive.shape[0] < self.archive_size:
                        archive = np.vstack((archive, x.reshape(1, dim)))
                    else:
                        # Replace a random archive element
                        replace_idx = self.rng.randint(archive.shape[0])
                        archive[replace_idx] = x

                    pop[i] = trial
                    f_pop[i] = f_trial

                    # Record successful parameters
                    success_F.append(F)
                    success_CR.append(CR)

                    # Update global best
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()

            # Parameter adaptation based on successful parameters
            if success_F:
                # Lehmer mean for F
                sum_F = sum(success_F)
                sum_F_sq = sum(f**2 for f in success_F)
                if sum_F > 0:
                    self.mu_F = (1 - self.c) * self.mu_F + self.c * (sum_F_sq / sum_F)
                else:
                    self.mu_F = (1 - self.c) * self.mu_F
                # Arithmetic mean for CR
                self.mu_CR = (1 - self.c) * self.mu_CR + self.c * np.mean(success_CR)
            else:
                # If no successes, drift slightly toward 0.5
                self.mu_F = (1 - self.c) * self.mu_F + self.c * 0.5
                self.mu_CR = (1 - self.c) * self.mu_CR + self.c * 0.5

            # Limit archive size (remove random entries if too large)
            if archive.shape[0] > self.archive_size:
                indices = self.rng.choice(archive.shape[0], size=archive.shape[0] - self.archive_size, replace=False)
                archive = np.delete(archive, indices, axis=0)

        return best_x, best_y
