import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact black-box minimization algorithm based on self‑adaptive differential evolution (jDE).
# Search state: Population of real‑valued vectors with associated objective values and adaptive parameters (F and CR).
# Candidate generation: Standard DE/rand/1 mutation with binomial crossover.
# Selection and replacement: Greedy selection keeping the better vector.
# Adaptation: jDE adaptation: each individual's F and CR are updated every generation with small probabilities (tau1, tau2) of random renewal.
# Exploration mechanisms: Mutation with random scaling factor (F) and crossover probability (CR) promotes diversity.
# Exploitation mechanisms: Selection pressure and crossover from successful individuals allow convergence.
# Boundary handling: Clamp to bounds after mutation/crossover.
# Budget strategy: Use full population initialization and then iterate generations until budget exhausted; handles small budgets by reducing population size.
# Closest known influences: jDE algorithm (Brest et al., 2006).
# Novelty or unusual aspects: None; straightforward implementation.
# Failure modes: May stagnate on highly multimodal or deceptive functions; population size may be too small for high‑dimensional problems.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

        # Determine population size, ensuring at least 4 for DE (if possible)
        # and not exceeding the budget.
        NP = max(4, int(10 * dim))
        NP = min(NP, budget)          # cannot exceed total evaluations
        NP = max(NP, 1)               # at least one individual
        self.NP = NP

        # Self‑adaptive parameter configuration
        self.tau1 = 0.1
        self.tau2 = 0.1

    def __call__(self, func):
        budget = self.budget
        NP = self.NP
        dim = self.dim

        # Obtain search bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            low = np.asarray(func.lower, dtype=float)
            high = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb, ub = func.bounds.lb, func.bounds.ub
            low = np.asarray(lb, dtype=float)
            high = np.asarray(ub, dtype=float)
        else:
            raise ValueError("Cannot determine bounds from func.")

        # Handle trivial case: only one evaluation available
        if budget == 1:
            x = low + (high - low) * np.random.rand(dim)
            y = func(x)
            return x, y

        # If NP is too small for standard DE mutation (needs at least 4),
        # fall back to pure random sampling
        if NP < 4:
            best_x = None
            best_y = np.inf
            for _ in range(budget):
                x = low + (high - low) * np.random.rand(dim)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x
            return best_x, best_y

        # ---- Initialise population ----
        pop = low + (high - low) * np.random.rand(NP, dim)
        fitness = np.full(NP, np.inf)

        # Evaluate initial population
        for i in range(NP):
            if budget <= 0:
                break
            fitness[i] = func(pop[i])
            budget -= 1

        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Initialise adaptive parameters
        # F in (0.1, 1.0], CR in [0,1]
        F = np.random.uniform(0.1, 1.0, NP)
        CR = np.random.uniform(0, 1, NP)

        # ---- Main DE loop (jDE self‑adaptation) ----
        while budget > 0:
            for i in range(NP):
                if budget == 0:
                    break

                # ---- Mutation (DE/rand/1) ----
                # Select three distinct individuals, none equal to i
                indices = np.random.permutation(NP)
                indices = indices[indices != i][:3]
                a, b, c = indices

                mutant = pop[a] + F[i] * (pop[b] - pop[c])

                # ---- Crossover (binomial) ----
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR[i]
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])

                # Clamp to bounds
                trial = np.clip(trial, low, high)

                # Evaluate trial
                f_trial = func(trial)
                budget -= 1

                # ---- Selection ----
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()

            else:
                # Full generation completed: update adaptive parameters
                # (only if budget still available, but we still update)
                rand_F = np.random.rand(NP) < self.tau1
                F[rand_F] = np.random.uniform(0.1, 1.0, np.sum(rand_F))

                rand_CR = np.random.rand(NP) < self.tau2
                CR[rand_CR] = np.random.uniform(0, 1, np.sum(rand_CR))

                # Continue to next generation if budget left
                continue
            break   # budget exhausted inside the inner loop

        return best_x, best_y
