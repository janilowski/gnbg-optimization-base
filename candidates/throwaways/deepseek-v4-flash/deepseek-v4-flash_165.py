import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Self-adaptive Differential Evolution (jDE variant) for black-box minimization.
# Search state: Population of NP candidate vectors each with its own scaling factor F and crossover rate CR.
# Candidate generation: For each target individual, a mutant is created using DE/rand/1 with
#   its own F and CR. The trial vector combines mutant and target via binomial crossover.
# Selection and replacement: Greedy selection – trial replaces target if it yields lower function value.
# Adaptation: F and CR are updated before each mutation using log-normal rules (jDE adaptation).
# Exploration mechanisms: Large F values (~0.9) encourage exploration; small CR values (~0.1) keep many target components.
# Exploitation mechanisms: The population contracts as successful (F,CR) values are retained and small mutations refine solutions.
# Boundary handling: Components outside bounds are reflected back into the feasible domain.
# Budget strategy: Evaluations are consumed exactly as the function is called; the algorithm stops when the budget is exhausted.
#   One final generation may be incomplete; any remaining budget is used to evaluate some leftover trial vectors.
# Closest known influences: jDE (Brest et al., 2006) – a classic self-adaptive DE.
# Novelty or unusual aspects: None – this is a straightforward implementation of jDE with bounds reflection,
#   tuned for robustness across dimensions by using population size = 4 + floor(3 * ln(dim)).
# Failure modes: May converge prematurely on highly multimodal landscapes or when the budget is too small.
#   Reflection may cause clustering near bounds. No restart mechanism is implemented.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # Population size rule: 4 + floor(3 * log(dim)) , min 10 for small dim.
        self.NP = max(10, 4 + int(3 * np.log(self.dim)))

        # Ensure we don't exceed budget unreasonably: at least 1 generation possible
        # But we will stop inside the loop when budget is exhausted.

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Function must provide 'lower'/'upper' or 'bounds.lb'/'bounds.ub'.")

        # Initialization
        NP = self.NP
        dim = self.dim
        budget = self.budget

        # Population vectors
        pop = lb + (ub - lb) * np.random.rand(NP, dim)
        # Fitness values
        fitness = np.full(NP, np.inf)
        for i in range(NP):
            fitness[i] = func(pop[i])
            budget -= 1
            if budget <= 0:
                # Ran out of budget immediately? return best so far.
                best_idx = np.argmin(fitness[:i+1])
                return pop[best_idx].copy(), fitness[best_idx]

        # Self-adaptive parameters: F in [0.1, 0.9], CR in [0, 0.9]
        F = 0.5 + 0.4 * np.random.rand(NP)          # initial F in [0.5, 0.9]
        CR = 0.5 + 0.4 * np.random.rand(NP)         # initial CR in [0.5, 0.9]

        # Best so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main loop
        while budget > 0:
            # For each target individual
            for i in range(NP):
                if budget <= 0:
                    break

                # --- Adaptation of F and CR (jDE style) ---
                # F: with prob tauF, generate new value between 0.1 and 1.0? jDE uses uniform [0.1, 1.0] with tauF=0.1.
                # CR: with prob tauCR, generate new value between 0 and 1.
                # We follow classic jDE: tauF = 0.1, tauCR = 0.1.
                r1, r2 = np.random.rand(2)
                if r1 < 0.1:
                    Fi = 0.1 + 0.9 * np.random.rand()
                else:
                    Fi = F[i]
                if r2 < 0.1:
                    CRi = np.random.rand()
                else:
                    CRi = CR[i]

                # --- Mutation: DE/rand/1 ---
                # Choose three distinct indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                mutant = pop[a] + Fi * (pop[b] - pop[c])

                # --- Boundary reflection ---
                # Reflect components outside [lb, ub] back inside.
                # First treat dimension-wise.
                mutant = np.where(mutant < lb, 2*lb - mutant, mutant)
                mutant = np.where(mutant > ub, 2*ub - mutant, mutant)
                # If still outside after reflection (edge case), clamp.
                mutant = np.clip(mutant, lb, ub)

                # --- Crossover: binomial ---
                # Always cross with at least one dimension from mutant.
                jrand = np.random.randint(dim)
                trial = np.where(
                    (np.random.rand(dim) < CRi) | (np.arange(dim) == jrand),
                    mutant,
                    pop[i]
                )

                # Evaluate trial
                trial_fitness = func(trial)
                budget -= 1

                # --- Selection ---
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    F[i] = Fi
                    CR[i] = CRi
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()
                # else: keep old (F, CR unchanged)

            # Ensure we break out if budget ran out in the inner loop
            if budget <= 0:
                break

        return best_x, best_y
