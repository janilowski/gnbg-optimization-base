import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a Differential Evolution (DE) optimizer for black-box minimization.
#          It uses a classic DE/rand/1/bin scheme with self-adaptive control parameters (jDE variant)
#          to maintain robustness across a wide range of problem dimensions and landscapes.
# Search state: Population of NP candidate solutions (each a vector of length dim).
#               Each individual also carries its own scaling factor F and crossover rate CR,
#               which evolve over time.
# Candidate generation: For each target individual, a mutant vector is created by adding
#                       the scaled difference of two randomly chosen distinct population members
#                       to a third random member (DE/rand/1). A trial vector is formed by
#                       binomial crossover between the target and the mutant with probability CR.
# Selection and replacement: Greedy selection: if the trial vector yields a lower objective
#                            value than the target, it replaces the target in the population.
# Adaptation: Each individual’s F and CR are randomly re‑initialized with small probabilities
#             (τ₁=0.1, τ₂=0.1) at the start of each generation. New values are drawn uniformly
#             from [0.1, 1.0] for F and [0.0, 1.0] for CR. This mechanism promotes diversity
#             and avoids hand‑tuning.
# Exploration mechanisms: The differential mutation step (rand/1) introduces large, random
#                         perturbations, especially when the population is diverse. The adaptive
#                         F and CR modulate step size and mixing, allowing the algorithm to
#                         escape local optima when needed.
# Exploitation mechanisms: Greedy selection coupled with small F (when adapted) and high CR
#                          (when adapted) encourages fine‑tuning around promising solutions.
#                          The best solution is tracked throughout the run.
# Boundary handling: Trial vectors that violate bounds are repaired by simple clipping to
#                    the nearest bound value.
# Budget strategy: The algorithm evaluates exactly one candidate per generation target index.
#                  The total number of evaluations is checked before each evaluation and the
#                  process stops immediately when the budget is exhausted. The first NP evaluations
#                  are used for the initial population.
# Closest known influences: The jDE (self‑adaptive differential evolution) algorithm by
#                           Brest et al. (2006) and the classic DE/rand/1/bin variant.
# Novelty or unusual aspects: None – this is a straightforward implementation of jDE
#                             adapted for a given budget and bound‑constrained minimization.
# Failure modes: May converge prematurely on highly multimodal or deceptive landscapes if
#                 population diversity is lost. Very high dimensions may require a larger
#                 population (and hence a larger budget) to maintain effective exploration.
#                 The fixed population size heuristic may be suboptimal for extreme dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Differential Evolution (jDE) minimizer for black‑box functions."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # --- Read bounds ---------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot read bounds from func. "
                                 "Expected func.lower/upper or func.bounds.lb/ub.")
        # Ensure bounds are 1‑D arrays
        lower = lower.flatten()
        upper = upper.flatten()
        dim = len(lower)

        # --- Population size heuristic -------------------------------------
        NP = max(10, min(50, 2 * dim))          # moderate population size
        if NP * 2 > self.budget:                # cannot afford initialisation + one generation
            NP = max(2, self.budget // 2)       # at least 2 individuals
        NP = int(NP)

        # --- Initialise population and control parameters ------------------
        pop = np.random.uniform(lower, upper, size=(NP, dim))
        # F and CR per individual (will be adapted later)
        F  = np.full(NP, 0.5)
        CR = np.full(NP, 0.9)
        # Evaluate initial population
        fit = np.array([func(x) for x in pop])   # consumes NP evaluations
        evals = NP

        # Track best
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # --- Evolution loop ------------------------------------------------
        # jDE adaptation probabilities
        tau1, tau2 = 0.1, 0.1

        while evals < self.budget:
            for i in range(NP):
                if evals >= self.budget:
                    break

                # --- Adaptive control (jDE) ---
                # Re‑sample F and CR with small probability
                if np.random.rand() < tau1:
                    F[i] = 0.1 + np.random.rand() * 0.9   # [0.1, 1.0]
                if np.random.rand() < tau2:
                    CR[i] = np.random.rand()               # [0.0, 1.0]

                # --- Mutation (DE/rand/1) ---
                # pick three distinct indices different from i
                idxs = [j for j in range(NP) if j != i]
                a, b, c = np.random.choice(idxs, size=3, replace=False)
                mutant = pop[a] + F[i] * (pop[b] - pop[c])

                # --- Crossover (binomial) ---
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]

                # --- Boundary handling (clip) ---
                trial = np.clip(trial, lower, upper)

                # --- Evaluation ---
                trial_fit = func(trial)
                evals += 1

                # --- Selection ---
                if trial_fit < fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fit
                    # update global best
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

            # End of generation (inner loop may also stop due to budget)

        return best_x, best_y
