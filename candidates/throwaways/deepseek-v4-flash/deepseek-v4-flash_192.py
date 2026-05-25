import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Self-adaptive Differential Evolution (jDE) for black-box minimization.
#          Maintains a population of candidate solutions, each with its own scaling factor F
#          and crossover probability CR. These control parameters are updated probabilistically
#          each generation, allowing the algorithm to adapt to the problem landscape.
# Search state: A population of size popSize, each individual is a dict containing:
#               'x' (dim-dimensional float array), 'F' (float in [0.1,1.0]), 'CR' (float in [0,1]).
# Candidate generation: For each individual i, a trial vector is created via DE/rand/1/bin:
#   - Choose three distinct random indices a,b,c != i.
#   - Mutation: v = x[a] + F_i * (x[b] - x[c])
#   - Crossover (binomial): u[j] = v[j] if rand_j < CR_i or j == j_rand else x[i][j]
#   - New F and CR are generated with probability tau1, tau2 from uniform distributions
#     and clipped to valid ranges; otherwise they remain unchanged.
# Selection and replacement: Greedy – if f(u) <= f(x[i]) then x[i] = u and the new F, CR are adopted;
#   otherwise x[i] and its parameters are kept.
# Adaptation: F and CR evolve per individual: with prob tau1 (~0.1) a new F ~ U(0.1,1.0);
#   with prob tau2 (~0.1) a new CR ~ U(0,1). Only successful updates pass the new parameters.
# Exploration mechanisms: DE mutation differences between random population members provide
#   global exploration. The probabilistic adaptation of F and CR can lead to occasional large
#   jumps (high F) or greedy behavior (high CR).
# Exploitation mechanisms: Crossover combines promising components; selection pressure from the
#   greedy replacement focuses on better individuals; low F/CR values lead to small local steps.
# Boundary handling: Any trial component outside [lower, upper] is clipped to the nearest bound.
# Budget strategy: The algorithm runs a single DE loop, evaluating at most budget candidates.
#   It initializes with one evaluation per individual, then continues generation after generation
#   until no more evaluations are available. The best-so-far solution is tracked.
# Closest known influences: Classic jDE (J. Brest et al., 2006) with uniform initialisation and
#   parameter ranges as in the original paper.
# Novelty or unusual aspects: None – a straightforward implementation of a well-known self-adaptive DE.
# Failure modes: May struggle on highly multimodal functions if population size is too small,
#   or on separable functions where component-wise adaptation is ineffective. No restart mechanism
#   is implemented, so premature convergence is possible on very low budgets.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

        # jDE parameters
        self.pop_size = max(20, min(100, 4 * dim))  # scale with dimension but cap
        self.tau1 = 0.1   # probability to update F
        self.tau2 = 0.1   # probability to update CR

    def __call__(self, func):
        # Read bounds from the provided function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.array(func.lower, dtype=float)
            upper = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lower = np.array(func.bounds.lb, dtype=float)
            upper = np.array(func.bounds.ub, dtype=float)
        else:
            # Fallback – unlikely to be needed
            raise AttributeError("Function must provide .lower/.upper or .bounds.lb/.bounds.ub")

        dim = len(lower)
        pop_size = self.pop_size
        budget = self.budget

        # Initialise population uniformly in the search space
        pop = np.zeros((pop_size, dim))
        F = np.full(pop_size, 0.5)       # default F
        CR = np.full(pop_size, 0.9)      # default CR
        for i in range(pop_size):
            pop[i] = lower + np.random.rand(dim) * (upper - lower)

        # Evaluate initial population
        fitness = np.array([func(pop[i]) for i in range(pop_size)])
        evals_used = pop_size
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        while evals_used < budget:
            # Estimate how many trial vectors we can generate this generation
            # (we will stop if evaluating the next trial would exceed budget)
            for i in range(pop_size):
                if evals_used >= budget:
                    break

                # Generate new F and CR for this individual with probabilities tau1, tau2
                new_F = F[i]
                new_CR = CR[i]
                if np.random.rand() < self.tau1:
                    new_F = np.random.uniform(0.1, 1.0)
                if np.random.rand() < self.tau2:
                    new_CR = np.random.uniform(0.0, 1.0)

                # Choose three distinct random indices a, b, c, all different from i
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, size=3, replace=False)

                # Mutation: DE/rand/1
                base = pop[a]
                diff = pop[b] - pop[c]
                trial = base + new_F * diff

                # Binomial crossover
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < new_CR
                mask[j_rand] = True
                trial = np.where(mask, trial, pop[i])

                # Boundary handling: clip to bounds
                trial = np.clip(trial, lower, upper)

                # Evaluate trial
                trial_fitness = func(trial)
                evals_used += 1

                # Selection
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    F[i] = new_F
                    CR[i] = new_CR
                    # Update global best
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()
                # else: keep parent (already unchanged)

        return best_x, best_y
