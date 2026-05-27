import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adaptive differential evolution (jDE) for black-box minimization. Maintains a population where each individual has its own crossover rate (CR) and scaling factor (F) that evolve over time.
# Search state: A population of size popsize (max(10, 2*dim)). Each individual is a tuple (position vector, F, CR). Best-so-far solution (best_x, best_y) is tracked.
# Candidate generation: For each parent i, a trial vector is created using DE/rand/1/bin: pick three distinct random parents (r0, r1, r2) and a random dimension jrand. For each dimension j, if rand < CR_i or j == jrand, trial[j] = x_r0[j] + F_i * (x_r1[j] - x_r2[j]); else trial[j] = x_i[j].
# Selection and replacement: Greedy selection: the trial vector replaces the parent if its function value is lower (minimization). The trial vector carries its own F and CR (mutated before generation). If the trial is not selected, the parent retains its own F and CR.
# Adaptation: At each generation, before generating the trial, each individual's F and CR are mutated with probability 0.1: F = uniform(0.1,1) if rand < 0.1, else keep; CR = uniform(0,1) if rand < 0.1, else keep. This enables adaptation of control parameters based on success.
# Exploration mechanisms: DE's differential mutation and crossover with adaptive parameters. The low probability of re-initializing F and CR allows exploration of parameter space. Random selection of base vector and difference vectors introduces diversity.
# Exploitation mechanisms: Greedy replacement maintains pressure towards better solutions. As population converges, difference vectors become small, reducing step size.
# Boundary handling: Coordinates are clipped to the bounds after mutation. This ensures all points are feasible.
# Budget strategy: The algorithm iterates generation by generation, evaluating one trial per individual. If the remaining budget is insufficient for a full generation, it stops after using remaining evaluations on a randomly permuted subset. This guarantees budget is never exceeded.
# Closest known influences: jDE (Brest et al., 2006) - self-adaptive differential evolution.
# Novelty or unusual aspects: None – standard jDE implementation with small adaptations in population sizing and budget handling.
# Failure modes: May struggle on highly multimodal or ill-conditioned problems where population diversity is lost prematurely. The fixed population size may be suboptimal for very high dimensions or extremely small budgets.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Population size: at least 10, at most 2*dim (but capped for very large dim)
        self.popsize = max(10, min(2 * dim, 200))  # reasonable upper bound

    def __call__(self, func):
        # Read bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            bounds = func.bounds
            lb = np.array(bounds.lb, dtype=float)
            ub = np.array(bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot determine bounds from function object")

        dim = self.dim
        npop = self.popsize
        budget = self.budget

        # Initialize population uniformly within bounds
        pop = np.random.uniform(lb, ub, size=(npop, dim))
        # Control parameters per individual (F in [0.1,1], CR in [0,1])
        F = np.full(npop, 0.5)
        CR = np.full(npop, 0.5)

        # Evaluate initial population, stop early if budget exhausted
        pop_fit = np.full(npop, np.inf)
        best_x = None
        best_y = np.inf
        evals = 0
        for i in range(npop):
            if evals >= budget:
                break
            y = func(pop[i])
            evals += 1
            pop_fit[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Main DE loop
        while evals < budget:
            # Randomize order to avoid bias
            order = np.random.permutation(npop)
            for idx in order:
                if evals >= budget:
                    break

                # Generate mutated control parameters for this individual
                new_F = F[idx]
                new_CR = CR[idx]
                if np.random.rand() < 0.1:
                    new_F = np.random.uniform(0.1, 1.0)
                if np.random.rand() < 0.1:
                    new_CR = np.random.uniform(0.0, 1.0)

                # Select three distinct random indices different from idx
                candidates = [i for i in range(npop) if i != idx]
                r0, r1, r2 = np.random.choice(candidates, size=3, replace=False)

                # Create trial vector via DE/rand/1/bin
                jrand = np.random.randint(dim)
                trial = pop[idx].copy()
                for j in range(dim):
                    if np.random.rand() < new_CR or j == jrand:
                        trial[j] = pop[r0][j] + new_F * (pop[r1][j] - pop[r2][j])
                # Clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                y_trial = func(trial)
                evals += 1

                # Update global best
                if y_trial < best_y:
                    best_y = y_trial
                    best_x = trial.copy()

                # Selection: replace parent if trial is better
                if y_trial < pop_fit[idx]:
                    pop[idx] = trial
                    pop_fit[idx] = y_trial
                    F[idx] = new_F
                    CR[idx] = new_CR

        return best_x, best_y
