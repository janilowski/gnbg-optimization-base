import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a classic Differential Evolution (DE/rand/1/bin) optimizer for black-box minimization. It uses a fixed population size, binomial crossover, and boundary clipping. It restarts the population when stagnation is detected.
# Search state: Population of candidate solutions and their fitness values; best solution and fitness.
# Candidate generation: For each target vector, three distinct population members are randomly selected. A mutant vector is generated as base + F * (diff1 - diff2). Then a trial vector is created via binomial crossover with the target, using crossover rate CR.
# Selection and replacement: The trial vector replaces the target if its fitness is better (lower). Greedy selection.
# Adaptation: The mutation factor F and crossover rate CR are kept constant. A stagnation counter triggers a restart of the population around the current best when no improvement is seen for several generations.
# Exploration mechanisms: The rand/1 mutation and random selection of base and differences encourage exploration; the population is reinitialized with diversity on stagnation.
# Exploitation mechanisms: The binomial crossover preserves parts of the target; the best solution is kept separate; after restart, population is sampled around the best to focus search.
# Boundary handling: All candidate solutions are clipped to the lower and upper bounds after mutation and crossover.
# Budget strategy: The algorithm stops when the number of function evaluations exceeds the budget. Population size is chosen adaptively.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997) with periodic restart.
# Novelty or unusual aspects: Uses a simple stagnation-based restart without any sophisticated adaptation of parameters.
# Failure modes: May converge prematurely to local optima in highly multimodal landscapes; performance depends on suitable F and CR values; no rotation invariance.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Set population size: large enough for DE, but not too large relative to budget
        if budget >= 20:
            self.pop_size = max(4, min(10 * dim, budget // 2))
        else:
            self.pop_size = max(1, budget)
        self.F = 0.8          # mutation factor
        self.CR = 0.9        # crossover rate
        self.stagnation_limit = max(5, 3 * dim)   # generations without improvement before restart

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lower = np.asarray(b.lb, dtype=float)
            upper = np.asarray(b.ub, dtype=float)
        else:
            raise ValueError("Function must provide lower/upper or bounds.lb/ub")
        dim = self.dim
        lower = lower[:dim]
        upper = upper[:dim]
        diff = upper - lower

        # Very small budget: pure random search
        if self.budget < 6:
            best_x = None
            best_y = np.inf
            evals = 0
            while evals < self.budget:
                x = lower + np.random.random(dim) * diff
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x
            return best_x, best_y

        pop_size = self.pop_size
        # Initialise population uniformly
        pop = lower + np.random.rand(pop_size, dim) * diff
        fitness = np.array([func(pop[i]) for i in range(pop_size)])
        evals = pop_size
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        stagnation = 0
        # Main loop
        while evals < self.budget:
            improved = False
            # For each target vector
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # Choose three distinct indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                r0, r1, r2 = np.random.choice(candidates, 3, replace=False)
                # Mutation
                mutant = pop[r0] + self.F * (pop[r1] - pop[r2])
                # Binomial crossover
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < self.CR
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                # Clip to bounds
                trial = np.clip(trial, lower, upper)
                # Evaluate
                y_trial = func(trial)
                evals += 1
                # Selection
                if y_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = y_trial
                    improved = True
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()
            if not improved:
                stagnation += 1
            else:
                stagnation = 0
            # Restart if stagnated and enough budget remains
            if stagnation >= self.stagnation_limit and evals < self.budget:
                # Keep best, re-initialize rest around it
                # Use 20% of range as standard deviation
                std = diff * 0.2
                for i in range(pop_size):
                    if evals >= self.budget:
                        break
                    # Random sample around best
                    x = best_x + np.random.randn(dim) * std
                    x = np.clip(x, lower, upper)
                    y = func(x)
                    evals += 1
                    pop[i] = x
                    fitness[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = x.copy()
                stagnation = 0
        return best_x, best_y
