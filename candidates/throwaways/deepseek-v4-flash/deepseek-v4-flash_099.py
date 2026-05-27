# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact differential evolution (DE) minimizer using the rand/1/bin variant.
# Search state: A population of candidate solutions (vectors) stored as a 2D numpy array.
# Candidate generation: For each target vector, three distinct random population members are selected. The donor vector is computed as base + F * (diff1 - diff2). Binomial crossover combines donor and target with crossover rate CR to produce a trial vector.
# Selection and replacement: Greedy selection: the trial vector replaces the target if its objective value is lower (minimization). The best-so-far solution is tracked separately.
# Adaptation: Mutation factor F and crossover rate CR are fixed at 0.8 and 0.9 respectively. No online adaptation.
# Exploration mechanisms: The mutation differential vector provides random exploration scaled by F. The diversity of the population is maintained through random selection of mutation vectors.
# Exploitation mechanisms: The difference vector pushes solutions towards better regions indirectly as the population converges. The best solution is always preserved (elitism via tracking global best, not in population unless it survives).
# Boundary handling: Trial vectors are clipped component‑wise to the lower and upper bounds.
# Budget strategy: Evaluations are counted strictly. If the remaining budget is insufficient for a full generation, the loop terminates early; incomplete trials are skipped. A fallback random search is used if even one generation cannot be completed.
# Closest known influences: Standard differential evolution (Storn & Price, 1997) with rand/1/bin strategy and fixed parameters.
# Novelty or unusual aspects: None; straightforward implementation with boundary clipping.
# Failure modes: May converge prematurely for highly multimodal or ill‑conditioned functions. Fixed parameters may not suit all problems. Clipping can introduce bias near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Differential evolution minimizer for black‑box benchmarks."""
    
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # fixed parameters
        self.F = 0.8      # mutation factor
        self.CR = 0.9     # crossover probability
    
    def __call__(self, func):
        # parse bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # assume scipy-style bounds with .lb and .ub
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot determine bounds from func object")
        lb = lb.flatten() if lb.ndim > 1 else lb
        ub = ub.flatten() if ub.ndim > 1 else ub
        
        # population size: proportional to dimension, limited by budget
        pop_size = max(4, min(4 * self.dim, self.budget // 2))
        # ensure we can at least evaluate the initial population
        if self.budget < pop_size + 1:
            # fallback: pure random search
            best_y = np.inf
            best_x = None
            remaining = self.budget
            while remaining > 0:
                x = lb + (ub - lb) * np.random.random(self.dim)
                y = func(x)
                remaining -= 1
                if y < best_y:
                    best_y = y
                    best_x = x
            return best_x, best_y
        
        # initialize population uniformly in bounds
        pop = lb + (ub - lb) * np.random.random((pop_size, self.dim))
        fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            fitness[i] = func(pop[i])
        evals = pop_size
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]
        
        # main DE loop
        while evals < self.budget:
            # number of generations we can finish with current pop_size
            needed = pop_size  # one trial per target
            if evals + needed > self.budget:
                # not enough budget for a full generation -> do partial or stop
                break
            # one generation: for each target
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # select three distinct indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                np.random.shuffle(candidates)
                a, b, c = candidates[:3]  # a=base, b,c = diff
                donor = pop[a] + self.F * (pop[b] - pop[c])
                # binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.random() < self.CR or j == j_rand:
                        trial[j] = donor[j]
                # boundary clipping
                trial = np.clip(trial, lb, ub)
                # evaluate
                trial_y = func(trial)
                evals += 1
                # greedy selection
                if trial_y < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()
        return best_x, best_y
