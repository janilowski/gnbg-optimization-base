# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This implementation uses a Self-Adaptive Differential Evolution (jDE variant) algorithm. 
# It maintains a population of candidate solutions and evolves them through mutation, crossover, and selection.
# Search state: A population of vectors (individuals), their corresponding function values, and individual-specific control parameters (F and CR).
# Candidate generation: Mutation is performed using the 'DE/rand/1' strategy, and crossover is binomial.
# Selection and replacement: A simple one-to-one elitist selection is used; a child replaces its parent if it is better or equal in fitness.
# Adaptation: Control parameters F (mutation scale) and CR (crossover probability) are self-adapted for each individual. 
# Small probabilities trigger the re-randomization of these parameters, allowing the search to transition between exploration and exploitation.
# Exploration mechanisms: Random selection of mutation base vectors and self-adaptive parameters help maintain diversity.
# Exploitation mechanisms: The selection process keeps better solutions, and the population naturally contracts around minima.
# Boundary handling: Candidates that violate bounds are clipped to the search space [lb, ub].
# Budget strategy: The algorithm tracks evaluations and terminates immediately before exceeding the limit. 
# The population size is scaled based on dimension and budget to ensure sufficient iterations.
# Closest known influences: The jDE algorithm by Brest et al. (2006).
# Novelty or unusual aspects: A compact implementation tailored for variable budgets and dimensions, focusing on robustness without external dependencies.
# Failure modes: Very low budget relative to dimensionality may prevent the population from converging; extremely rugged or discontinuous landscapes might trap the population in local optima.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Self-Adaptive Differential Evolution (jDE) for black-box minimization.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Heuristic for population size: scale with dimension but cap to allow iterations
        self.pop_size = int(min(budget // 3, max(10, min(10 * dim, 50))))
        self.eval_count = 0

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and func.lower is not None:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback if no bounds are provided, though the spec suggests they exist
            lb = np.full(self.dim, -100.0)
            ub = np.full(self.dim, 100.0)

        # Initialize population
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        fitness = np.zeros(self.pop_size)
        
        # Individual control parameters: F (mutation factor) and CR (crossover rate)
        F = np.full(self.pop_size, 0.5)
        CR = np.full(self.pop_size, 0.9)

        best_x = None
        best_y = float('inf')

        # Initial evaluation
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            val = func(pop[i])
            fitness[i] = val
            self.eval_count += 1
            if val < best_y:
                best_y = val
                best_x = pop[i].copy()

        # Evolution loop
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Self-adaptation of parameters F and CR
                fi = F[i]
                cri = CR[i]
                if np.random.rand() < 0.1:
                    fi = 0.1 + np.random.rand() * 0.9
                if np.random.rand() < 0.1:
                    cri = np.random.rand()

                # Mutation: DE/rand/1
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + fi * (b - c)
                
                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) < cri
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary handling: Clipping
                trial = np.clip(trial, lb, ub)

                # Selection
                val = func(trial)
                self.eval_count += 1
                
                if val <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    F[i] = fi
                    CR[i] = cri
                    if val < best_y:
                        best_y = val
                        best_x = trial.copy()

        return best_x, best_y
