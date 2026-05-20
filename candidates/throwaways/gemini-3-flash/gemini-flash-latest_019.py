# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simplified Success-History Based Adaptive Differential Evolution (SHADE-lite).
# Search state: A population of candidate solutions, associated fitness values, and a memory of successful search parameters (F and CR).
# Candidate generation: Uses the DE/rand/1 mutation strategy where a mutant vector is created by adding a scaled difference between two random members to a third. Binomial crossover then mixes the parent and mutant.
# Selection and replacement: Greedy selection; a child replaces its parent if its objective value is less than or equal to the parent's.
# Adaptation: The scaling factor (F) and crossover rate (CR) are sampled from a historical memory of successful parameters, which is updated when a child outperforms its parent.
# Exploration mechanisms: Random vector selection in the mutation step and the diversity of the initial population.
# Exploitation mechanisms: Greedy selection and the adaptation of parameters toward values that have previously yielded improvements.
# Boundary handling: Candidates are clipped to the box constraints defined by the function attributes.
# Budget strategy: A strict counter is maintained. The loop terminates immediately when the evaluation limit is reached.
# Closest known influences: SHADE (Tanabe and Fukunaga, 2013) and JADE.
# Novelty or unusual aspects: Minimalist parameter memory implementation designed for robustness across varying dimensions and budget sizes.
# Failure modes: May converge prematurely on highly multi-modal landscapes if the population size is too small for the dimensionality, or struggle with extremely narrow ridges.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the Differential Evolution algorithm.
        
        :param budget: Maximum number of function evaluations.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        
        # Population size heuristics
        self.pop_size = max(10, min(100, 10 * dim))
        
        # Memory for parameter adaptation (SHADE-lite)
        self.memory_size = 10
        self.memory_f = np.full(self.memory_size, 0.5)
        self.memory_cr = np.full(self.memory_size, 0.5)
        self.memory_idx = 0

    def __call__(self, func):
        """
        Executes the optimization process.
        
        :param func: The objective function to minimize.
        :return: A tuple (best_x, best_y).
        """
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        else:
            # Fallback to defaults if bounds are missing
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Initialization
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.zeros(self.pop_size)
        eval_count = 0

        # Initial evaluation
        for i in range(self.pop_size):
            if eval_count >= self.budget:
                # Return best found so far if budget exhausted during init
                best_idx = np.argmin(fitness[:i]) if i > 0 else 0
                return pop[best_idx], fitness[best_idx]
            fitness[i] = func(pop[i])
            eval_count += 1

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Optimization loop
        while eval_count < self.budget:
            success_f = []
            success_cr = []
            
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break

                # Pick parameters from memory
                m_idx = np.random.randint(0, self.memory_size)
                # Cauchy distribution for F, Normal for CR
                f = np.random.standard_cauchy() * 0.1 + self.memory_f[m_idx]
                while f <= 0:
                    f = np.random.standard_cauchy() * 0.1 + self.memory_f[m_idx]
                f = min(f, 1.0)
                
                cr = np.random.normal(self.memory_cr[m_idx], 0.1)
                cr = np.clip(cr, 0.0, 1.0)

                # Mutation: DE/rand/1
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                mutant = pop[r1] + f * (pop[r2] - pop[r3])
                
                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) < cr
                # Ensure at least one dimension is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary handling
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                trial_f = func(trial)
                eval_count += 1
                
                # Selection
                if trial_f <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    success_f.append(f)
                    success_cr.append(cr)
                    if trial_f < best_y:
                        best_y = trial_f
                        best_x = trial.copy()

            # Update parameter memory if successes occurred
            if success_f:
                # Lehmer mean for F
                self.memory_f[self.memory_idx] = sum(s**2 for s in success_f) / sum(success_f)
                # Arithmetic mean for CR
                self.memory_cr[self.memory_idx] = sum(success_cr) / len(success_cr)
                self.memory_idx = (self.memory_idx + 1) % self.memory_size

        return best_x, best_y

# No top-level execution code as per requirements.
