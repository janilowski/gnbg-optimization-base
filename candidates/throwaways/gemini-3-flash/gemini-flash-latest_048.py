# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Success-History Based Adaptive Differential Evolution (SHADE) variant.
# Search state: A population of candidate solutions, their fitness values, and a historical memory of successful control parameters (F and Cr).
# Candidate generation: Uses the 'current-to-pbest/1' mutation strategy, which balances local exploitation near the best solutions with global exploration, followed by binomial crossover.
# Selection and replacement: Standard DE greedy selection where a trial vector replaces its parent only if it achieves a lower or equal objective value.
# Adaptation: Control parameters F (scaling factor) and Cr (crossover probability) are sampled from distributions centered on values stored in a success-history memory. This memory is updated using a weighted Lehmer mean of parameters that produced successful improvements.
# Exploration mechanisms: Maintained through a population-based approach and the random differential terms in the mutation strategy.
# Exploitation mechanisms: Enhanced by the 'current-to-pbest' mutation, which directs individuals toward one of the top-performing members of the current population.
# Boundary handling: Trial vectors are clipped to the specified lower and upper bounds.
# Budget strategy: Evaluations are counted strictly; the population size is scaled based on dimensionality and budget, and the search terminates immediately upon budget exhaustion.
# Closest known influences: The SHADE algorithm (Tanabe & Fukunaga, 2013) and JADE (Zhang & Sanderson, 2009).
# Novelty or unusual aspects: A compact, single-class implementation of success-history adaptation tailored for black-box benchmark constraints.
# Failure modes: Like most DE variants, it may struggle with extremely high-dimensional landscapes where the budget is very low relative to the dimension, or landscapes with highly deceptive global optima.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the SHADE-inspired Differential Evolution algorithm.
        
        Args:
            budget (int): Total number of function evaluations allowed.
            dim (int): Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        
        # Adaptive population size heuristic
        # Ensure at least 4 individuals for DE mutation logic
        self.pop_size = max(4, min(10 * dim, 100))
        if self.pop_size > budget:
            self.pop_size = budget

    def __call__(self, func):
        """
        Executes the optimization process.
        
        Args:
            func: The objective function to minimize.
        
        Returns:
            tuple: (best_x, best_y) found during the search.
        """
        # 1. Bounds extraction
        if hasattr(func, 'bounds') and hasattr(func.bounds, 'lb'):
            lb, ub = np.asarray(func.bounds.lb), np.asarray(func.bounds.ub)
        elif hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.asarray(func.lower), np.asarray(func.upper)
        else:
            # Default bounds if none provided (rare in benchmarks)
            lb, ub = np.full(self.dim, -5.0), np.full(self.dim, 5.0)

        best_x = None
        best_y = float('inf')

        def evaluate(x):
            nonlocal best_x, best_y
            if self.eval_count >= self.budget:
                return None
            y = func(x)
            self.eval_count += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # 2. Initialization
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        fitness = np.zeros(self.pop_size)
        
        for i in range(self.pop_size):
            y = evaluate(pop[i])
            if y is None: break
            fitness[i] = y

        # Success-History Memory initialization
        H = 6
        memory_f = np.full(H, 0.5)
        memory_cr = np.full(H, 0.5)
        memory_idx = 0

        # Optimization Loop
        while self.eval_count < self.budget:
            success_f = []
            success_cr = []
            diff_fitness = []

            # Sort population to identify p-best
            indices = np.argsort(fitness)
            p_best_count = max(1, int(0.15 * self.pop_size))
            p_best_indices = indices[:p_best_count]

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Sample parameters from history
                r = np.random.randint(0, H)
                cr = np.clip(np.random.normal(memory_cr[r], 0.1), 0, 1)
                
                # Sample F using Cauchy distribution
                while True:
                    f = memory_f[r] + 0.1 * np.random.standard_cauchy()
                    if f > 0:
                        f = min(f, 1.0)
                        break
                
                # current-to-pbest/1 mutation
                x_i = pop[i]
                x_pbest = pop[np.random.choice(p_best_indices)]
                
                r1 = np.random.randint(0, self.pop_size)
                while r1 == i: r1 = np.random.randint(0, self.pop_size)
                r2 = np.random.randint(0, self.pop_size)
                while r2 == i or r2 == r1: r2 = np.random.randint(0, self.pop_size)
                
                v = x_i + f * (x_pbest - x_i) + f * (pop[r1] - pop[r2])
                
                # Boundary handling: Clamping
                v = np.clip(v, lb, ub)
                
                # Binomial crossover
                j_rand = np.random.randint(0, self.dim)
                mask = np.random.rand(self.dim) < cr
                mask[j_rand] = True
                trial = np.where(mask, v, x_i)
                
                # Evaluation and Selection
                trial_y = evaluate(trial)
                if trial_y is not None:
                    if trial_y <= fitness[i]:
                        if trial_y < fitness[i]:
                            success_f.append(f)
                            success_cr.append(cr)
                            diff_fitness.append(fitness[i] - trial_y)
                        fitness[i] = trial_y
                        pop[i] = trial

            # Update history memory
            if success_f:
                weights = np.array(diff_fitness) / np.sum(diff_fitness)
                # Weighted Lehmer mean for F
                memory_f[memory_idx] = np.sum(weights * (np.array(success_f)**2)) / np.sum(weights * np.array(success_f))
                # Weighted Arithmetic mean for Cr
                memory_cr[memory_idx] = np.sum(weights * np.array(success_cr))
                
                memory_idx = (memory_idx + 1) % H

        return best_x, best_y
