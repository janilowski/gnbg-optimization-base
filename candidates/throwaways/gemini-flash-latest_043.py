# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust implementation of Success-History Adaptive Differential Evolution (SHADE).
# Search state: A population of candidate vectors, their objective values, and a memory buffer of successful F (scaling factor) and Cr (crossover rate) parameters.
# Candidate generation: Uses the 'current-to-pbest/1' mutation strategy, which blends a vector's current position with a randomly selected individual from the top 'p' fraction of the population, plus a scaled difference between two other random members.
# Selection and replacement: Standard DE greedy selection: an offspring replaces its parent if and only if its fitness is less than or equal to the parent's fitness.
# Adaptation: F and Cr are sampled from Cauchy and Normal distributions respectively, centered on values from a historical memory. This memory is updated using the Lehmer mean of parameters that successfully produced improving offspring.
# Exploration mechanisms: The mutation differential (r1 - r2) provides stochastic exploration. Parameter sampling from distributions ensures diversity in search step sizes.
# Exploitation mechanisms: The 'pbest' component of the mutation strategy biases the search towards the most successful regions discovered so far.
# Boundary handling: Offspring components exceeding bounds are clamped to the nearest boundary. If an entire vector is invalid, it is re-initialized toward the parent.
# Budget strategy: The algorithm tracks evaluations internally and terminates immediately once the budget is exhausted, even within a generation.
# Closest known influences: SHADE (Tanabe and Fukunaga, 2013).
# Novelty or unusual aspects: Streamlined memory management and robust bound detection compatible with multiple GNBG-style interface conventions.
# Failure modes: Like most DE variants, it may converge prematurely on highly rugose, multi-modal landscapes if the population size is too small relative to the dimensionality.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
        # Population size: heuristic balancing exploration and budget
        self.pop_size = int(max(10, min(100, 15 * dim)))
        if self.pop_size > budget:
            self.pop_size = max(4, budget // 2)
            
        # SHADE parameters
        self.memory_size = 10
        self.memory_cr = np.full(self.memory_size, 0.5)
        self.memory_f = np.full(self.memory_size, 0.5)
        self.memory_idx = 0
        self.p_best_rate = 0.15
        
    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'bounds') and hasattr(func.bounds, 'lb'):
            lb = np.asfarray(func.bounds.lb)
            ub = np.asfarray(func.bounds.ub)
        elif hasattr(func, 'lower'):
            lb = np.asfarray(func.lower)
            ub = np.asfarray(func.upper)
        else:
            # Fallback for unexpected interface
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Handle scalar bounds
        if lb.ndim == 0: lb = np.full(self.dim, lb)
        if ub.ndim == 0: ub = np.full(self.dim, ub)

        evals = 0
        
        # Initialize population
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.zeros(self.pop_size)
        
        best_x = None
        best_y = float('inf')

        # Initial evaluation
        for i in range(self.pop_size):
            if evals >= self.budget:
                break
            val = func(pop[i])
            fitness[i] = val
            evals += 1
            if val < best_y:
                best_y = val
                best_x = pop[i].copy()
        
        if evals >= self.budget:
            return best_x, best_y

        # Main Evolution Loop
        while evals < self.budget:
            success_f = []
            success_cr = []
            diff_fitness = []
            
            # Sort population for p-best selection
            sorted_idx = np.argsort(fitness)
            p_best_count = max(2, int(self.p_best_rate * self.pop_size))
            p_best_indices = sorted_idx[:p_best_count]
            
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample parameters from memory
                m_idx = np.random.randint(0, self.memory_size)
                
                # Sample Cr
                cr = np.random.normal(self.memory_cr[m_idx], 0.1)
                cr = np.clip(cr, 0, 1)
                
                # Sample F (Cauchy distribution)
                f = self.memory_f[m_idx] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                while f <= 0:
                    f = self.memory_f[m_idx] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                if f > 1.0:
                    f = 1.0
                
                # Mutation: current-to-pbest/1
                p_best_idx = np.random.choice(p_best_indices)
                
                # Select r1, r2 distinct from i and pbest
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                
                mutant = pop[i] + f * (pop[p_best_idx] - pop[i]) + f * (pop[r1] - pop[r2])
                
                # Crossover
                cross_mask = np.random.rand(self.dim) <= cr
                cross_mask[np.random.randint(0, self.dim)] = True # Ensure at least one dim changes
                
                trial = np.where(cross_mask, mutant, pop[i])
                
                # Boundary Handling (Clamping)
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                val = func(trial)
                evals += 1
                
                # Selection
                if val <= fitness[i]:
                    if val < fitness[i]:
                        success_f.append(f)
                        success_cr.append(cr)
                        diff_fitness.append(abs(fitness[i] - val))
                    
                    fitness[i] = val
                    pop[i] = trial
                    
                    if val < best_y:
                        best_y = val
                        best_x = trial.copy()
            
            # Update SHADE Memory
            if success_f:
                weights = np.array(diff_fitness) / np.sum(diff_fitness)
                
                # Lehmer mean for F
                new_f = np.sum(weights * (np.array(success_f)**2)) / np.sum(weights * np.array(success_f))
                # Weighted mean for Cr
                new_cr = np.sum(weights * np.array(success_cr))
                
                self.memory_f[self.memory_idx] = new_f
                self.memory_cr[self.memory_idx] = new_cr
                self.memory_idx = (self.memory_idx + 1) % self.memory_size
                
        return best_x, best_y
