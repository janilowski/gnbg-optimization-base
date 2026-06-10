# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm is a compact implementation of Success-History Based Adaptive Differential Evolution (SHADE). It adapts mutation (F) and crossover (Cr) parameters based on a history of successful updates, allowing it to transition from exploration to exploitation as the search progresses.
# Search state: The state consists of a population of candidate solutions (vectors), their fitness values, and a fixed-size memory (history) of effective F and Cr parameters.
# Candidate generation: New candidates are generated using the 'current-to-pbest/1' mutation strategy: v = x + F * (x_pbest - x) + F * (x_r1 - x_r2), followed by binomial crossover.
# Selection and replacement: A greedy selection mechanism is used where a child replaces its parent in the population only if its fitness is better (or equal).
# Adaptation: The algorithm maintains a memory of size H. After each generation, successful F and Cr values are used to update the memory via a weighted Lehmer mean and arithmetic mean, respectively, prioritizing strategies that yielded larger fitness improvements.
# Exploration mechanisms: Exploration is driven by the differential mutation using randomly selected members of the population and a relatively high initial mutation factor.
# Exploitation mechanisms: Exploitation is facilitated by the 'pbest' component of the mutation, which directs individuals toward the top-performing members of the current population.
# Boundary handling: Candidates that fall outside the search space are clipped to the lower and upper bounds.
# Budget strategy: The algorithm tracks evaluations and terminates immediately once the budget is reached. The population size is scaled based on the dimensionality of the problem to ensure a reasonable number of generations.
# Closest known influences: SHADE (Tanabe and Fukunaga, 2013), JADE.
# Novelty or unusual aspects: Minimalist implementation of success-history adaptation tailored for a black-box setting with varying budgets.
# Failure modes: May converge prematurely on highly multi-modal landscapes if the population size is too small or if the p-best selection is too aggressive.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Population size: adaptive to dimension, but capped to ensure enough generations
        self.pop_size = int(max(10, min(dim * 10, 100)))
        # Memory size for SHADE adaptation
        self.memory_size = 6
        self.eval_count = 0

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = func.lower, func.upper
        elif hasattr(func, 'bounds'):
            lb, ub = func.bounds.lb, func.bounds.ub
        else:
            # Fallback if no bounds are detected
            lb, ub = np.full(self.dim, -5.0), np.full(self.dim, 5.0)
            
        lb = np.asarray(lb)
        ub = np.asarray(ub)

        # Initialize population
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        fitness = np.zeros(self.pop_size)
        
        best_x = None
        best_y = float('inf')

        # Initial evaluation
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            fitness[i] = func(pop[i])
            self.eval_count += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()

        # SHADE adaptation memory
        mem_f = np.full(self.memory_size, 0.5)
        mem_cr = np.full(self.memory_size, 0.5)
        k_idx = 0
        
        # p-best parameter (top p% individuals)
        p_min = 2 / self.pop_size
        p_max = 0.2
        
        while self.eval_count < self.budget:
            success_f = []
            success_cr = []
            diff_fitness = []
            
            new_pop = np.copy(pop)
            new_fitness = np.copy(fitness)
            
            # Generate new candidates
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Pick parameters from memory
                idx = np.random.randint(0, self.memory_size)
                cr = np.clip(np.random.normal(mem_cr[idx], 0.1), 0, 1)
                
                # Cauchy distribution for F
                f = mem_f[idx] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                while f <= 0:
                    f = mem_f[idx] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                f = min(f, 1.0)
                
                # Current-to-pbest mutation
                p = np.random.uniform(p_min, p_max)
                num_pbest = max(1, int(p * self.pop_size))
                pbest_indices = np.argsort(fitness)[:num_pbest]
                x_pbest = pop[np.random.choice(pbest_indices)]
                
                # Select two random distinct indices r1, r2
                r1 = np.random.randint(0, self.pop_size)
                while r1 == i:
                    r1 = np.random.randint(0, self.pop_size)
                r2 = np.random.randint(0, self.pop_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, self.pop_size)
                
                # Mutate
                mutant = pop[i] + f * (x_pbest - pop[i]) + f * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                
                # Crossover
                j_rand = np.random.randint(0, self.dim)
                trial = np.where(np.random.rand(self.dim) < cr, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                
                # Evaluate
                score = func(trial)
                self.eval_count += 1
                
                if score <= fitness[i]:
                    if score < fitness[i]:
                        success_f.append(f)
                        success_cr.append(cr)
                        diff_fitness.append(abs(fitness[i] - score))
                    
                    new_pop[i] = trial
                    new_fitness[i] = score
                    
                    if score < best_y:
                        best_y = score
                        best_x = trial.copy()

            # Update population
            pop = new_pop
            fitness = new_fitness
            
            # Update SHADE memory
            if success_f:
                weights = np.array(diff_fitness) / np.sum(diff_fitness)
                
                # Weighted arithmetic mean for Cr
                mem_cr[k_idx] = np.sum(weights * np.array(success_cr))
                
                # Weighted Lehmer mean for F
                num = np.sum(weights * (np.array(success_f)**2))
                den = np.sum(weights * np.array(success_f))
                mem_f[k_idx] = num / den if den != 0 else mem_f[k_idx]
                
                k_idx = (k_idx + 1) % self.memory_size

        return best_x, best_y
