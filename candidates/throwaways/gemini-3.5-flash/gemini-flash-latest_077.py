# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A hybrid optimization algorithm combining Success-History Adaptive Differential Evolution (SHADE) with Linear Population Size Reduction (L-PSR) and a final fine-grained Adaptive Coordinate Descent Local Search (CD-LS).
# Search state: Population of candidates, objective fitness values, historical memory of mutation scale (F) and crossover rate (Cr), best discovered solution, and coordinate step-sizes for local search.
# Candidate generation: Generates trial vectors via SHADE's current-to-pbest/1 mutation strategy combined with binomial crossover. Parameter values (F and Cr) are sampled dynamically for each individual from historical success memories.
# Selection and replacement: Standard greedy selection compares trial vectors against parent vectors. If a trial vector is better or equal, it replaces the parent. Successful parameters are stored for memory updates.
# Adaptation: Employs L-PSR to linearly reduce population size over time, focusing resources on the best candidates. Memory of successful parameters is updated using a weighted Lehmer mean for F and a weighted arithmetic mean for Cr.
# Exploration mechanisms: Maintained by DE mutation steps using differences of random population members, randomized F parameters from a Cauchy distribution, and initially wide population distributions.
# Exploitation mechanisms: Guided by current-to-pbest mutation pointing towards the best-performing elite subset, and a local search refinement that performs adaptive-step coordinate descent on the best overall individual near the end of the budget.
# Boundary handling: Out-of-bounds mutant components undergo a bounce-back transformation, placing them randomly between the parent parameter and the exceeded boundary, followed by strict clipping.
# Budget strategy: Strictly tracks function evaluations. The search is split into a global phase (90% of budget) and a local refinement phase (last 10% or remaining budget), terminating immediately upon budget exhaustion.
# Closest known influences: SHADE (Tanabe & Fukunaga, 2013), L-SHADE, and classic Hooke-Jeeves coordinate descent.
# Novelty or unusual aspects: Dynamic budget-aware partitioning that guarantees a seamless handoff between the global adaptive DE exploration phase and the coordinate-descent local refinement phase.
# Failure modes: Highly rugged, non-separable functions with massive local optima traps may cause premature convergence during the population reduction phase.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # 1. Read bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)
            
        # 2. Setup budget tracking
        evals_used = 0
        
        def evaluate(x):
            nonlocal evals_used
            if evals_used >= self.budget:
                return float('inf')
            evals_used += 1
            return func(x)
            
        # Divide budget: 90% for global adaptive search, 10% reserved for local search
        local_search_budget = max(20, int(0.10 * self.budget))
        global_search_budget = self.budget - local_search_budget
        
        # 3. Initialization of SHADE
        # Initial population size scales with dimensions
        N_init = int(min(150, max(15, 10 * self.dim)))
        N_min = 6
        N = N_init
        
        # Sample initial population uniformly inside bounds
        population = np.random.uniform(lb, ub, size=(N, self.dim))
        fitness = np.zeros(N)
        
        best_x = None
        best_y = float('inf')
        
        for i in range(N):
            val = evaluate(population[i])
            fitness[i] = val
            if val < best_y:
                best_y = val
                best_x = population[i].copy()
                
        # Historical memory for mutation & crossover rates (SHADE)
        H = 10
        M_F = np.full(H, 0.5)
        M_Cr = np.full(H, 0.5)
        k_mem = 0
        
        # 4. Global Search Phase (SHADE + L-PSR)
        p_min = 2 / N_init
        
        while evals_used < global_search_budget and N >= N_min:
            success_F = []
            success_Cr = []
            success_df = []
            
            # Sort population to determine elite (pbest) pool
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            p_val = max(p_min, np.random.uniform(0.05, 0.20))
            pbest_num = max(1, int(round(p_val * N)))
            
            for i in range(N):
                if evals_used >= global_search_budget:
                    break
                    
                # Sample F and Cr from memory
                r_idx = np.random.randint(0, H)
                
                # Sample Cr from normal distribution
                cr = np.random.normal(M_Cr[r_idx], 0.1)
                cr = np.clip(cr, 0.0, 1.0)
                
                # Sample F from Cauchy distribution
                while True:
                    f = np.random.standard_cauchy() * 0.1 + M_F[r_idx]
                    if f > 0.0:
                        break
                f = min(1.0, f)
                
                # Mutation: current-to-pbest/1
                # Select pbest
                pbest_idx = np.random.randint(0, pbest_num)
                x_pbest = population[pbest_idx]
                
                # Select r1 (different from i)
                r1 = np.random.randint(0, N)
                while r1 == i:
                    r1 = np.random.randint(0, N)
                    
                # Select r2 (different from i and r1)
                r2 = np.random.randint(0, N)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, N)
                    
                v_mut = population[i] + f * (x_pbest - population[i]) + f * (population[r1] - population[r2])
                
                # Boundary handling: bounce-back towards parent
                out_lower = v_mut < lb
                out_upper = v_mut > ub
                v_mut[out_lower] = (population[i][out_lower] + lb[out_lower]) / 2.0
                v_mut[out_upper] = (population[i][out_upper] + ub[out_upper]) / 2.0
                v_mut = np.clip(v_mut, lb, ub)
                
                # Binomial crossover
                j_rand = np.random.randint(0, self.dim)
                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < cr or j == j_rand:
                        trial[j] = v_mut[j]
                        
                # Selection
                trial_fitness = evaluate(trial)
                if trial_fitness < fitness[i]:
                    diff = fitness[i] - trial_fitness
                    success_df.append(diff)
                    success_F.append(f)
                    success_Cr.append(cr)
                    
                    fitness[i] = trial_fitness
                    population[i] = trial
                    
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()
                        
            # Update memory if success occurred
            if len(success_F) > 0:
                success_df = np.array(success_df)
                success_F = np.array(success_F)
                success_Cr = np.array(success_Cr)
                
                weights = success_df / np.sum(success_df)
                
                # Weighted Lehmer mean for F
                sum_w_F = np.sum(weights * success_F)
                if sum_w_F > 1e-9:
                    mean_F_L = np.sum(weights * (success_F ** 2)) / sum_w_F
                else:
                    mean_F_L = M_F[k_mem]
                    
                # Weighted arithmetic mean for Cr
                mean_Cr = np.sum(weights * success_Cr)
                
                M_F[k_mem] = mean_F_L
                M_Cr[k_mem] = mean_Cr
                k_mem = (k_mem + 1) % H
                
            # Linear Population Size Reduction (L-PSR)
            N_target = int(N_init - (evals_used / global_search_budget) * (N_init - N_min))
            N_target = max(N_min, N_target)
            if N_target < N:
                # Discard worst performing individuals
                sorted_idx = np.argsort(fitness)
                population = population[sorted_idx[:N_target]]
                fitness = fitness[sorted_idx[:N_target]]
                N = N_target
                
        # 5. Local Search Phase (Adaptive Coordinate Descent refinement)
        # Allocate any remaining budget to fine local tuning around best_x
        remaining_budget = self.budget - evals_used
        if remaining_budget > 0 and best_x is not None:
            # Step sizes scaled per coordinate based on initial range
            step_sizes = 0.05 * (ub - lb)
            improved = True
            
            # Simple local search loops while budget is available
            while improved and evals_used < self.budget:
                improved = False
                for d in range(self.dim):
                    if evals_used >= self.budget:
                        break
                        
                    for direction in [-1.0, 1.0]:
                        if evals_used >= self.budget:
                            break
                            
                        candidate = best_x.copy()
                        candidate[d] += direction * step_sizes[d]
                        candidate = np.clip(candidate, lb, ub)
                        
                        cand_y = evaluate(candidate)
                        
                        if cand_y < best_y:
                            best_y = cand_y
                            best_x = candidate.copy()
                            step_sizes[d] *= 1.5  # Expand step size
                            improved = True
                            break
                    else:
                        step_sizes[d] *= 0.5  # Contract step size if both directions fail
                        
        return best_x, best_y
