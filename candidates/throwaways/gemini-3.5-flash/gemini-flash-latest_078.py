# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An adaptive Differential Evolution (JADE) hybrid with localized restarts and archive-based mutation, specifically tuned to handle non-separable, multimodal black-box landscapes.
# Search state: Population of candidate solutions, historical archive of successful mutation and crossover parameters, an archive of recently replaced solutions, and the overall best solution found.
# Candidate generation: Generates mutant vectors using a "current-to-pbest/1" strategy, which mixes the current point, a randomly selected top-tier candidate, and differences from the population and failure archive.
# Selection and replacement: Standard greedy selection where a trial vector replaces its parent if its objective value is lower or equal. Replaced parents are moved to the archive.
# Adaptation: Employs self-adaptation of the mutation scale (F) and crossover rate (CR) using success-history metrics (Lehmer mean for F, arithmetic mean for CR).
# Exploration mechanisms: Handled via a large initial exploration radius, randomized mutant selection, and periodic soft/hard restarts when population diversity falls below a threshold.
# Exploitation mechanisms: "current-to-pbest" mutation directs search towards top performers. If stagnation occurs, half of the population is re-seeded in a tight Gaussian cluster around the global best.
# Boundary handling: Out-of-bounds variables are projected back using a mid-point reflection scheme between the parent's coordinate and the violated bound.
# Budget strategy: Strictly monitors evaluations on every function call. Early termination checks ensure the algorithm halts immediately when the budget is reached.
# Closest known influences: JADE (Adaptive Differential Evolution with Success-History Measure) and L-SHADE.
# Novelty or unusual aspects: A hybrid soft-restart mechanism that blends global re-exploration (50% of the population) with localized refinement (50% around the best-known point) when stagnation is detected.
# Failure modes: Can struggle on extremely high-dimensional spaces with extremely tight budgets where pop-size scale causes premature exhaustion of evaluations before convergence.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        
    def __call__(self, func):
        # Determine bounds robustly
        lb, ub = None, None
        if hasattr(func, 'bounds') and func.bounds is not None:
            if hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
                lb = func.bounds.lb
                ub = func.bounds.ub
            elif isinstance(func.bounds, (list, tuple)) and len(func.bounds) == 2:
                lb, ub = func.bounds
        if lb is None or ub is None:
            lb = getattr(func, 'lower', None)
            ub = getattr(func, 'upper', None)
            
        if lb is None:
            lb = np.full(self.dim, -5.0)
        if ub is None:
            ub = np.full(self.dim, 5.0)
            
        # Ensure numpy arrays
        lb = np.atleast_1d(lb).astype(float)
        ub = np.atleast_1d(ub).astype(float)
        
        # Hyperparameters
        # Adaptive population sizing based on dimension and budget
        pop_size = int(np.clip(10 * self.dim, 10, 100))
        if pop_size * 5 > self.budget:
            pop_size = max(5, self.budget // 5)
            
        p_best_rate = 0.15
        archive_max_size = pop_size
        
        # Adaptation memory
        mu_cr = 0.5
        mu_f = 0.5
        c_adapt = 0.1
        
        # State initialization
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.full(pop_size, np.inf)
        
        # Evaluate initial population safely within budget
        best_x = None
        best_y = np.inf
        
        for i in range(pop_size):
            if self.evals >= self.budget:
                break
            pop[i] = np.clip(pop[i], lb, ub)
            fit = func(pop[i])
            self.evals += 1
            fitness[i] = fit
            if fit < best_y:
                best_y = fit
                best_x = pop[i].copy()
                
        archive = []
        stagnation_counter = 0
        last_best_y = best_y
        
        while self.evals < self.budget:
            # Check for stagnation or convergence to trigger restart
            diversity = np.mean(np.std(pop, axis=0)) if pop_size > 1 else 0.0
            fitness_range = np.max(fitness) - np.min(fitness) if pop_size > 1 else 0.0
            
            if (diversity < 1e-6 or fitness_range < 1e-10 or stagnation_counter > 50) and self.evals < self.budget:
                # Soft restart: half global, half local around best_x
                half = pop_size // 2
                if half > 0:
                    pop[:half] = np.random.uniform(lb, ub, (half, self.dim))
                    scale = 0.05 * (ub - lb)
                    pop[half:] = best_x + np.random.normal(0, scale, (pop_size - half, self.dim))
                else:
                    pop[0] = np.random.uniform(lb, ub, self.dim)
                pop = np.clip(pop, lb, ub)
                
                # Re-evaluate
                for i in range(pop_size):
                    if self.evals >= self.budget:
                        break
                    fit = func(pop[i])
                    self.evals += 1
                    fitness[i] = fit
                    if fit < best_y:
                        best_y = fit
                        best_x = pop[i].copy()
                
                # Reset search state
                archive = []
                mu_cr = 0.5
                mu_f = 0.5
                stagnation_counter = 0
                last_best_y = best_y
                continue
                
            # Parameter generation
            cr_list = np.random.normal(mu_cr, 0.1, pop_size)
            cr_list = np.clip(cr_list, 0.0, 1.0)
            
            f_list = []
            while len(f_list) < pop_size:
                val = mu_f + 0.1 * np.random.standard_cauchy()
                if val > 0:
                    f_list.append(min(val, 1.0))
            f_list = np.array(f_list)
            
            success_cr = []
            success_f = []
            
            # Sort population to identify p-best
            sorted_indices = np.argsort(fitness)
            p_best_count = max(1, int(p_best_rate * pop_size))
            p_best_indices = sorted_indices[:p_best_count]
            
            for i in range(pop_size):
                if self.evals >= self.budget:
                    break
                    
                # Select mutation components
                p_best_idx = np.random.choice(p_best_indices)
                x_pbest = pop[p_best_idx]
                
                # r1 from population (excluding i)
                r1_candidates = [idx for idx in range(pop_size) if idx != i]
                if not r1_candidates:
                    r1_candidates = [i]
                r1 = np.random.choice(r1_candidates)
                x_r1 = pop[r1]
                
                # r2 from union of population and archive (excluding i and r1)
                r2_candidates = [idx for idx in range(pop_size) if idx != i and idx != r1]
                union_pool = [pop[idx] for idx in r2_candidates] + archive
                if not union_pool:
                    union_pool = [pop[i]]
                r2_idx = np.random.choice(len(union_pool))
                x_r2 = union_pool[r2_idx]
                
                # Mutation
                F = f_list[i]
                v = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                
                # Crossover
                CR = cr_list[i]
                j_rand = np.random.randint(self.dim)
                cross_mask = np.random.rand(self.dim) < CR
                cross_mask[j_rand] = True
                
                u = np.where(cross_mask, v, pop[i])
                
                # Boundary handling with midpoint reflection
                out_lower = u < lb
                out_upper = u > ub
                u[out_lower] = 0.5 * (pop[i, out_lower] + lb[out_lower])
                u[out_upper] = 0.5 * (pop[i, out_upper] + ub[out_upper])
                
                # Evaluate trial vector
                trial_fit = func(u)
                self.evals += 1
                
                if trial_fit <= fitness[i]:
                    # Archive current individual before replacement
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max_size:
                        archive.pop(np.random.randint(len(archive)))
                        
                    pop[i] = u
                    fitness[i] = trial_fit
                    success_cr.append(CR)
                    success_f.append(F)
                    
                    if trial_fit < best_y:
                        best_y = trial_fit
