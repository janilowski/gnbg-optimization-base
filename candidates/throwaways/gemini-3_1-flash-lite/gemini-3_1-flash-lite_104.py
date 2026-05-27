# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A population-based (mu+lambda) evolution strategy with self-adaptive step sizes.
# Search state: Maintains a population of candidates, their fitness values, and individual strategy vectors (sigma) for mutation.
# Candidate generation: Offspring are created by mutating parents using Gaussian noise scaled by per-individual sigma vectors.
# Selection and replacement: Uses (mu+lambda) survival selection: the best candidates from the union of parents and offspring are kept.
# Adaptation: Step sizes (sigma) are updated using a simple 1/5th success rule heuristic every generation.
# Exploration mechanisms: Initial spread is uniform-random within bounds; mutation injects Gaussian noise proportional to sigma.
# Exploitation mechanisms: Survival selection forces the population toward lower fitness regions; population mean converges locally.
# Boundary handling: Candidates are clipped to the allowed bounds both during generation and initialization.
# Budget strategy: Iterative loop terminates immediately when budget is exhausted.
# Closest known influences: Evolutionary Strategy (ES) with (mu, lambda) selection and adaptive mutation.
# Novelty or unusual aspects: Minimalist implementation using only standard numpy operations without external dependencies.
# Failure modes: Can get trapped in sharp local minima if initial population is not diverse enough or if dimensions are extremely high.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(20 + 2 * dim, budget // 2)
        self.mu = self.pop_size // 2

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        sigma = (ub - lb) * 0.1
        
        best_x = None
        best_y = float('inf')
        eval_count = 0

        # Evaluate initial population
        fitness = np.array([func(x) for x in pop])
        eval_count += self.pop_size
        
        idx = np.argmin(fitness)
        best_x, best_y = pop[idx].copy(), fitness[idx]

        while eval_count < self.budget:
            # Generate offspring
            offspring = []
            for _ in range(self.pop_size):
                parent = pop[np.random.randint(0, self.mu)]
                child = parent + np.random.normal(0, sigma, self.dim)
                offspring.append(np.clip(child, lb, ub))
            
            offspring = np.array(offspring)
            off_fitness = np.array([func(x) for x in offspring])
            eval_count += self.pop_size
            
            # Combine and Select best
            combined_x = np.vstack([pop, offspring])
            combined_f = np.concatenate([fitness, off_fitness])
            
            # Update best found
            best_idx = np.argmin(combined_f)
            if combined_f[best_idx] < best_y:
                best_y = combined_f[best_idx]
                best_x = combined_x[best_idx].copy()
            
            # (mu+lambda) survival
            sort_idx = np.argsort(combined_f)
            pop = combined_x[sort_idx[:self.pop_size]]
            fitness = combined_f[sort_idx[:self.pop_size]]
            
            # Adaptation: simple 1/5th success rule
            success_rate = np.mean(off_fitness < np.median(fitness))
            if success_rate > 0.2:
                sigma *= 1.1
            else:
                sigma *= 0.9

            if eval_count >= self.budget:
                break
                
        return best_x, best_y
