import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simplified Success-History Based Adaptive Differential Evolution (SHADE) algorithm.
# Search state: A population of candidate solutions, their fitness values, and a memory of successful search parameters (Scale Factor F and Crossover Rate CR).
# Candidate generation: Uses the 'DE/current-to-pbest/1' mutation strategy combined with binomial crossover. 
# Selection and replacement: Standard DE selection: a trial vector replaces its parent if it has a lower or equal objective value.
# Adaptation: Parameters F and CR are sampled from distributions (Cauchy for F, Normal for CR) centered around historical successful values.
# Exploration mechanisms: Random selection of vectors for mutation (r1, r2) and the diversity-maintaining properties of the DE population.
# Exploitation mechanisms: The 'pbest' component in the mutation strategy directs search towards the top-performing individuals in the current population.
# Boundary handling: Individuals exceeding bounds are moved halfway between their previous position and the violated boundary to maintain diversity while staying feasible.
# Budget strategy: Evaluations are counted strictly; the population size is capped and potentially reduced for very small budgets to ensure at least a few generations occur.
# Closest known influences: SHADE (Tanabe and Fukunaga, 2013), JADE (Zhang and Sanderson, 2009).
# Novelty or unusual aspects: Simplified parameter memory (single history slot) and aggressive population size management to accommodate extremely tight budgets.
# Failure modes: May struggle with highly non-separable or extremely high-dimensional functions (D > 500) where the budget is insufficient for population-based convergence.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
        # Adaptive population size logic
        # Aim for at least 10-20 generations, but keep population within reasonable bounds for DE
        self.pop_size = max(10, min(100, self.dim * 10))
        if self.pop_size * 2 > self.budget:
            self.pop_size = max(4, self.budget // 5)

    def _get_bounds(self, func):
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = func.lower, func.upper
        elif hasattr(func, 'bounds'):
            lb, ub = func.bounds.lb, func.bounds.ub
        else:
            # Fallback if no bounds are provided
            lb, ub = -5.0 * np.ones(self.dim), 5.0 * np.ones(self.dim)
        return np.array(lb, dtype=float), np.array(ub, dtype=float)

    def __call__(self, func):
        lb, ub = self._get_bounds(func)
        
        # Initialize population
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.zeros(self.pop_size)
        
        best_idx = 0
        evals = 0
        
        for i in range(self.pop_size):
            if evals >= self.budget:
                # Handle cases where budget is smaller than pop_size
                fitness[i:] = np.inf
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < fitness[best_idx]:
                best_idx = i

        # Adaptation parameters (Mean CR and Mean F)
        m_cr, m_f = 0.5, 0.5
        p_best_rate = 0.1  # Top 10% for mutation

        while evals < self.budget:
            success_cr = []
            success_f = []
            
            # Sort indices for p-best selection
            indices_sorted = np.argsort(fitness)
            
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample F and CR
                cr = np.clip(np.random.normal(m_cr, 0.1), 0, 1)
                f = -1.0
                while f <= 0:
                    f = np.random.standard_cauchy() * 0.1 + m_f
                f = min(f, 1.0)
                
                # Mutation: current-to-pbest/1
                p_best_idx = indices_sorted[np.random.randint(0, max(2, int(self.pop_size * p_best_rate)))]
                
                r1 = np.random.randint(0, self.pop_size)
                while r1 == i: r1 = np.random.randint(0, self.pop_size)
                
                r2 = np.random.randint(0, self.pop_size)
                while r2 == i or r2 == r1: r2 = np.random.randint(0, self.pop_size)
                
                v_mut = pop[i] + f * (pop[p_best_idx] - pop[i]) + f * (pop[r1] - pop[r2])
                
                # Binomial Crossover
                j_rand = np.random.randint(0, self.dim)
                mask = np.random.rand(self.dim) < cr
                mask[j_rand] = True
                trial = np.where(mask, v_mut, pop[i])
                
                # Boundary Handling: midpoint correction
                low_mask = trial < lb
                high_mask = trial > ub
                trial[low_mask] = (lb[low_mask] + pop[i][low_mask]) / 2.0
                trial[high_mask] = (ub[high_mask] + pop[i][high_mask]) / 2.0
                
                # Selection
                f_trial = func(trial)
                evals += 1
                
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    success_cr.append(cr)
                    success_f.append(f)
                    if f_trial < fitness[best_idx]:
                        best_idx = i

            # Update adaptation means (Lehmer mean for F)
            if success_cr:
                m_cr = 0.9 * m_cr + 0.1 * np.mean(success_cr)
                m_f = 0.9 * m_f + 0.1 * (np.sum(np.array(success_f)**2) / np.sum(success_f))

        return pop[best_idx], fitness[best_idx]
