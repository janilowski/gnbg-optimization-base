# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Differential Evolution (DE/rand/1/bin) optimizer with dither for robustness,
# designed for black-box minimization. It adapts to dimensionality and budget.
# Search state: A population of candidate vectors with associated fitness values,
# plus the best solution found so far.
# Candidate generation: For each target vector, three distinct random individuals are selected,
# a mutant is created using base + F * (diff1 - diff2), then binomial crossover with
# probability Cr produces a trial vector.
# Selection and replacement: Greedy selection – if the trial has lower (better) fitness,
# it replaces the target vector in the population.
# Adaptation: The scaling factor F is sampled uniformly from [0.5, 1.0] each generation (dither),
# and the crossover rate Cr is fixed at 0.9.
# Exploration mechanisms: Mutation with random differentials and dither, plus
# binomial crossover; boundary reflection keeps candidates inside the feasible domain.
# Exploitation mechanisms: Greedy replacement retains high-quality solutions,
# and the best solution is tracked throughout the run.
# Boundary handling: Reflective repair – any coordinate outside [lb, ub] is reflected back
# inside the bounds.
# Budget strategy: The algorithm stops immediately when the evaluation budget is exhausted.
# Initial population size is computed as max(4, min(5*dim, budget//5, 100)) but never exceeds
# the budget. For very small budgets (<4 evaluations), it falls back to uniform random sampling.
# Closest known influences: Classic DE (Storn & Price, 1997) with dither (random F each generation).
# Novelty or unusual aspects: None; straightforward implementation with careful budget accounting.
# Failure modes: May converge prematurely on multimodal landscapes due to lack of restart;
# requires population size ≥4 to work correctly; struggles with high-dimensional problems
# if budget is too small.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Differential Evolution optimizer for black-box minimization."""
    
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        # Other initialization is done in __call__ because bounds are needed.
    
    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        # Ensure array shapes
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)
        elif lb.shape[0] != self.dim:
            lb = np.broadcast_to(lb, (self.dim,)).copy()
            ub = np.broadcast_to(ub, (self.dim,)).copy()
        
        evals = 0
        best_x = None
        best_y = np.inf
        
        # Handle very small budgets (cannot form a population of 4)
        if self.budget < 4:
            # Pure random search
            while evals < self.budget:
                x = np.random.uniform(lb, ub, size=self.dim)
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y
        
        # Choose population size
        pop_size = max(4, min(5 * self.dim, self.budget // 5, 100))
        pop_size = min(pop_size, self.budget)  # never exceed budget
        # But pop_size must be at least 4 for DE, already ensured.
        
        # Initialize population uniformly in bounds
        pop = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        fitness = np.empty(pop_size)
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()
        
        # DE loop
        # Use a fixed crossover rate
        Cr = 0.9
        
        while evals < self.budget:
            # Sample scaling factor for this generation (dither)
            F = np.random.uniform(0.5, 1.0)
            
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Choose three distinct random indices != i
                indices = list(range(pop_size))
                indices.remove(i)
                idx = np.random.choice(indices, size=3, replace=False)
                a, b, c = idx
                
                # Mutation: base + F * (diff1 - diff2)
                mutant = pop[a] + F * (pop[b] - pop[c])
                
                # Binomial crossover with target vector pop[i]
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.random(self.dim) < Cr,
                                 mutant, pop[i])
                # Always cross at least one gene (DE/rand/1/bin guarantee)
                trial[j_rand] = mutant[j_rand]
                
                # Boundary reflection
                # Reflect coordinates below lb
                below = trial < lb
                trial[below] = lb[below] + (lb[below] - trial[below])
                # Reflect coordinates above ub
                above = trial > ub
                trial[above] = ub[above] - (trial[above] - ub[above])
                # Second pass: if still out (rare), clip to bounds
                trial = np.clip(trial, lb, ub)
                
                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1
                
                # Selection: replace if better
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()
        
        return best_x, best_y
