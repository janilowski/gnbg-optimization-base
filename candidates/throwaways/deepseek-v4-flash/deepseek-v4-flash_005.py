# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A differential evolution (DE) variant with population-based search, adaptive crossover rate, and jitter-based mutation. It maintains a small population and uses deterministic crowding for diversity preservation.
# Search state: A population of candidate solutions stored in a 2D numpy array of shape (population_size, dim), with associated fitness values in a parallel array.
# Candidate generation: For each parent, generate a trial vector using DE/rand/1/binomial crossover with jitter applied to the scaling factor F. Three distinct population members are selected for mutation.
# Selection and replacement: Deterministic crowding: if the trial vector is better than the nearest parent (by Euclidean distance), it replaces that parent; otherwise it is discarded. This preserves diversity by preventing premature convergence.
# Adaptation: Crossover rate Cr is periodically adjusted based on the ratio of successful mutations; F uses uniform jitter in [0.5, 1.0] for each mutation.
# Exploration mechanisms: Jitter on F provides varying perturbation scales; binomial crossover allows mixing components from different parents; the crowding selection maintains population spread.
# Exploitation mechanisms: As the population converges, mutation steps become smaller naturally; successful offspring replace parents, refining good solutions.
# Boundary handling: Midpoint reflection: if a coordinate falls outside bounds, it is reflected back; if still out of bounds, it is clamped to the nearest bound.
# Budget strategy: Uses all budget exactly: generates one trial per function call, stops when budget exhausted. Budget check happens before each evaluation.
# Closest known influences: Classic DE/rand/1/bin with deterministic crowding (Mahfoud, 1995) and adaptive Cr tracking.
# Novelty or unusual aspects: Small fixed population size (max(10, dim*2)) combined with crowding selection for multimodal optimization; simple adaptive Cr mechanism.
# Failure modes: May get stuck in local optima on highly multimodal functions; small population limits exploration on very high-dimensional problems; reflection boundary handling may cause clustered solutions near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
        # Population size: small to medium, scaled with dimension
        self.pop_size = max(10, min(50, dim * 2))
        
        # DE parameters
        self.F = 0.8           # Base scaling factor
        self.Cr = 0.9          # Crossover rate
        self.Cr_memory = 0.9   # Running successful Cr for adaptation
        
        # Adaptation tracking
        self.success_counter = 0
        self.total_trials = 0
        self.adapt_frequency = max(10, dim * 5)
        
        # Initialize population
        self.population = None
        self.fitness = None
        self.evaluations = 0
        self.budget_exhausted = False
        
    def _initialize_population(self, lower, upper):
        """Initialize population uniformly in the search space."""
        self.population = np.random.uniform(lower, upper, size=(self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        
    def _boundary_handling(self, x, lower, upper):
        """Midpoint reflection boundary handling."""
        for i in range(len(x)):
            if x[i] < lower[i]:
                x[i] = lower[i] + (lower[i] - x[i])
            elif x[i] > upper[i]:
                x[i] = upper[i] - (x[i] - upper[i])
            # If still out of bounds, clamp
            if x[i] < lower[i]:
                x[i] = lower[i]
            elif x[i] > upper[i]:
                x[i] = upper[i]
        return x
    
    def _nearest_parent(self, trial, parent_indices):
        """Find the nearest parent to the trial vector among candidate parents."""
        candidates = self.population[parent_indices]
        distances = np.sum((candidates - trial) ** 2, axis=1)
        return parent_indices[np.argmin(distances)]
    
    def __call__(self, func):
        # Get bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.array(func.lower, dtype=float)
            upper = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lower = np.array(func.bounds.lb, dtype=float)
            upper = np.array(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub")
        
        # Initialize
        self._initialize_population(lower, upper)
        
        # Evaluate initial population
        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                self.budget_exhausted = True
                break
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
        
        if self.budget_exhausted:
            best_idx = np.argmin(self.fitness)
            return self.population[best_idx].copy(), self.fitness[best_idx]
        
        # Main DE loop with deterministic crowding
        generation = 0
        while self.evaluations < self.budget:
            generation += 1
            self.total_trials = 0
            self.success_counter = 0
            
            # Shuffle population indices for random parent selection
            indices = np.random.permutation(self.pop_size)
            
            for idx in indices:
                if self.evaluations >= self.budget:
                    break
                
                # Select three distinct random individuals different from current
                available = [i for i in range(self.pop_size) if i != idx]
                r1, r2, r3 = np.random.choice(available, size=3, replace=False)
                
                # Mutation with jitter on F
                F_jitter = self.F * (0.5 + np.random.random() * 0.5)
                mutant = self.population[r1] + F_jitter * (self.population[r2] - self.population[r3])
                
                # Binomial crossover
                trial = self.population[idx].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.random() < self.Cr or j == j_rand:
                        trial[j] = mutant[j]
                
                # Boundary handling
                trial = self._boundary_handling(trial, lower, upper)
                
                # Evaluate trial
                trial_fitness = func(trial)
                self.evaluations += 1
                self.total_trials += 1
                
                # Deterministic crowding: replace nearest parent if trial is better
                parent_indices = [idx]
                
                # Also consider the three mutation parents as potential crowding targets
                # to maintain better diversity
                crowding_candidates = [idx, r1, r2, r3]
                nearest_idx = self._nearest_parent(trial, crowding_candidates)
                
                if trial_fitness < self.fitness[nearest_idx]:
                    self.fitness[nearest_idx] = trial_fitness
                    self.population[nearest_idx] = trial
                    self.success_counter += 1
            
            # Adaptive Cr update
            if generation % self.adapt_frequency == 0 and self.total_trials > 0:
                success_rate = self.success_counter / self.total_trials
                if success_rate > 0.3:
                    self.Cr = min(0.95, self.Cr * 1.05)
                elif success_rate < 0.1:
                    self.Cr = max(0.3, self.Cr * 0.95)
        
        # Return best solution found
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx].copy(), self.fitness[best_idx]
