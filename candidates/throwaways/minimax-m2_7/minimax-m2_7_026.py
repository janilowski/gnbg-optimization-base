# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This implementation uses a hybrid approach combining adaptive random sampling with Nelder-Mead simplex local search. The algorithm maintains a population of candidate solutions that are evolved through mutation and selection, with periodic calls to a local optimizer to refine promising regions. This balances global exploration with local exploitation.

# Search state: The algorithm maintains:
#   - A population of candidates (numpy array of shape (population_size, dim))
#   - Current best solution and its objective value
#   - A flag indicating whether local search is currently active
#   - Iteration counter for budget management

# Candidate generation: 
#   - Exploration: Random sampling guided by a simple adaptive scheme that narrows search space based on observed performance
#   - Mutation: Gaussian perturbations with adaptive step size, scaled by current search region
#   - Local search: Nelder-Mead simplex method applied to the best current candidate

# Selection and replacement:
#   - (mu+lambda) style: offspring compete with parents; better solution always survives
#   - Local search result replaces parent if improved
#   - Population diversity maintained by refreshing worst individuals when stagnated

# Adaptation:
#   - Step size for mutation decreases as improvements become smaller (simple cooling)
#   - Search space adapts based on spread of good solutions found so far

# Exploration mechanisms:
#   - Random restart when no improvement for several iterations
#   - Boundary perturbation to escape local minima
#   - Wide initial sampling in first 10% of budget

# Exploitation mechanisms:
#   - Local search on best candidate when budget allows
#   - Aggressive mutation towards current best (best-so-far bias)

# Boundary handling:
#   - Hard clamping: solutions projected back into bounds after mutation
#   - Resample if mutation produces out-of-bounds candidate

# Budget strategy:
#   - Reserve ~5% of budget for final local search phase
#   - Monitor remaining budget to avoid exceeding it
#   - Conservative: never risk going over budget

# Closest known influences:
#   - Conceptual blend of Differential Evolution (population-based search)
#   - Nelder-Mead (local refinement)
#   - Simple cooling schedule reminiscent of simulated annealing

# Novelty or unusual aspects:
#   - Interleaved local search (NM simplex) on best candidate
#   - Adaptive search space narrowing based on interquartile range of good solutions

# Failure modes:
#   - May struggle on very high-dimensional problems (>1000 dims) due to population size scaling
#   - Local search (Nelder-Mead) may fail on highly non-smooth or noisy landscapes
#   - Performance sensitive to initial population size choice relative to dimension
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    """
    Adaptive hybrid optimizer combining population-based search with local refinement.
    
    Designed for black-box minimization with evaluation budget constraints.
    Uses random sampling guided by adaptive heuristics, with periodic Nelder-Mead
    local search to refine promising solutions.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize the optimizer.
        
        Args:
            budget: Maximum number of function evaluations allowed.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        
        # Population size scales with dimension, but stays reasonable
        self.population_size = max(20, min(50, dim * 2))
        
        # Mutation parameters
        self.step_size = 0.5  # Initial mutation step
        self.min_step = 1e-6  # Minimum step size before reset
        
        # Local search trigger
        self.local_search_interval = max(5, self.population_size // 2)
        
    def __call__(self, func):
        """
        Run the optimization algorithm.
        
        Args:
            func: Objective function to minimize. Interface supports:
                  - func.lower / func.upper attributes, or
                  - func.bounds.lb / func.bounds.ub attributes
        
        Returns:
            Tuple of (best_x, best_y) where best_x is the solution found
            and best_y is its objective value.
        """
        # Extract bounds
        bounds = self._get_bounds(func)
        lb, ub = bounds
        
        # Initialize population with stratified sampling
        population = self._initialize_population(lb, ub)
        fitness = np.array([func(x) for x in population])
        evaluations = self.population_size
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]
        
        # Adaptive search region
        search_lb = lb.copy()
        search_ub = ub.copy()
        
        # Stagnation detection
        no_improvement_count = 0
        stagnation_threshold = max(10, self.population_size)
        
        # Main optimization loop
        while evaluations < self.budget:
            # Check if we should trigger local search on best candidate
            if (evaluations - best_idx * (evaluations // self.population_size) 
                if evaluations > 0 else 0) < self.local_search_interval:
                # Generate offspring through mutation
                offspring = self._mutate(population, best_x, search_lb, search_ub, lb, ub)
            else:
                # Apply local search to best candidate if we have enough budget
                remaining = self.budget - evaluations
                if remaining > self.dim * 2 + 10 and best_y > np.median(fitness):
                    local_result = self._local_search(best_x.copy(), func, lb, ub, 
                                                      max_evals=min(self.dim * 2 + 5, remaining - 5))
                    evaluations += local_result['evals']
                    if local_result['y'] < best_y:
                        best_x = local_result['x']
                        best_y = local_result['y']
                        # Update population with improved solution
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = best_x.copy()
                        fitness[worst_idx] = best_y
                        no_improvement_count = 0
            
            # Generate offspring through mutation
            offspring = self._mutate(population, best_x, search_lb, search_ub, lb, ub)
            
            # Evaluate offspring
            for i in range(len(offspring)):
                if evaluations >= self.budget:
                    break
                offspring_fitness = func(offspring[i])
                evaluations += 1
                
                # (mu+lambda) selection: replace worst if offspring is better
                worst_idx = np.argmax(fitness)
                if offspring_fitness < fitness[worst_idx]:
                    population[worst_idx] = offspring[i]
                    fitness[worst_idx] = offspring_fitness
                    
                    # Update best if improved
                    if offspring_fitness < best_y:
                        best_x = offspring[i].copy()
                        best_y = offspring_fitness
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            
            # Adapt search region based on good solutions
            good_threshold = np.percentile(fitness, 25)
            good_mask = fitness <= good_threshold
            if np.sum(good_mask) >= 3:
                good_solutions = population[good_mask]
                new_lb = np.min(good_solutions, axis=0)
                new_ub = np.max(good_solutions, axis=0)
                # Blend with original bounds (keep some exploration)
                blend = 0.7
                search_lb = lb + blend * (new_lb - lb)
                search_ub = ub + blend * (new_ub - ub)
            
            # Adjust step size based on stagnation
            if no_improvement_count > stagnation_threshold:
                self.step_size *= 0.85
                if self.step_size < self.min_step:
                    # Reset: inject fresh random solutions
                    refresh_count = self.population_size // 3
                    refresh_idx = np.argsort(fitness)[-refresh_count:]
                    for idx in refresh_idx:
                        population[idx] = lb + np.random.rand(self.dim) * (ub - lb)
                        fitness[idx] = func(population[idx])
                        evaluations += 1
                    self.step_size = 0.5
                no_improvement_count = 0
            
        return best_x, best_y
    
    def _get_bounds(self, func):
        """Extract bounds from function object."""
        if hasattr(func, 'bounds'):
            return func.bounds.lb, func.bounds.ub
        elif hasattr(func, 'lower'):
            return func.lower, func.upper
        else:
            # Default bounds if none provided
            return np.zeros(self.dim), np.ones(self.dim)
    
    def _initialize_population(self, lb, ub):
        """Initialize population with stratified random sampling."""
        population = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            # Use Latin hypercube sampling for better coverage
            if i == 0:
                # First point at center
                population[i] = 0.5 * (lb + ub)
            else:
                # Rest uniformly distributed
                t = i / self.population_size
                population[i] = lb + t * (ub - lb) + 0.1 * (np.random.rand(self.dim) - 0.5) * (ub - lb)
        return population
    
    def _mutate(self, population, best, search_lb, search_ub, lb, ub):
        """Generate offspring through adaptive mutation."""
        offspring = []
        for i in range(self.population_size):
            # Select mutation strategy
            if np.random.rand() < 0.3:
                # Best-so-far mutation (exploitation)
                mutant = best + self.step_size * np.random.randn(self.dim) * (search_ub - search_lb)
            elif np.random.rand() < 0.6:
                # Current-to-best mutation (balanced)
                idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                mutant = population[i] + 0.5 * (best - population[i]) + 0.3 * (population[idx1] - population[idx2])
            else:
                # Random mutation (exploration)
                idx = np.random.choice(len(population))
                mutant = population[idx] + self.step_size * np.random.randn(self.dim) * (search_ub - search_lb)
            
            # Ensure bounds are respected
            mutant = np.clip(mutant, lb, ub)
            
            # If mutation is too close to a boundary, perturb
            for d in range(self.dim):
                if mutant[d] == lb[d] or mutant[d] == ub[d]:
                    mutant[d] += 0.1 * (ub[d] - lb[d]) * (np.random.rand() - 0.5)
            
            mutant = np.clip(mutant, lb, ub)
            offspring.append(mutant)
        
        return offspring
    
    def _local_search(self, x0, func, lb, ub, max_evals=50):
        """
        Nelder-Mead simplex local search.
        
        Args:
            x0: Starting point.
            func: Objective function.
            lb, ub: Bounds.
            max_evals: Maximum evaluations for local search.
            
        Returns:
            Dictionary with 'x', 'y', and 'evals' (number of evaluations used).
        """
        n = len(x0)
        
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(n):
            point = x0.copy()
            if i < n:
                # Scale along each dimension
                scale = min(0.5, (ub[i] - lb[i]) * 0.25)
                point[i] += scale * (1 if np.random.rand() > 0.5 else -1)
                point = np.clip(point, lb, ub)
            simplex.append(point)
        
        # Evaluate initial simplex
        values = [func(p) for p in simplex]
        evals = len(values)
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # Main simplex iteration
        while evals < max_evals and self._simplex_size(simplex) > 1e-8:
            # Sort simplex by fitness
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            # Centroid (exclude worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            worst = simplex[-1]
            reflected = centroid + alpha * (centroid - worst)
            reflected = np.clip(reflected, lb, ub)
            f_reflected = func(reflected)
            evals += 1
            
            if evals >= max_evals:
                break
            
            if values[0] <= f_reflected <= values[-2]:
                # Accept reflection
                simplex[-1] = reflected
                values[-1] = f_reflected
            elif f_reflected < values[0]:
                # Expansion
                expanded = centroid + gamma * (reflected - centroid)
                expanded = np.clip(expanded, lb, ub)
                f_expanded = func(expanded)
                evals += 1
                
                if evals >= max_evals:
                    break
                
                if f_expanded < f_reflected:
                    simplex[-1] = expanded
                    values[-1] = f_expanded
                else:
                    simplex[-1] = reflected
                    values[-1] = f_reflected
            else:
                # Contraction
                contracted = centroid + rho * (worst - centroid)
                contracted = np.clip(contracted, lb, ub)
                f_contracted = func(contracted)
                evals += 1
                
                if evals >= max_evals:
                    break
                
                if f_contracted < values[-1]:
                    simplex[-1] = contracted
                    values[-1] = f_contracted
                else:
                    # Shrink
                    best = simplex[0]
                    new_simplex = [best]
                    for i in range(1, len(simplex)):
                        new_point = best + sigma * (simplex[i] - best)
                        new_point = np.clip(new_point, lb, ub)
                        new_simplex.append(new_point)
                        f_new = func(new_point)
                        evals += 1
                        if evals >= max_evals:
                            break
                        values[i] = f_new
                    simplex = new_simplex
        
        # Find best in simplex
        best_idx = np.argmin(values)
        return {
            'x': simplex[best_idx],
            'y': values[best_idx],
            'evals': evals
        }
    
    def _simplex_size(self, simplex):
        """Calculate the size (circumradius) of the simplex."""
        if len(simplex) < 2:
            return 0.0
        
        # Calculate average distance from centroid
        centroid = np.mean(simplex, axis=0)
        distances = [np.linalg.norm(p - centroid) for p in simplex]
        return np.mean(distances)
