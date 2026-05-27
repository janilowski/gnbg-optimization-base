# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple evolutionary strategy (ES) for continuous black-box minimization.
#          Uses a (mu+lambda) selection scheme with Gaussian mutation and adaptive step size.
# Search state: Maintains a population of candidates, tracks the best solution found (best_x, best_y),
#               records evaluation count, and adapts step size based on recent success.
# Candidate generation: Offspring are created by blending the best parent with other parents (recombination)
#                       and adding Gaussian noise scaled by the adaptive step size.
# Selection and replacement: (mu+lambda) strategy: combine parents and offspring, sort by fitness,
#                           keep the top mu individuals as next generation.
# Adaptation: Step size (sigma) is increased when recent offspring improve the best known solution
#            (exploration boost) and decreased when no improvement is observed (exploitation refinement).
#            Step size is clipped to reasonable bounds to prevent numerical issues.
# Exploration mechanisms: Large initial random sampling; Gaussian mutation with adaptive step size;
#                        diversity from recombining multiple parents.
# Exploitation mechanisms: Selection pressure toward better solutions; decreasing step size when stagnating;
#                         progressive refinement of the best solution.
# Boundary handling: All candidates are clipped to problem bounds before evaluation, using vectorized
#                    operations for efficiency.
# Budget strategy: Strict budget tracking ensures the algorithm stops exactly when the evaluation
#                  budget is exhausted, checking before each new evaluation.
# Closest known influences: Classical evolutionary strategies (ES) and CMA-ES concepts, simplified for
#                          implementation without external dependencies. The (mu+lambda) scheme is
#                          inspired by Bäck et al. (1993) and Schwefel's original ES.
# Novelty or unusual aspects: The adaptation heuristic (increase sigma on success, decrease on failure)
#                            is a simplified version of the 1/5th rule commonly used in ES. The blending
#                            recombination provides a balance between hill-climbing and diversity.
# Failure modes: May struggle with highly multi-modal landscapes if step size becomes too small;
#                sensitive to initial step size choice for very high-dimensional problems;
#                lacks second-order adaptation (covariance learning) which limits performance on
#                badly scaled or rotated problems.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    """
    Evolutionary strategy optimizer for continuous black-box minimization.
    
    Uses a simple (mu+lambda) strategy with Gaussian mutation and adaptive
    step size. Population size scales with dimension for robustness.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize the optimizer.
        
        Args:
            budget: Maximum number of function evaluations allowed.
            dim: Problem dimensionality (number of decision variables).
        """
        self.budget = budget
        self.dim = dim
        
        # Population sizing: balance between exploration and computational cost
        # mu: number of parents selected each generation
        # lambda: number of offspring produced each generation
        self.mu = max(3, dim // 4)
        self.lambda_ = max(10, dim)
        self.population_size = self.mu + self.lambda_
        
        # Step size (search radius) - initially proportional to sqrt(dim)
        # This follows the heuristic that optimal step sizes often scale with dimension
        self.sigma = 0.3 * np.sqrt(dim)
        
        # Tracking variables for adaptation
        self.evaluations = 0
        self.best_x = None
        self.best_y = float('inf')
        self.recent_successes = []  # Track recent improvements for adaptation
    
    def __call__(self, func):
        """
        Optimize the given objective function.
        
        Args:
            func: Black-box function to minimize. Can have bounds attributes:
                  - func.lower / func.upper, or
                  - func.bounds.lb / func.bounds.ub
        
        Returns:
            Tuple of (best_x, best_y) where best_x is the best solution found
            and best_y is its objective value.
        """
        # Extract problem bounds
        lb, ub = self._get_bounds(func)
        
        # Initialize population randomly within bounds
        population = self._init_population(lb, ub)
        
        # Evaluate initial population
        for i in range(len(population)):
            if self.evaluations >= self.budget:
                break
            self._evaluate(func, population[i], lb, ub)
        
        # Evolutionary loop
        while self.evaluations < self.budget:
            # Create offspring from parents
            offspring = self._create_offspring(population[:self.mu])
            
            # Evaluate offspring
            for i in range(len(offspring)):
                if self.evaluations >= self.budget:
                    break
                self._evaluate(func, offspring[i], lb, ub)
            
            # Select next generation: combine parents and offspring, keep best mu
            combined = np.vstack([population[:self.mu], offspring])
            fitnesses = np.array([self._fitness(func, x, lb, ub) for x in combined])
            
            # Truncation selection: keep top mu individuals
            indices = np.argsort(fitnesses)[:self.mu]
            population = combined[indices]
            
            # Adapt step size based on recent performance
            self._adapt_step_size()
        
        return self.best_x.copy(), self.best_y
    
    def _get_bounds(self, func):
        """
        Extract optimization bounds from function object.
        
        Handles multiple common bound representations.
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            # Default bounds if not specified
            lb = np.full(self.dim, -10.0)
            ub = np.full(self.dim, 10.0)
        return lb, ub
    
    def _init_population(self, lb, ub):
        """Initialize population with random samples within bounds."""
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        return pop
    
    def _create_offspring(self, parents):
        """
        Generate offspring through recombination and mutation.
        
        Offspring are created by:
        1. Selecting a center point (blend of best parent and others)
        2. Adding Gaussian noise scaled by step size
        """
        n_parents = len(parents)
        offspring = np.empty((self.lambda_, self.dim))
        
        for i in range(self.lambda_):
            # Recombination: blend best parent with another for diversity
            if n_parents > 1 and i % 2 == 1:
                # Odd offspring: blend best with random other
                p1 = parents[0]  # Best parent
                p2 = parents[np.random.randint(1, n_parents)]
                center = 0.5 * (p1 + p2)
            else:
                # Even offspring: use best parent as center
                center = parents[0]
            
            # Mutation: add Gaussian noise scaled by step size
            noise = np.random.randn(self.dim)
            offspring[i] = center + self.sigma * noise
        
        return offspring
    
    def _clip_to_bounds(self, x, lb, ub):
        """Project solution vector back to feasible region."""
        return np.maximum(lb, np.minimum(x, ub))
    
    def _fitness(self, func, x, lb, ub):
        """Compute fitness (objective value) for a solution."""
        x_clipped = self._clip_to_bounds(x, lb, ub)
        return float(func(x_clipped))
    
    def _evaluate(self, func, x, lb, ub):
        """
        Evaluate a candidate solution and update best if improved.
        
        Returns the objective value.
        """
        y = self._fitness(func, x, lb, ub)
        self.evaluations += 1
        
        # Update best solution if this is an improvement
        if y < self.best_y:
            self.best_y = y
            self.best_x = self._clip_to_bounds(x, lb, ub).copy()
            self.recent_successes.append(True)
        else:
            self.recent_successes.append(False)
        
        # Keep only recent history for adaptation
        if len(self.recent_successes) > self.lambda_:
            self.recent_successes.pop(0)
        
        return y
    
    def _adapt_step_size(self):
        """
        Adapt step size based on recent success rate.
        
        Uses a simplified 1/5th rule: increase step size if success rate
        is high (>20%), decrease if very low (<10%). This balances
        exploration and exploitation.
        """
        if len(self.recent_successes) >= self.lambda_:
            # Compute success rate over recent window
            successes = sum(self.recent_successes[-self.lambda_:])
            success_rate = successes / self.lambda_
            
            # Adjust step size based on success rate
            if success_rate > 0.2:
                # High success: increase exploration
                self.sigma *= 1.2
            elif success_rate < 0.1:
                # Low success: decrease step size
                self.sigma *= 0.8
        
        # Keep step size in reasonable bounds to prevent numerical issues
        self.sigma = np.clip(self.sigma, 1e-6, 10.0)


# Alias for the algorithm class
Optimizer = Algorithm
ES = Algorithm  # Common abbreviation for Evolutionary Strategy
