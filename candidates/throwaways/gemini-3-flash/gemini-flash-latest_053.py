# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust Differential Evolution (DE) implementation using the DE/rand/1/bin strategy, optimized for constrained budgets and black-box environments.
# Search state: A population of candidate vectors and their corresponding objective values.
# Candidate generation: New candidates are created using differential mutation (adding a scaled difference of two random members to a third) followed by binomial crossover with the target individual.
# Selection and replacement: A greedy selection mechanism is used where a trial vector replaces its parent in the population only if its objective value is lower or equal.
# Adaptation: The mutation scale factor (F) is dithered (randomized per generation) to improve search robustness across different landscape scales.
# Exploration mechanisms: Differential mutation provides global exploration, while the 'rand/1' strategy ensures diversity is maintained longer than 'best/1' strategies.
# Exploitation mechanisms: Binomial crossover and greedy selection allow the population to converge on promising basins of attraction.
# Boundary handling: Candidate vectors are clipped to the hypercube defined by the problem bounds to ensure feasibility.
# Budget strategy: A strict evaluation counter is maintained. The population size is dynamically adjusted based on the dimension and total budget to ensure a sufficient number of generations.
# Closest known influences: Storn and Price's original Differential Evolution; SciPy's 'differential_evolution' implementation logic.
# Novelty or unusual aspects: Dynamic population sizing to prevent early budget exhaustion in high dimensions while maintaining search quality in low dimensions.
# Failure modes: May converge slowly on extremely high-dimensional or highly non-separable landscapes if the budget is very small relative to the dimension.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the optimization algorithm.
        
        Args:
            budget (int): Total number of function evaluations allowed.
            dim (int): Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Executes the minimization process.
        
        Args:
            func (callable): The objective function to minimize.
            
        Returns:
            tuple: (best_x, best_y) found during the search.
        """
        # 1. Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = func.lower, func.upper
        elif hasattr(func, 'bounds'):
            lb, ub = func.bounds.lb, func.bounds.ub
        else:
            # Fallback to a standard range if no bounds are provided
            lb, ub = -5.0, 5.0

        # Ensure lb and ub are numpy arrays
        if isinstance(lb, (int, float)):
            lb = np.full(self.dim, lb)
        if isinstance(ub, (int, float)):
            ub = np.full(self.dim, ub)
        lb = np.asarray(lb)
        ub = np.asarray(ub)

        # 2. Setup Search Parameters
        # Scale population size based on dimension but cap it relative to budget
        pop_size = min(max(10, 5 * self.dim), self.budget // 4)
        if pop_size < 4: pop_size = 4  # DE requires at least 4 members
        
        # Differential Evolution Hyperparameters
        cr = 0.9  # Crossover probability
        f_base = 0.8 # Scale factor
        
        # 3. Initialization
        eval_count = 0
        
        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return float('inf')
            eval_count += 1
            return func(x)

        # Initialize population using Latin Hypercube-like random sampling
        pop = lb + np.random.rand(pop_size, self.dim) * (ub - lb)
        fitness = np.array([evaluate(ind) for ind in pop])
        
        # Track global best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # 4. Main Evolution Loop
        while eval_count < self.budget:
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                
                # Mutation: DE/rand/1
                # Select 3 unique indices distinct from current index i
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Dither F for robustness
                f = f_base + 0.2 * (np.random.rand() - 0.5)
                mutant = a + f * (b - c)
                
                # Boundary handling: Clipping
                mutant = np.clip(mutant, lb, ub)
                
                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) < cr
                # Ensure at least one dimension is inherited from the mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                trial_fitness = evaluate(trial)
                if trial_fitness <= fitness[i]:
                    fitness[i] = trial_fitness
                    pop[i] = trial
                    
                    # Update global best
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

            # Optional: Simple stagnation check/restart could go here if budget allows
            # but standard DE is usually sufficient for standard benchmarks.

        return best_x, best_y

# The harness expects Algorithm to be defined.
# The search uses numpy for vectorized operations and randomness.
# It strictly respects the evaluation budget and handles bounds dynamically.
