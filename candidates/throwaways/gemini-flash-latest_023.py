# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust Differential Evolution (DE) implementation using the DE/rand/1/bin strategy. It features a small population size for efficiency in low-budget scenarios and an automated restart mechanism to handle multi-modality and stagnation in high-budget scenarios.
# Search state: The population of candidate vectors (pop) and their corresponding objective function values (fitness).
# Candidate generation: Mutation is performed using the classic DE/rand/1 strategy (base vector plus a scaled difference of two others). This is followed by binomial crossover to create a trial vector.
# Selection and replacement: Greedy selection is used, where the trial vector replaces the target vector only if its objective value is less than or equal to the target's value.
# Adaptation: The mutation scale factor (F) is dithered randomly in the range [0.5, 1.0] for each candidate to balance global and local search. The crossover probability (Cr) is fixed at 0.9.
# Exploration mechanisms: Initial random sampling across the search space, the stochastic nature of the DE mutation operator, and a restart mechanism that triggers if the population's fitness variance drops below a threshold.
# Exploitation mechanisms: Greedy replacement and binomial crossover, which encourages the search to follow directions of improvement discovered in the population.
# Boundary handling: Simple clipping of candidate vectors to the provided lower and upper bounds.
# Budget strategy: Explicit tracking of evaluations to ensure the budget is never exceeded. The search loop checks the count before every function call.
# Closest known influences: Storn & Price (1997) Differential Evolution.
# Novelty or unusual aspects: Combines a standard DE with a simple population-variance-based restart to ensure progress continues even after local convergence if budget remains.
# Failure modes: May struggle with extremely high-dimensional landscapes where the population size cannot sufficiently cover the search space within the given budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initialize the optimizer.
        :param budget: Total number of function evaluations allowed.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_x = None
        self.best_y = float('inf')

    def __call__(self, func):
        """
        Run the optimization process.
        :param func: Objective function to minimize.
        :return: (best_x, best_y)
        """
        # Retrieve bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback for safety (though not expected in GNBG)
            lb = np.full(self.dim, -10.0)
            ub = np.full(self.dim, 10.0)

        # Population size: rule of thumb but constrained for efficiency
        pop_size = max(10, min(100, 10 * self.dim))
        
        # Hyperparameters
        cr = 0.9  # Crossover probability
        
        self.evals = 0
        self.best_y = float('inf')
        self.best_x = None

        while self.evals < self.budget:
            # Initialize population
            pop = lb + np.random.rand(pop_size, self.dim) * (ub - lb)
            fitness = np.full(pop_size, float('inf'))

            # Initial evaluation of the population
            for i in range(pop_size):
                if self.evals >= self.budget:
                    return self.best_x, self.best_y
                
                val = func(pop[i])
                self.evals += 1
                fitness[i] = val
                
                if val < self.best_y:
                    self.best_y = val
                    self.best_x = pop[i].copy()

            # Evolution loop
            while self.evals < self.budget:
                # Check for convergence to trigger a restart
                if np.max(fitness) - np.min(fitness) < 1e-13:
                    break

                for i in range(pop_size):
                    if self.evals >= self.budget:
                        break

                    # Mutation: DE/rand/1
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    abc = np.random.choice(idxs, 3, replace=False)
                    a, b, c = pop[abc]
                    
                    # Dithered mutation factor
                    f_val = np.random.uniform(0.5, 1.0)
                    mutant = a + f_val * (b - c)
                    
                    # Binomial Crossover
                    cross_points = np.random.rand(self.dim) < cr
                    # Ensure at least one dimension is inherited from mutant
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    
                    trial = np.where(cross_points, mutant, pop[i])
                    
                    # Boundary handling: clipping
                    trial = np.clip(trial, lb, ub)
                    
                    # Selection
                    val = func(trial)
                    self.evals += 1
                    
                    if val <= fitness[i]:
                        fitness[i] = val
                        pop[i] = trial
                        if val < self.best_y:
                            self.best_y = val
                            self.best_x = trial.copy()

        return self.best_x, self.best_y
