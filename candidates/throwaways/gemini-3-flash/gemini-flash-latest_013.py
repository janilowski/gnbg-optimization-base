# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust Self-Adaptive Differential Evolution (jDE) variant.
# Search state: A population of candidate vectors, their objective values, and individual-specific control parameters (F and CR).
# Candidate generation: Uses the DE/rand/1/bin strategy where a mutant vector is created by scaling the difference between two random members and adding it to a third. Binomial crossover is then applied.
# Selection and replacement: Greedy selection; an offspring replaces its parent only if it achieves a lower or equal objective value.
# Adaptation: Employs the jDE mechanism where control parameters F (mutation scale) and CR (crossover probability) are updated with a small probability for each individual in each generation, allowing the search to evolve its own hyperparameters.
# Exploration mechanisms: Differential mutation provides global exploration, while the stochastic nature of F and CR allows the algorithm to escape stagnation.
# Exploitation mechanisms: Greedy selection and the "bin" crossover favor retaining good components of successful solutions.
# Boundary handling: Offspring vectors that exceed the search space boundaries are clipped to the nearest valid value (clamping).
# Budget strategy: The algorithm tracks the number of evaluations and terminates immediately once the budget is exhausted, returning the best solution found. Initial population evaluation is prioritized.
# Closest known influences: jDE (Brest et al., 2006) and standard Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: Compact implementation of parameter self-adaptation specifically tuned for black-box constraints and variable dimensions.
# Failure modes: Like most DE variants, it may converge prematurely on highly multi-modal landscapes if the population size is too small or the budget is extremely restricted relative to the dimension.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the Self-Adaptive Differential Evolution algorithm.
        
        Args:
            budget: Maximum number of function evaluations.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        # Population size: heuristic choice between 10 and 100 based on dimension.
        self.N = min(100, max(10, 5 * dim))
        
        # jDE adaptation parameters
        self.tau1 = 0.1  # Probability to update F
        self.tau2 = 0.1  # Probability to update CR
        self.F_l, self.F_u = 0.1, 0.9 # Range for F

    def __call__(self, func):
        """
        Executes the optimization process.
        
        Args:
            func: The objective function to minimize.
            
        Returns:
            A tuple (best_x, best_y) representing the best solution found.
        """
        # Extract bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback for unexpected interface
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)

        evals = 0
        best_x = None
        best_y = float('inf')

        # Initialize population
        # X: population vectors, Y: fitness values
        # F: mutation scale, CR: crossover rate
        X = lb + (ub - lb) * np.random.rand(self.N, self.dim)
        Y = np.full(self.N, float('inf'))
        F = np.full(self.N, 0.5)
        CR = np.full(self.N, 0.9)

        # Evaluate initial population
        for i in range(self.N):
            if evals >= self.budget:
                break
            Y[i] = func(X[i])
            evals += 1
            if Y[i] < best_y:
                best_y = Y[i]
                best_x = np.copy(X[i])

        # Main Evolution Loop
        while evals < self.budget:
            for i in range(self.N):
                if evals >= self.budget:
                    break

                # 1. Parameter Adaptation (jDE)
                if np.random.rand() < self.tau1:
                    F[i] = self.F_l + np.random.rand() * self.F_u
                if np.random.rand() < self.tau2:
                    CR[i] = np.random.rand()

                # 2. Mutation (DE/rand/1)
                # Pick 3 distinct indices different from i
                indices = [idx for idx in range(self.N) if idx != i]
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                mutant = X[r1] + F[i] * (X[r2] - X[r3])
                
                # Boundary Handling: Clipping
                mutant = np.clip(mutant, lb, ub)

                # 3. Crossover (Binomial)
                cross_points = np.random.rand(self.dim) < CR[i]
                # Ensure at least one dimension is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial_x = np.where(cross_points, mutant, X[i])

                # 4. Selection
                trial_y = func(trial_x)
                evals += 1
                
                if trial_y <= Y[i]:
                    X[i] = trial_x
                    Y[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = np.copy(trial_x)

        return best_x, best_y
