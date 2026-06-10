# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust Differential Evolution (DE) variant using a 'current-to-best/1' mutation strategy and crossover, adapted for black-box optimization within a fixed budget.
# Search state: A population of candidate solutions (vectors) and their corresponding objective values, alongside the tracking of the best individual found so far.
# Candidate generation: New candidates are generated using the formula: v = x + F1 * (best - x) + F2 * (r1 - r2). This combines the current point, a vector toward the best point, and a random differential vector. Binomial crossover is then applied.
# Selection and replacement: A greedy selection mechanism is used where the child replaces the parent only if its objective value is lower (minimization).
# Adaptation: The mutation scale factors (F1, F2) and crossover probability (Cr) are dithered (randomized slightly per iteration) to provide a simple form of adaptation without complex history tracking.
# Exploration mechanisms: Differential vectors (r1 - r2) provide stochastic exploration based on the current diversity of the population.
# Exploitation mechanisms: The 'current-to-best' component greedily pulls the population toward the best known region.
# Boundary handling: Candidates are clipped to the hypercube defined by the problem bounds.
# Budget strategy: The algorithm tracks the number of calls to the objective function, terminating immediately when the budget is exhausted and returning the best result found.
# Closest known influences: Standard Differential Evolution (Storn & Price), JADE/SHADE influences in mutation structure.
# Novelty or unusual aspects: Minimalist implementation designed for robustness across varying dimensions and budget sizes, scaling population size dynamically based on available budget and dimensionality.
# Failure modes: May struggle with highly deceptive landscapes or extremely high-dimensional spaces where the budget is insufficient to develop a meaningful population distribution.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the optimization algorithm.
        :param budget: Total number of function evaluations allowed.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        
        # Heuristic for population size: balance between diversity and convergence speed.
        self.pop_size = max(5, min(dim * 10, budget // 20))
        
    def __call__(self, func):
        """
        Executes the optimization process.
        :param func: The objective function to minimize.
        :return: A tuple (best_x, best_y) representing the best solution found.
        """
        # 1. Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback to defaults if bounds are not provided as expected
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # 2. Initialize Population
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.full(self.pop_size, np.inf)
        
        best_x = None
        best_y = np.inf

        # Initial evaluation
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            val = func(pop[i])
            self.eval_count += 1
            fitness[i] = val
            if val < best_y:
                best_y = val
                best_x = np.copy(pop[i])

        # 3. Evolution Loop
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Parameters: randomized for basic adaptation (dithering)
                f1 = 0.5 + 0.2 * np.random.rand()
                f2 = 0.5 + 0.2 * np.random.rand()
                cr = 0.7 + 0.2 * np.random.rand()

                # Pick random indices for mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(idxs, 2, replace=False)

                # Mutation: current-to-best/1 variant
                # mutant = x_i + F1 * (best - x_i) + F2 * (r1 - r2)
                mutant = pop[i] + f1 * (best_x - pop[i]) + f2 * (pop[r1] - pop[r2])
                
                # Crossover (Binomial)
                cross_points = np.random.rand(self.dim) < cr
                # Ensure at least one dimension is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary Handling: Clipping
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                val = func(trial)
                self.eval_count += 1
                
                # Selection
                if val <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_y:
                        best_y = val
                        best_x = np.copy(trial)
            
            # If population has converged or budget is tight, check termination
            if self.eval_count >= self.budget:
                break
                
        return best_x, best_y

# The harness expects the Algorithm class to be available at the top level.
# No top-level execution code is included.
