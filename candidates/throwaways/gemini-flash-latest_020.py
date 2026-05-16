# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) variant using a "best/1/bin" strategy with jittered parameters.
# Search state: A population of candidate solutions (vectors), their associated objective values, and the current global best.
# Candidate generation: New candidates are created using the current best individual as a base, adding a scaled difference between two random population members, followed by binomial crossover.
# Selection and replacement: Greedy selection is used; a child replaces its parent in the population only if its fitness is better or equal.
# Adaptation: The scaling factor (F) and crossover probability (Cr) are not fixed but jittered slightly around common defaults (0.5 and 0.9) to provide a balance of search behaviors without explicit tuning.
# Exploration mechanisms: Differential mutation provides a diverse set of directions derived from the current distribution of the population.
# Exploitation mechanisms: The "best/1/bin" strategy focuses search around the current known optimum, and binomial crossover allows for coordinate-aligned refinement.
# Boundary handling: Candidates are clipped to the hypercube defined by the lower and upper bounds.
# Budget strategy: The algorithm tracks evaluations and terminates immediately once the budget is reached, regardless of where it is in the generation cycle.
# Closest known influences: Classic Differential Evolution (Storn & Price), specifically the DE/best/1/bin variant.
# Novelty or unusual aspects: Extremely lightweight implementation optimized for robust performance across varying dimensions with zero external dependencies beyond NumPy.
# Failure modes: May converge prematurely on highly multi-modal landscapes if the population loses diversity too quickly, or struggle with extremely high-dimensional spaces where the budget is very small relative to the dimension.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the DE-based optimizer.
        :param budget: Total number of function evaluations allowed.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        # Heuristic for population size: larger for more dimensions, but capped.
        self.pop_size = min(max(10, 2 * dim), 50)

    def __call__(self, func):
        """
        Executes the optimization process.
        :param func: The objective function to minimize.
        :return: (best_x, best_y) the best found position and its value.
        """
        # Extract bounds from the function object
        if hasattr(func, 'lower') and func.lower is not None:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Default bounds if none provided
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Initialize population
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.zeros(self.pop_size)
        
        best_x = None
        best_y = float('inf')
        eval_count = 0

        # Initial evaluation of the population
        for i in range(self.pop_size):
            if eval_count >= self.budget:
                break
            val = func(pop[i])
            fitness[i] = val
            eval_count += 1
            if val < best_y:
                best_y = val
                best_x = np.copy(pop[i])

        # Main DE loop
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break

                # Parameter jittering for robustness
                f_scale = 0.4 + 0.4 * np.random.rand()  # F in [0.4, 0.8]
                cr_prob = 0.8 + 0.2 * np.random.rand()  # Cr in [0.8, 1.0]

                # Mutation: best/1 strategy
                # Pick two distinct indices different from the current best and current i
                indices = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(indices, 2, replace=False)
                
                # Mutant vector
                mutant = best_x + f_scale * (pop[r1] - pop[r2])
                
                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) < cr_prob
                # Ensure at least one dimension is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary handling: clipping
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                val = func(trial)
                eval_count += 1
                
                # Selection (greedy)
                if val <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_y:
                        best_y = val
                        best_x = np.copy(trial)
                        
        return best_x, best_y
