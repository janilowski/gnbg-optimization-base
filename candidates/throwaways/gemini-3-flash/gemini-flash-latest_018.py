# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adaptive Differential Evolution (jDE variant) implementation designed for robust black-box minimization.
# Search state: A population of candidate solutions, their associated objective values, and individual-specific control parameters (F and CR).
# Candidate generation: Mutation uses the DE/rand/1 strategy (selecting three distinct random vectors from the population) followed by binomial crossover.
# Selection and replacement: Greedy survival (one-to-one selection). A candidate replaces its parent in the population if its objective value is lower or equal.
# Adaptation: Control parameters F (mutation scale) and CR (crossover probability) are updated for each individual with a fixed probability (tau) to maintain diversity in search behavior.
# Exploration mechanisms: Random selection of mutation vectors and the stochastic nature of F and CR updates.
# Exploitation mechanisms: Greedy selection and binomial crossover which incorporates components from the current target vector.
# Boundary handling: Candidate vectors are clipped to the hyper-rectangle defined by the lower and upper bounds of the search space.
# Budget strategy: Evaluations are tracked strictly. The search loop terminates immediately when the evaluation limit is reached.
# Closest known influences: Brest et al. (2006) - "Self-Adapting Control Parameters in Differential Evolution".
# Novelty or unusual aspects: Extremely compact implementation of self-adaptive logic within a single class structure, handling multiple bound-attribute conventions.
# Failure modes: Like most DE variants, it may stagnate on highly non-separable or extremely rugged landscapes if the population loses diversity too early relative to the budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the Differential Evolution optimizer.
        :param budget: Total number of function evaluations allowed.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        # Heuristic for population size: generally between 5*dim and 10*dim, capped for efficiency.
        self.pop_size = min(max(10, 5 * dim), 100)
        
        # Hyperparameters for jDE self-adaptation
        self.tau1 = 0.1  # Probability to update F
        self.tau2 = 0.1  # Probability to update CR
        self.F_low, self.F_up = 0.1, 0.9
        
    def __call__(self, func):
        """
        Executes the optimization process.
        :param func: The objective function to minimize.
        :return: (best_x, best_y) found during the search.
        """
        # Boundary extraction
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback for undefined bounds (though benchmark usually provides them)
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Optimization State
        eval_count = 0
        best_x = None
        best_y = float('inf')

        def evaluate(x):
            nonlocal eval_count, best_x, best_y
            if eval_count >= self.budget:
                return None
            res = func(x)
            eval_count += 1
            if res < best_y:
                best_y = res
                best_x = x.copy()
            return res

        # Population Initialization
        # Shape: (pop_size, dim)
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.zeros(self.pop_size)
        
        # Initialize individual control parameters
        F = np.full(self.pop_size, 0.5)
        CR = np.full(self.pop_size, 0.9)

        for i in range(self.pop_size):
            val = evaluate(pop[i])
            if val is None: break
            fitness[i] = val

        # Main Evolution Loop
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break

                # Self-adaptation of parameters F and CR
                curr_F = F[i]
                curr_CR = CR[i]
                if np.random.rand() < self.tau1:
                    curr_F = self.F_low + np.random.rand() * self.F_up
                if np.random.rand() < self.tau2:
                    curr_CR = np.random.rand()

                # Mutation: DE/rand/1
                indices = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                mutant = pop[r1] + curr_F * (pop[r2] - pop[r3])
                
                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) <= curr_CR
                # Ensure at least one dimension is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary Handling: Clipping
                trial = np.clip(trial, lb, ub)
                
                # Selection
                trial_fitness = evaluate(trial)
                if trial_fitness is not None:
                    if trial_fitness <= fitness[i]:
                        pop[i] = trial
                        fitness[i] = trial_fitness
                        F[i] = curr_F
                        CR[i] = curr_CR

        return best_x, best_y
