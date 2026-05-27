# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A minimalist but robust implementation of Differential Evolution (DE/rand/1/bin).
# Search state: A population of candidate vectors and their corresponding objective values.
# Candidate generation: Mutation is performed by adding the scaled difference of two random members to a third (rand/1). Recombination uses binomial crossover (bin) to mix the mutant with the target vector.
# Selection and replacement: A simple greedy selection mechanism is used; the trial vector replaces the target vector only if its objective value is lower (minimization).
# Adaptation: Employs fixed scaling factor (F=0.8) and crossover probability (CR=0.9), which are standard defaults for general-purpose robustness.
# Exploration mechanisms: Random selection of three distinct population members for mutation ensures diversity.
# Exploitation mechanisms: Greedy selection retains the best traits in the population over generations.
# Boundary handling: Individual components are clipped to the hypercube defined by the problem bounds after mutation/crossover.
# Budget strategy: The algorithm tracks evaluations and terminates immediately when the budget is exhausted, returning the best solution found.
# Closest known influences: Storn and Price's original Differential Evolution algorithm.
# Novelty or unusual aspects: Highly condensed implementation designed for reliability across various dimensions and scales without external dependencies.
# Failure modes: May converge prematurely on highly rugged landscapes or exhibit slow convergence on extremely high-dimensional ridge-like functions compared to second-order methods.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Differential Evolution (DE/rand/1/bin) optimizer for black-box minimization.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            # Fallback if bounds are not strictly provided in standard slots
            lb = np.zeros(self.dim) - 100.0
            ub = np.zeros(self.dim) + 100.0

        # Parameter configuration
        # Small populations for small budgets, larger for high dimensions
        pop_size = int(max(10, min(self.dim * 10, 100, self.budget // 2)))
        F = 0.8  # Scaling factor
        CR = 0.9 # Crossover probability

        # State tracking
        best_x = None
        best_y = float('inf')

        def safe_eval(x):
            nonlocal best_x, best_y
            if self.eval_count >= self.budget:
                return None
            y = func(x)
            self.eval_count += 1
            if y < best_y:
                best_y = y
                best_x = np.copy(x)
            return y

        # Initialization
        population = lb + (ub - lb) * np.random.rand(pop_size, self.dim)
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            val = safe_eval(population[i])
            if val is None:
                return best_x, best_y
            fitness[i] = val

        # Main Evolution Loop
        while self.eval_count < self.budget:
            for i in range(pop_size):
                # Mutation: rand/1
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + F * (b - c)
                
                # Crossover: bin
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Boundary handling: clip
                trial = np.clip(trial, lb, ub)

                # Selection
                trial_y = safe_eval(trial)
                if trial_y is None:
                    return best_x, best_y
                
                if trial_y <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_y

        return best_x, best_y
