# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Self-Adaptive Differential Evolution (jDE) implementation designed for robust black-box optimization.
# Search state: A population of candidate solution vectors, their corresponding objective values, and individual control parameters (F and Cr) for each member.
# Candidate generation: Mutation is performed using the "rand/1" strategy, followed by binomial crossover to generate trial vectors.
# Selection and replacement: A greedy selection mechanism where a trial vector replaces its parent in the population if it yields a lower or equal objective value.
# Adaptation: Control parameters F (differential weight) and Cr (crossover probability) are self-adapted for each individual. With a probability of 0.1, these parameters are re-randomized, allowing the search to evolve its own hyper-parameters.
# Exploration mechanisms: Differential mutation provides a diverse direction for search, while the "rand/1" strategy prevents premature convergence compared to "best/1" strategies.
# Exploitation mechanisms: Greedy selection ensures that improvements are preserved, and the population naturally contracts around promising regions.
# Boundary handling: Trial vectors are clipped to the specified hypercube bounds (box constraints) before evaluation.
# Budget strategy: The algorithm tracks evaluations strictly, initializing the population first and then iterating through generations until the budget is exhausted.
# Closest known influences: The jDE algorithm by Brest et al. (2006).
# Novelty or unusual aspects: A streamlined, single-class implementation that dynamically detects function bound attributes and manages budget constraints without external dependencies beyond NumPy.
# Failure modes: Can struggle with extremely rugged landscapes if the population size is too small, or may converge slowly on very high-dimensional functions with limited budgets.
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
        self.eval_count = 0
        
        # Population size heuristic: between 10 and 100 based on dimension
        self.pop_size = max(10, min(dim * 10, 100))
        
        # Self-adaptation hyperparameters
        self.fl = 0.1  # Lower bound for F
        self.fu = 0.9  # Upper bound range for F
        self.tau1 = 0.1 # Probability to update F
        self.tau2 = 0.1 # Probability to update Cr

    def _get_bounds(self, func):
        """Extracts lower and upper bounds from the function object."""
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            return np.array(func.lower), np.array(func.upper)
        if hasattr(func, 'bounds'):
            return np.array(func.bounds.lb), np.array(func.bounds.ub)
        raise ValueError("Function object does not provide bounds in expected format.")

    def __call__(self, func):
        """
        Executes the optimization process.
        :param func: The objective function to minimize.
        :return: (best_x, best_y) tuple.
        """
        lb, ub = self._get_bounds(func)
        
        # Initialize population
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        # Individual F and Cr
        F = np.full(self.pop_size, 0.5)
        Cr = np.full(self.pop_size, 0.9)
        
        # Initial evaluations
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            if self.eval_count < self.budget:
                fitness[i] = func(pop[i])
                self.eval_count += 1
            else:
                # If budget is extremely small, fill remaining with infinity
                fitness[i] = float('inf')

        # Track global best
        best_idx = np.argmin(fitness)
        best_y = fitness[best_idx]
        best_x = pop[best_idx].copy()

        # Evolution loop
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # 1. Parameter Adaptation (jDE style)
                if np.random.rand() < self.tau1:
                    F[i] = self.fl + np.random.rand() * self.fu
                if np.random.rand() < self.tau2:
                    Cr[i] = np.random.rand()

                # 2. Mutation (rand/1)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + F[i] * (b - c)
                
                # 3. Crossover (Binomial)
                cross_points = np.random.rand(self.dim) <= Cr[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # 4. Boundary Handling (Clipping)
                trial = np.clip(trial, lb, ub)
                
                # 5. Selection
                val = func(trial)
                self.eval_count += 1
                
                if val <= fitness[i]:
                    fitness[i] = val
                    pop[i] = trial
                    if val < best_y:
                        best_y = val
                        best_x = trial.copy()

        return best_x, best_y
