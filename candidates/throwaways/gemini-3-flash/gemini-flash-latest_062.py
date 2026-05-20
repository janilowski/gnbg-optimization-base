# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust Differential Evolution (DE) implementation using the "rand/1/bin" strategy with randomized scale and crossover parameters.
# Search state: A population of candidate vectors stored as a 2D numpy array, along with their corresponding objective function values.
# Candidate generation: Trial vectors are produced using differential mutation (combining three distinct random members) followed by binomial crossover with the target vector.
# Selection and replacement: A greedy selection mechanism is used where a trial vector replaces its parent in the population only if its objective value is lower or equal.
# Adaptation: To handle diverse landscapes without manual tuning, the mutation scale (F) and crossover probability (Cr) are randomly sampled from fixed ranges ([0.4, 0.9] and [0.5, 1.0] respectively) for each individual trial.
# Exploration mechanisms: Exploration is driven by the diversity of the initial population and the stochastic nature of the "rand/1" mutation, which samples differences between random members.
# Exploitation mechanisms: Exploitation occurs through the greedy selection process, which gradually collapses the population towards regions of lower objective values.
# Boundary handling: Trial vectors are clipped to the provided lower and upper bounds using numpy.clip.
# Budget strategy: The algorithm tracks evaluations and terminates the search process immediately before exceeding the user-defined budget.
# Closest known influences: Storn and Price's original Differential Evolution; basic JADE-style parameter sampling concepts.
# Novelty or unusual aspects: Employs a simplified parameter jittering approach to maintain robustness across various dimensions and function types without complex history-based adaptation.
# Failure modes: May converge slowly on highly non-separable or extremely high-dimensional problems; can get stuck in local optima if the population diversity is lost too early.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the Differential Evolution algorithm.
        
        Args:
            budget: Total number of allowed function evaluations.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        """
        Executes the minimization process.
        
        Args:
            func: The objective function to minimize.
        
        Returns:
            (best_x, best_y): The best solution found.
        """
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback if no bounds are provided (though benchmark usually provides them)
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Population size heuristic: 10 * dim, but capped for efficiency and budget
        pop_size = min(100, max(10, 10 * self.dim))
        
        # Initialize population
        pop = lb + np.random.rand(pop_size, self.dim) * (ub - lb)
        scores = np.full(pop_size, np.inf)
        
        best_x = None
        best_y = np.inf

        # Initial evaluation
        for i in range(pop_size):
            if self.eval_count >= self.budget:
                break
            scores[i] = func(pop[i])
            self.eval_count += 1
            if scores[i] < best_y:
                best_y = scores[i]
                best_x = np.copy(pop[i])

        # Main DE loop
        while self.eval_count < self.budget:
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Selection of three distinct individuals for mutation
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Randomized parameters for robustness
                f = np.random.uniform(0.4, 0.9)
                cr = np.random.uniform(0.5, 1.0)
                
                # Mutation
                mutant = a + f * (b - c)
                
                # Binomial Crossover
                cross_points = np.random.rand(self.dim) < cr
                # Ensure at least one dimension is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary handling
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                score_trial = func(trial)
                self.eval_count += 1
                
                # Greedy Selection
                if score_trial <= scores[i]:
                    scores[i] = score_trial
                    pop[i] = trial
                    if score_trial < best_y:
                        best_y = score_trial
                        best_x = np.copy(trial)
                        
        return best_x, best_y

# The harness expects the class 'Algorithm' to be present in the module.
# No top-level execution code is included.
