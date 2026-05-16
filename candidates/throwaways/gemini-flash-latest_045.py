# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Self-Adaptive Differential Evolution (jDE) implementation tailored for robust black-box optimization.
# Search state: A population of N candidate solution vectors and their corresponding objective values, alongside individual-specific mutation (F) and crossover (Cr) parameters.
# Candidate generation: New candidates are produced via the DE/rand/1 mutation scheme (v = x_r1 + F * (x_r2 - x_r3)) followed by binomial crossover with the target individual.
# Selection and replacement: A greedy selection strategy is employed; the trial vector replaces the parent in the population only if its objective value is less than or equal to the parent's value.
# Adaptation: The mutation scale factor (F) and crossover probability (Cr) are self-adapted for each individual. With a probability (tau=0.1), these parameters are re-sampled, allowing successful configurations to persist and propagate.
# Exploration mechanisms: Differential mutation using three random distinct members ensures broad coverage of the search space, especially in the early stages or when the population is diverse.
# Exploitation mechanisms: As the population converges, the differential vectors become smaller, naturally focusing the search. Greedy replacement ensures the population moves toward local minima.
# Boundary handling: Candidates are constrained to the feasible region using clipping (clamping) to the lower and upper bounds provided by the objective function.
# Budget strategy: Population size is dynamically determined based on dimensions and budget, ensuring a balance between diversity and the number of generations. Evaluation is strictly tracked to never exceed the budget.
# Closest known influences: jDE (Brest et al., 2006).
# Novelty or unusual aspects: Minimalist implementation of jDE parameters designed to be self-contained and resilient to varying function landscapes without manual tuning.
# Failure modes: May converge prematurely on highly multi-modal landscapes if the population size is forced to be very small due to budget constraints, or may be slow on extremely high-dimensional problems.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the Differential Evolution algorithm.
        
        :param budget: Total number of allowed function evaluations.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        """
        Executes the optimization process.
        
        :param func: The objective function to minimize.
        :return: A tuple (best_x, best_y) representing the best solution found.
        """
        # Retrieve bounds from the function object
        if hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        elif hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        else:
            # Default bounds if none are provided
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Heuristic for population size: at least 4, at most 10*dim, but constrained by budget
        pop_size = max(4, min(10 * self.dim, self.budget // 4))
        
        # Initialize population
        pop = lb + np.random.rand(pop_size, self.dim) * (ub - lb)
        y = np.zeros(pop_size)
        
        # Initial parameters for jDE
        F = np.full(pop_size, 0.5)
        Cr = np.full(pop_size, 0.9)
        
        best_x = None
        best_y = float('inf')

        # Evaluate initial population
        for i in range(pop_size):
            if self.eval_count >= self.budget:
                break
            y[i] = func(pop[i])
            self.eval_count += 1
            if y[i] < best_y:
                best_y = y[i]
                best_x = np.copy(pop[i])

        # Evolution loop
        while self.eval_count < self.budget:
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break

                # Adaptation of F and Cr (jDE logic)
                tau = 0.1
                current_F = F[i]
                current_Cr = Cr[i]
                if np.random.rand() < tau:
                    current_F = 0.1 + np.random.rand() * 0.9
                if np.random.rand() < tau:
                    current_Cr = np.random.rand()

                # Mutation: DE/rand/1
                idxs = [idx for idx in range(pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                
                mutant = pop[r1] + current_F * (pop[r2] - pop[r3])
                
                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) <= current_Cr
                # Ensure at least one component is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary handling: Clipping
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                score = func(trial)
                self.eval_count += 1
                
                # Selection
                if score <= y[i]:
                    pop[i] = trial
                    y[i] = score
                    F[i] = current_F
                    Cr[i] = current_Cr
                    
                    if score < best_y:
                        best_y = score
                        best_x = np.copy(trial)
                        
        return best_x, best_y

# The harness will instantiate Algorithm and call it. 
# End of module.
