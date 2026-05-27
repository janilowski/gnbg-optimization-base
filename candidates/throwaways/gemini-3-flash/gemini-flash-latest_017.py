# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Self-Adaptive Differential Evolution (jDE) implementation. It uses per-individual mutation and crossover parameters that evolve alongside the solutions.
# Search state: A population of candidate vectors, their objective values, and associated control parameters (F and Cr) for each individual.
# Candidate generation: Mutation is performed using the DE/rand/1 strategy (base vector + scaled difference of two random members). Binomial crossover is then applied to combine the mutant with the target vector.
# Selection and replacement: A simple greedy selection mechanism where the trial vector replaces the parent in the population if its objective value is less than or equal to the parent's.
# Adaptation: Employs the jDE mechanism: with a small probability (tau1, tau2), the mutation scale (F) and crossover rate (Cr) are randomly reset. Successful parameters are implicitly preserved through the survival of the individual.
# Exploration mechanisms: Initial uniform sampling across the search space and the stochastic nature of the mutation/crossover logic, particularly when F is large.
# Exploitation mechanisms: Greedy selection and the convergence of the population towards the global minimum as the difference vectors shrink.
# Boundary handling: Trial vectors are clipped to the hyper-rectangle defined by the lower and upper bounds.
# Budget strategy: Evaluations are consumed sequentially. The search continues until the total number of function calls reaches the specified budget. Population size is dynamically scaled based on dimension and budget.
# Closest known influences: Differential Evolution (Storn & Price), jDE (Brest et al.).
# Novelty or unusual aspects: Compact implementation of self-adaptation in a single class without external dependencies beyond NumPy.
# Failure modes: Can struggle with extremely high-dimensional landscapes or highly deceptive functions where the budget is too small to form a representative population.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        
        # Hyperparameters for jDE
        self.tau1 = 0.1  # Probability to update F
        self.tau2 = 0.1  # Probability to update Cr
        self.F_low, self.F_high = 0.1, 0.9
        
        # Adaptive population size
        self.pop_size = max(min(dim * 10, 100), 10)
        if self.pop_size > self.budget:
            self.pop_size = max(5, self.budget // 2)

    def __call__(self, func):
        # Extract bounds
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

        # Initialize population
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        
        # Initialize control parameters: F (mutation) and Cr (crossover)
        F = np.full(self.pop_size, 0.5)
        Cr = np.full(self.pop_size, 0.9)
        
        # Evaluate initial population
        fitness = np.zeros(self.pop_size)
        best_y = float('inf')
        best_x = None

        for i in range(self.pop_size):
            if self.eval_count < self.budget:
                fitness[i] = func(pop[i])
                self.eval_count += 1
                if fitness[i] < best_y:
                    best_y = fitness[i]
                    best_x = pop[i].copy()
            else:
                fitness[i] = float('inf')

        # Main Evolution Loop
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # 1. Parameter Adaptation (jDE)
                current_F = F[i]
                current_Cr = Cr[i]
                if np.random.rand() < self.tau1:
                    current_F = self.F_low + np.random.rand() * self.F_high
                if np.random.rand() < self.tau2:
                    current_Cr = np.random.rand()
                
                # 2. Mutation (DE/rand/1)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + current_F * (b - c)
                
                # 3. Crossover (Binomial)
                cross_points = np.random.rand(self.dim) < current_Cr
                # Ensure at least one component is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # 4. Boundary Handling (Clipping)
                trial = np.clip(trial, lb, ub)
                
                # 5. Selection
                f_trial = func(trial)
                self.eval_count += 1
                
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    F[i] = current_F
                    Cr[i] = current_Cr
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_y
