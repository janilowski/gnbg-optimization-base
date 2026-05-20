# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adaptive Differential Evolution (jDE) variant. It employs a population of candidate solutions that evolve through mutation, crossover, and selection, with internal parameters F and CR adapting over time for each individual.
# Search state: A population of vectors, their corresponding objective values, and associated control parameters (F, CR) for each individual.
# Candidate generation: Uses the DE/rand/1/bin strategy. For each individual, three distinct others are chosen to create a mutant vector via scaled difference. A trial vector is then formed via binomial crossover with the parent.
# Selection and replacement: Greedy selection. A trial vector replaces its parent in the next generation if and only if its objective value is less than or equal to the parent's.
# Adaptation: Control parameters F (scale factor) and CR (crossover rate) are adapted per individual. With a small probability (tau=0.1), these parameters are re-randomized, allowing successful parameter settings to persist and propagate.
# Exploration mechanisms: Differential mutation provides global search capabilities, especially in early stages or with high F values. Random initialization covers the search space.
# Exploitation mechanisms: Greedy selection ensures the population moves towards local minima. As the population converges, the differential vectors become smaller, naturally narrowing the search.
# Boundary handling: Trial vectors are clipped to the hypercube defined by the problem bounds.
# Budget strategy: The algorithm tracks the number of function evaluations and terminates immediately once the budget is exhausted.
# Closest known influences: jDE (Brest et al., 2006) and standard Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: Minimalist implementation of self-adaptation within a single class, specifically designed for robust performance across varying dimensions without manual tuning.
# Failure modes: Like most DE variants, it may struggle with highly rugose landscapes with very narrow global optima or extremely high-dimensional spaces where the population size becomes a bottleneck for the budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    A Self-Adaptive Differential Evolution (jDE) implementation for black-box minimization.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Heuristic for population size: at least 4 for DE, scaled with dimension
        self.pop_size = max(10, min(10 * dim, 100))
        self.eval_count = 0

    def __call__(self, func):
        # Determine search bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = func.lower, func.upper
        elif hasattr(func, 'bounds'):
            lb, ub = func.bounds.lb, func.bounds.ub
        else:
            # Fallback if bounds are not provided as expected
            lb = np.full(self.dim, -100.0)
            ub = np.full(self.dim, 100.0)

        # Initialize population and parameters
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        
        # Self-adaptive parameters: F (mutation scale) and CR (crossover rate)
        F = np.full(self.pop_size, 0.5)
        CR = np.full(self.pop_size, 0.9)
        
        # Initial evaluations
        fitness = np.zeros(self.pop_size)
        best_y = float('inf')
        best_x = None

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            fitness[i] = func(pop[i])
            self.eval_count += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()

        # Evolution loop
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Adaptation of F and CR (jDE logic)
                tau1 = tau2 = 0.1
                F_low, F_up = 0.1, 0.9
                curr_F = F[i]
                curr_CR = CR[i]
                
                if np.random.rand() < tau1:
                    curr_F = F_low + np.random.rand() * F_up
                if np.random.rand() < tau2:
                    curr_CR = np.random.rand()

                # Mutation: DE/rand/1
                indices = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                mutant = pop[r1] + curr_F * (pop[r2] - pop[r3])
                
                # Boundary handling
                mutant = np.clip(mutant, lb, ub)
                
                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) < curr_CR
                # Ensure at least one component is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                trial_fitness = func(trial)
                self.eval_count += 1
                
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    F[i] = curr_F
                    CR[i] = curr_CR
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
