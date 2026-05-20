# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Success-Adaptive Differential Evolution (DE) algorithm utilizing a "DE/best/1/bin" strategy with dynamic parameter adaptation.
# Search state: A population of candidate vectors, their corresponding fitness values, and memory registers for the mutation scale (F) and crossover rate (CR).
# Candidate generation: New candidates are created by perturbing the current best individual with a scaled difference of two random members, followed by binomial crossover.
# Selection and replacement: A greedy survival mechanism where a child replaces its parent in the population only if its fitness is lower or equal.
# Adaptation: Control parameters F and CR are sampled from distributions whose means are updated using the Lehmer mean of parameters that successfully produced better offspring (SHADE-inspired).
# Exploration mechanisms: Stochastic selection of population members for mutation and a random initialization across the search space.
# Exploitation mechanisms: The "DE/best/1" strategy focuses search efforts around the best-found solution, and the adaptive CR favors local search when beneficial.
# Boundary handling: Candidates are clipped to the box constraints defined by the problem bounds.
# Budget strategy: The population size is automatically scaled based on the dimension and total budget. The evaluation loop strictly monitors the budget count to terminate exactly when exhausted.
# Closest known influences: SHADE (Success-history based Adaptive Differential Evolution) and JADE.
# Novelty or unusual aspects: Highly condensed adaptive logic designed for robustness in a single-file, zero-dependency environment.
# Failure modes: May struggle with extremely high-dimensional landscapes if the budget is very low, or potentially stall on highly rugged, discrete-like landscapes where gradient-like signals are absent.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Heuristic for population size: balance between exploration and generation count
        self.pop_size = max(5, min(budget // 20, 10 * dim, 50))
        # Adaptive parameter means
        self.mu_f = 0.5
        self.mu_cr = 0.5
        self.archive_f = []
        self.archive_cr = []

    def __call__(self, func):
        # Extract bounds from the function object
        if hasattr(func, 'lower'):
            lb, ub = np.asarray(func.lower), np.asarray(func.upper)
        else:
            lb, ub = np.asarray(func.bounds.lb), np.asarray(func.bounds.ub)

        evals = 0
        
        # Initialize population
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.full(self.pop_size, np.inf)
        
        best_x = None
        best_y = np.inf

        # Initial evaluation
        for i in range(self.pop_size):
            if evals >= self.budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()

        # Evolution loop
        while evals < self.budget:
            success_f = []
            success_cr = []
            
            # Identify indices for iteration
            indices = np.random.permutation(self.pop_size)
            
            for i in indices:
                if evals >= self.budget:
                    break
                
                # Parameter sampling
                f = np.clip(np.random.normal(self.mu_f, 0.1), 0.1, 1.0)
                cr = np.clip(np.random.normal(self.mu_cr, 0.1), 0.0, 1.0)
                
                # Mutation: DE/best/1
                # Find two random distinct individuals excluding the current parent
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                
                # Generate mutant
                mutant = best_x + f * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                
                # Binomial Crossover
                cross_points = np.random.rand(self.dim) < cr
                if not np.any(cross_points): # Ensure at least one component is changed
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                score = func(trial)
                evals += 1
                
                if score <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = score
                    success_f.append(f)
                    success_cr.append(cr)
                    if score < best_y:
                        best_y = score
                        best_x = trial.copy()
            
            # Update adaptation means if successes were found
            if success_f:
                # Lehmer mean for F to give more weight to larger successful steps
                self.mu_f = 0.8 * self.mu_f + 0.2 * (sum(s**2 for s in success_f) / sum(success_f))
                # Arithmetic mean for CR
                self.mu_cr = 0.8 * self.mu_cr + 0.2 * np.mean(success_cr)
                
            # Exit strategy for very small budgets
            if self.pop_size > self.budget - evals and evals < self.budget:
                # If remaining budget is too small for a full generation, just try a few more perturbations
                idx = np.random.randint(0, self.pop_size)
                while evals < self.budget:
                    noise = np.random.normal(0, 0.01 * (ub - lb))
                    candidate = np.clip(best_x + noise, lb, ub)
                    score = func(candidate)
                    evals += 1
                    if score < best_y:
                        best_y = score
                        best_x = candidate

        return best_x, best_y
