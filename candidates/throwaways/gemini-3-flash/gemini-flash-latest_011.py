# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Adaptive Differential Evolution (DE) implementation using a current-to-best mutation strategy and success-based parameter adaptation.
# Search state: A population of candidate solutions, their corresponding fitness values, and two scalars representing the moving-average means of the adaptive crossover (CR) and mutation (F) parameters.
# Candidate generation: Mutation follows a "current-to-best/1" scheme: v = x + F * (best - x) + F * (r1 - r2). Binomial crossover is applied to the mutant to produce a trial vector.
# Selection and replacement: A greedy selection mechanism is used where a trial vector replaces its parent in the population only if its objective value is less than or equal to the parent's value.
# Adaptation: The mutation factor (F) and crossover probability (CR) are sampled for each individual from Cauchy and Normal distributions, respectively. The parameters of these distributions (location/mean) are updated based on the successful parameters from the previous generation using a simplified JADE adaptation logic.
# Exploration mechanisms: Random vector selection (r1, r2) and the stochastic nature of the adaptation distributions provide exploration.
# Exploitation mechanisms: The "current-to-best" mutation vector directs the search towards the most promising region found so far, while greedy selection preserves improvements.
# Boundary handling: Trial vectors are clipped to the hyper-rectangle defined by the problem's lower and upper bounds.
# Budget strategy: The algorithm tracks evaluations and stops immediately once the budget is exhausted. The population size is heuristically scaled according to the budget and dimension.
# Closest known influences: JADE (Adaptive Differential Evolution with Optional External Archive) by Zhang & Sanderson.
# Novelty or unusual aspects: Simplified success-history adaptation logic designed to be extremely lightweight and robust across varying dimensions and budget constraints.
# Failure modes: On extremely small budgets relative to the dimension, the population may not have enough iterations to converge. In highly multimodal or deceptive landscapes, the adaptation might reduce diversity too quickly.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Heuristic for population size: balance between diversity and iterations
        self.pop_size = max(min(budget // 10, 10 * dim), 5)
        # Ensure pop_size is sensible
        if self.pop_size > budget:
            self.pop_size = budget
        self.mu_f = 0.5
        self.mu_cr = 0.5

    def __call__(self, func):
        # Extract bounds from func
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        else:
            lb, ub = np.full(self.dim, -5.0), np.full(self.dim, 5.0)

        # Initialize population
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.full(self.pop_size, np.inf)
        best_y = np.inf
        best_x = pop[0].copy()
        
        evals = 0
        for i in range(self.pop_size):
            if evals < self.budget:
                fitness[i] = func(pop[i])
                evals += 1
                if fitness[i] < best_y:
                    best_y = fitness[i]
                    best_x = pop[i].copy()
            else:
                break

        # Main Evolution Loop
        while evals < self.budget:
            success_f = []
            success_cr = []
            
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample parameters for this candidate
                # Mutation factor F: Cauchy distribution (truncated)
                f = np.random.standard_cauchy() * 0.1 + self.mu_f
                while f <= 0:
                    f = np.random.standard_cauchy() * 0.1 + self.mu_f
                if f > 1.0: f = 1.0
                
                # Crossover rate CR: Normal distribution (clipped)
                cr = np.clip(np.random.normal(self.mu_cr, 0.1), 0, 1)
                
                # Candidate generation
                if self.pop_size >= 4:
                    # current-to-best/1 mutation
                    choices = [idx for idx in range(self.pop_size) if idx != i]
                    r1, r2 = np.random.choice(choices, 2, replace=False)
                    mutant = pop[i] + f * (best_x - pop[i]) + f * (pop[r1] - pop[r2])
                else:
                    # Fallback mutation for tiny populations
                    mutant = best_x + f * (lb + np.random.rand(self.dim)*(ub - lb) - pop[i])
                
                # Binomial crossover
                cross_mask = np.random.rand(self.dim) < cr
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                f_trial = func(trial)
                evals += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    success_f.append(f)
                    success_cr.append(cr)
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()
            
            # Adaptation of mu_f and mu_cr
            if success_f:
                c = 0.1
                self.mu_cr = (1 - c) * self.mu_cr + c * np.mean(success_cr)
                # Lehmer mean for mutation factor f
                self.mu_f = (1 - c) * self.mu_f + c * (sum(s**2 for s in success_f) / sum(success_f))

        return best_x, best_y
