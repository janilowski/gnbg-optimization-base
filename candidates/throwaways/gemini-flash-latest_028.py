# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Success-History based Adaptive Differential Evolution (SHADE) variant.
# Search state: A population of candidate vectors, their corresponding objective values, and a historical memory (H=6) of successful mutation (F) and crossover (CR) parameters.
# Candidate generation: Employs the 'DE/current-to-pbest/1' mutation strategy followed by binomial crossover. This strategy balances global exploration with local refinement by biasing mutation towards one of the top-performing individuals.
# Selection and replacement: Greedy selection where an offspring replaces its parent in the population if its objective value is better than or equal to the parent's.
# Adaptation: Parameters F and CR are sampled per individual from Cauchy and Normal distributions. The means of these distributions are updated using a success-history memory, where successful parameters are archived and used to calculate new means via weighted Lehmer and arithmetic averaging.
# Exploration mechanisms: Maintained through the differential mutation logic and the diverse initial population sampled uniformly within the bounds.
# Exploitation mechanisms: Driven by the 'p-best' mutation component and the greedy selection process.
# Boundary handling: Simple clipping to the provided lower and upper bounds.
# Budget strategy: Continuous monitoring of evaluations; the search loop terminates as soon as the evaluation count reaches the budget.
# Closest known influences: SHADE (Tanabe & Fukunaga, 2013) and JADE (Zhang & Sanderson, 2009).
# Novelty or unusual aspects: A highly condensed implementation of the SHADE architecture focused on robustness and self-adaptation without external dependencies.
# Failure modes: May converge prematurely on highly multimodal landscapes if the budget is too small to allow the population to escape local optima; the clipping boundary handling can also lead to stagnation on the search space edges.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        try:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        except AttributeError:
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        
        # Ensure lb and ub are arrays of the correct dimension
        if lb.ndim == 0: lb = np.full(self.dim, lb)
        if ub.ndim == 0: ub = np.full(self.dim, ub)

        # Internal state and hyperparameters
        eval_count = 0
        pop_size = min(100, max(20, 10 * self.dim))
        H = 6
        p_best_rate = 0.1
        memory_f = np.full(H, 0.5)
        memory_cr = np.full(H, 0.5)
        memory_idx = 0
        
        best_x = None
        best_y = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_y
            if eval_count >= self.budget:
                return best_y
            res = func(x)
            eval_count += 1
            if res < best_y:
                best_y = res
                best_x = x.copy()
            return res

        # Initialization
        pop = lb + np.random.rand(pop_size, self.dim) * (ub - lb)
        scores = np.zeros(pop_size)
        for i in range(pop_size):
            if eval_count < self.budget:
                scores[i] = evaluate(pop[i])
            else:
                scores[i] = np.inf

        # Main Loop
        while eval_count < self.budget:
            sorted_idx = np.argsort(scores)
            success_f = []
            success_cr = []
            success_df = []
            
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                
                # Sample parameters
                idx_h = np.random.randint(0, H)
                cr = np.clip(np.random.normal(memory_cr[idx_h], 0.1), 0, 1)
                f = -1.0
                while f <= 0:
                    f = memory_f[idx_h] + 0.1 * np.random.standard_cauchy()
                f = min(f, 1.0)
                
                # Mutation: current-to-pbest/1
                p_best_idx = sorted_idx[:max(2, int(pop_size * p_best_rate))]
                x_pbest = pop[np.random.choice(p_best_idx)]
                
                r1, r2 = -1, -1
                while r1 == i or r1 == -1: r1 = np.random.randint(0, pop_size)
                while r2 == i or r2 == r1 or r2 == -1: r2 = np.random.randint(0, pop_size)
                
                v = pop[i] + f * (x_pbest - pop[i]) + f * (pop[r1] - pop[r2])
                
                # Binomial Crossover
                mask = np.random.rand(self.dim) < cr
                mask[np.random.randint(0, self.dim)] = True
                trial = np.where(mask, v, pop[i])
                
                # Boundary Handling
                trial = np.clip(trial, lb, ub)
                
                # Evaluation and Selection
                score_trial = evaluate(trial)
                if score_trial <= scores[i]:
                    success_df.append(abs(scores[i] - score_trial))
                    success_f.append(f)
                    success_cr.append(cr)
                    scores[i] = score_trial
                    pop[i] = trial

            # Update success history memory
            if success_f:
                success_f = np.array(success_f)
                success_cr = np.array(success_cr)
                success_df = np.array(success_df)
                weights = success_df / (np.sum(success_df) + 1e-9)
                
                # Weighted arithmetic mean for CR
                memory_cr[memory_idx] = np.sum(weights * success_cr)
                # Weighted Lehmer mean for F
                memory_f[memory_idx] = np.sum(weights * (success_f**2)) / (np.sum(weights * success_f) + 1e-9)
                
                memory_idx = (memory_idx + 1) % H

        return best_x, best_y
