# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adaptive Differential Evolution (DE) variant using a 'current-to-best/1' mutation strategy.
# Search state: A population of candidate solutions (vectors), their corresponding objective values, and the globally best solution found so far.
# Candidate generation: New candidates are generated using the current-to-best/1 mutation rule: v = x + F1 * (best - x) + F2 * (r1 - r2), followed by binomial crossover.
# Selection and replacement: A greedy selection mechanism compares the trial vector with the target vector; the one with the lower objective value survives to the next generation.
# Adaptation: Mutation factors (F) and crossover probability (Cr) are dithered per candidate to provide a range of search behaviors without explicit parameter tuning.
# Exploration mechanisms: Randomly selected population members (r1, r2) provide directional diversity. Dithering parameters allows for larger jumps.
# Exploitation mechanisms: The 'current-to-best' component explicitly pulls the population toward the current global optimum.
# Boundary handling: Candidates are clipped to the hypercube defined by the problem bounds before evaluation.
# Budget strategy: The algorithm maintains a strict evaluation counter and terminates immediately once the budget is exhausted, returning the best solution found.
# Closest known influences: JADE (Adaptive Differential Evolution), SHADE, and standard DE/current-to-best/1/bin.
# Novelty or unusual aspects: Extremely compact implementation designed for robustness across varying dimensions and budget constraints within a single class structure.
# Failure modes: May converge prematurely on highly rugged, multi-modal landscapes if the population size is too small or if the 'best' individual is in a deep local optimum.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the optimizer.
        :param budget: Total number of function evaluations allowed.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        
        # Heuristic for population size: balance between diversity and convergence speed.
        # Ensure pop_size is at least 4 for DE operations and doesn't consume the budget too fast.
        self.pop_size = max(6, min(20 + dim, budget // 4))

    def __call__(self, func):
        """
        Executes the optimization process.
        :param func: The objective function to minimize.
        :return: (best_x, best_y)
        """
        # Retrieve bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback if no bounds are provided, though GNBG benchmarks usually provide them.
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Initialization
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        scores = np.full(self.pop_size, np.inf)
        
        best_x = None
        best_y = np.inf
        eval_count = 0

        # Initial population evaluation
        for i in range(self.pop_size):
            if eval_count >= self.budget:
                break
            scores[i] = func(pop[i])
            eval_count += 1
            if scores[i] < best_y:
                best_y = scores[i]
                best_x = pop[i].copy()

        # Evolution loop
        while eval_count < self.budget:
            new_pop = np.copy(pop)
            
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                
                # Parameters dithering for adaptation
                f1 = 0.5 + 0.4 * np.random.rand()
                f2 = 0.5 + 0.4 * np.random.rand()
                cr = 0.7 + 0.3 * np.random.rand()
                
                # Mutation: current-to-best/1 strategy
                # Select two random distinct indices excluding i
                indices = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(indices, 2, replace=False)
                
                mutant = (pop[i] + 
                          f1 * (best_x - pop[i]) + 
                          f2 * (pop[r1] - pop[r2]))
                
                # Binomial Crossover
                cross_mask = np.random.rand(self.dim) < cr
                # Ensure at least one dimension is changed
                cross_mask[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_mask, mutant, pop[i])
                
                # Boundary handling: Clipping
                trial = np.clip(trial, lb, ub)
                
                # Selection
                trial_score = func(trial)
                eval_count += 1
                
                if trial_score <= scores[i]:
                    pop[i] = trial
                    scores[i] = trial_score
                    if trial_score < best_y:
                        best_y = trial_score
                        best_x = trial.copy()
                        
        return best_x, best_y
