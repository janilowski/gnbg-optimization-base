# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adaptive Differential Evolution (jDE) algorithm. It maintains a population of candidate solutions and evolves them using mutation and crossover, with parameters (F, Cr) that evolve alongside the solutions.
# Search state: A population of vectors (shape: pop_size x dim), their corresponding fitness values, and per-individual mutation (F) and crossover (Cr) rates.
# Candidate generation: Uses the DE/rand/1 mutation strategy where a mutant vector is created from three random distinct population members, followed by binomial crossover with the target vector.
# Selection and replacement: A greedy selection strategy is used; a trial vector replaces its parent in the population if its fitness is better (lower) or equal.
# Adaptation: Control parameters F and Cr are self-adapted using the jDE logic: there is a small probability (0.1) in each step that a member's F or Cr is randomized, allowing successful parameters to persist through the population.
# Exploration mechanisms: Maintained by the differential mutation process and the random initialization of the population across the search space.
# Exploitation mechanisms: Greedy replacement and the tendency of the population to contract around the current global best as the search progresses.
# Boundary handling: Simple clipping of candidate vectors to the provided lower and upper bounds.
# Budget strategy: A strict counter is maintained for function evaluations. The algorithm terminates and returns the best found result immediately once the budget is exhausted.
# Closest known influences: The jDE algorithm by Brest et al. (2006) and the standard Differential Evolution by Storn and Price.
# Novelty or unusual aspects: A simplified, lightweight version of jDE designed for robustness in a single-class implementation with minimal overhead.
# Failure modes: May converge slowly on extremely high-dimensional landscapes or get trapped in local optima on highly multi-modal functions if the budget is very restricted relative to the dimension.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        
        # Adaptive population size based on dimension and budget
        self.pop_size = max(5, min(budget // 2, 10 + 2 * dim))
        if self.pop_size > 50:
            self.pop_size = 50

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback if bounds are missing
            lb = -5.0 * np.ones(self.dim)
            ub = 5.0 * np.ones(self.dim)

        # Initialize population
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        
        # Initialize control parameters (F and Cr) for jDE
        F = np.full(self.pop_size, 0.5)
        Cr = np.full(self.pop_size, 0.9)
        
        scores = np.zeros(self.pop_size)
        best_x = None
        best_y = float('inf')

        # Initial evaluation
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = func(pop[i])
            self.eval_count += 1
            scores[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Evolution loop
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # jDE Parameter Adaptation
                fi = F[i]
                cri = Cr[i]
                if np.random.rand() < 0.1:
                    fi = 0.1 + np.random.rand() * 0.9
                if np.random.rand() < 0.1:
                    cri = np.random.rand()
                
                # Mutation (DE/rand/1)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + fi * (b - c)
                
                # Crossover (Binomial)
                cross_points = np.random.rand(self.dim) < cri
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary Handling (Clipping)
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                y_trial = func(trial)
                self.eval_count += 1
                
                # Selection
                if y_trial <= scores[i]:
                    pop[i] = trial
                    scores[i] = y_trial
                    F[i] = fi
                    Cr[i] = cri
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()
                        
        return best_x, best_y
