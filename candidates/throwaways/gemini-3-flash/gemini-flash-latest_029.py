# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Self-Adaptive Differential Evolution (jDE) algorithm tailored for robust black-box minimization.
# Search state: A population of candidate vectors, their corresponding function values, and per-individual control parameters (F and CR).
# Candidate generation: Uses the DE/rand/1 mutation strategy combined with binomial crossover. Mutation generates a donor vector from three random distinct members.
# Selection and replacement: Simple greedy selection; a trial vector replaces its parent if its objective value is less than or equal to the parent's.
# Adaptation: Control parameters F (mutation scale) and CR (crossover probability) are adapted for each individual. With a probability (tau), these parameters are randomized, allowing the search to evolve effective settings.
# Exploration mechanisms: Differential mutation provides search directions based on the current distribution of the population. Global exploration is maintained by the stochastic nature of the mutation vectors.
# Exploitation mechanisms: Greedy selection ensures the population moves towards local minima. Binomial crossover preserves coordinates from the successful parent.
# Boundary handling: Trial vectors are clipped to the hypercube defined by the problem bounds before evaluation.
# Budget strategy: The algorithm tracks evaluations strictly, initializing the population first and then iterating until the evaluation counter hits the budget limit.
# Closest known influences: Based on the jDE algorithm by Brest et al. (2006), a well-regarded variant of Differential Evolution.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness across varying dimensions and budget constraints without external dependencies.
# Failure modes: May converge prematurely on highly multi-modal landscapes if the population size is too small or the budget is extremely restricted.
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
        # Population size: heuristic choice between 10*dim and 4, capped by budget.
        self.pop_size = max(4, min(10 * dim, budget // 2))
        
        # Hyperparameters for jDE adaptation
        self.tau1 = 0.1  # Probability to update F
        self.tau2 = 0.1  # Probability to update CR
        self.F_l, self.F_u = 0.1, 0.9
        
    def __call__(self, func):
        """
        Runs the minimization.
        :param func: The objective function.
        :return: (best_x, best_y)
        """
        # 1. Extract bounds
        if hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and func.bounds.lb is not None:
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        elif hasattr(func, 'lower') and func.lower is not None:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        else:
            # Fallback for unbounded problems (not typical in GNBG)
            lb = np.full(self.dim, -100.0)
            ub = np.full(self.dim, 100.0)

        # 2. Initialize population
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        pop_y = np.full(self.pop_size, np.inf)
        
        # Per-individual parameters
        F = np.full(self.pop_size, 0.5)
        CR = np.full(self.pop_size, 0.9)
        
        eval_count = 0
        best_x = None
        best_y = np.inf

        # 3. Initial Evaluation
        for i in range(self.pop_size):
            if eval_count >= self.budget:
                break
            pop_y[i] = func(pop[i])
            eval_count += 1
            if pop_y[i] < best_y:
                best_y = pop_y[i]
                best_x = pop[i].copy()

        # 4. Main Evolution Loop
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                
                # Parameter Adaptation (jDE logic)
                current_F = F[i]
                current_CR = CR[i]
                if np.random.rand() < self.tau1:
                    current_F = self.F_l + np.random.rand() * self.F_u
                if np.random.rand() < self.tau2:
                    current_CR = np.random.rand()
                
                # Mutation: DE/rand/1
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                mutant = pop[r1] + current_F * (pop[r2] - pop[r3])
                
                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) < current_CR
                # Ensure at least one dimension is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial_x = np.where(cross_points, mutant, pop[i])
                
                # Boundary handling: Clipping
                trial_x = np.clip(trial_x, lb, ub)
                
                # Evaluation
                trial_y = func(trial_x)
                eval_count += 1
                
                # Selection
                if trial_y <= pop_y[i]:
                    pop[i] = trial_x
                    pop_y[i] = trial_y
                    F[i] = current_F
                    CR[i] = current_CR
                    
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial_x.copy()
                        
        return best_x, best_y
