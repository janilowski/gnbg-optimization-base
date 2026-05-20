# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust Differential Evolution (DE) variant using a 'best/1/bin' strategy with stochastic parameter variation.
# Search state: A population of vectors representing candidate solutions and their associated fitness values, plus a reference to the best solution found.
# Candidate generation: For each member of the population, a mutation vector is generated using the "best/1" strategy (current_best + F * (r1 - r2)). A trial vector is then created via binomial crossover with the parent.
# Selection and replacement: Greedy selection is employed; the trial vector replaces the parent in the population if its objective value is lower than or equal to the parent's.
# Adaptation: Mutation scale (F) and crossover probability (Cr) are randomly sampled for each trial generation from predetermined ranges (F in [0.4, 1.0], Cr in [0.1, 0.9]) to ensure diverse search behavior without needing fixed hyperparameter tuning.
# Exploration mechanisms: The population-based nature of DE combined with random differential mutation ensures broad coverage of the search space.
# Exploitation mechanisms: The "best/1" mutation strategy focuses the search around the current global minimum, providing strong local refinement.
# Boundary handling: Trial vectors are clipped to the hypercube defined by the problem bounds using np.clip.
# Budget strategy: The population size is scaled based on dimensionality. The search continues in a generational loop, strictly monitoring the evaluation count to terminate exactly when the budget is exhausted.
# Closest known influences: Classical Differential Evolution (Storn & Price, 1997) and self-adaptive variants like jDE (Brest et al., 2006).
# Novelty or unusual aspects: A streamlined, vectorized implementation that balances exploration and exploitation efficiently within a single Python class.
# Failure modes: May converge prematurely on highly multi-modal functions if the population size is too small, or struggle on extremely high-dimensional problems where the budget is insufficient to populate the space.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    An implementation of a Differential Evolution (DE) optimizer designed for 
    black-box minimization within a strict evaluation budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        
        # Determine population size: scale with dimension but keep within practical limits.
        # Ensure we have at least 4 members for mutation r1, r2, r3 logic.
        self.pop_size = max(10, min(dim * 10, 50))
        if self.pop_size > budget:
            self.pop_size = budget

    def __call__(self, func):
        # Determine search bounds from the provided function object.
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        else:
            # Fallback bounds if none are provided.
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Initialize population within bounds.
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.full(self.pop_size, np.inf)

        best_idx = 0
        best_x = None
        best_y = np.inf

        # Initial evaluation of the population.
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            
            fitness[i] = func(pop[i])
            self.eval_count += 1
            
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()
                best_idx = i

        # Main evolution loop.
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Parameter sampling for stochastic adaptation.
                f_scale = np.random.uniform(0.4, 1.0)
                cr_prob = np.random.uniform(0.1, 0.9)

                # Selection of random indices for mutation (excluding current parent).
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)

                # Mutation: best/1 strategy.
                # trial_mutant = best + F * (r1 - r2)
                mutant = best_x + f_scale * (pop[r1] - pop[r2])

                # Binomial Crossover.
                cross_points = np.random.rand(self.dim) < cr_prob
                # Ensure at least one dimension is inherited from the mutant.
                cross_points[np.random.randint(0, self.dim)] = True
                
                trial_x = np.where(cross_points, mutant, pop[i])

                # Boundary Handling: Clip to feasible space.
                trial_x = np.clip(trial_x, lb, ub)

                # Evaluation.
                trial_y = func(trial_x)
                self.eval_count += 1

                # Selection: Replacement if trial is better.
                if trial_y <= fitness[i]:
                    pop[i] = trial_x
                    fitness[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial_x.copy()
                        best_idx = i

        return best_x, best_y
