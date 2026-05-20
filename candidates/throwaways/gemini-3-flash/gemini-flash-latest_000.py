# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust and compact Differential Evolution (DE) variant using the DE/rand/1/bin strategy.
# Search state: A population of candidate solution vectors and their associated fitness values.
# Candidate generation: New candidates are generated using differential mutation (adding a scaled difference between two random members to a third) followed by binomial crossover.
# Selection and replacement: A simple greedy selection process where a trial vector replaces its parent if it yields a lower or equal objective value.
# Adaptation: Employs a dithered scale factor (F) to maintain diversity and a high crossover rate (CR) to exploit parameter correlations.
# Exploration mechanisms: Stochastic initialization across the search space and differential mutation which scales with population diversity.
# Exploitation mechanisms: Greedy selection ensures the population strictly improves over time, and crossover preserves dimensions from successful parents.
# Boundary handling: Trial vectors are clipped to the provided lower and upper bounds before evaluation.
# Budget strategy: The algorithm initializes the population and then iteratively refines it until the evaluation budget is exhausted.
# Closest known influences: Standard Differential Evolution (Storn & Price).
# Novelty or unusual aspects: Dynamic population sizing relative to the budget and dimensionality to ensure the strategy remains viable under tight constraints.
# Failure modes: May converge prematurely on highly multimodal or discontinuous landscapes if the budget prevents a large enough population.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the Differential Evolution algorithm.
        
        Args:
            budget (int): Maximum number of function evaluations allowed.
            dim (int): Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        
        # Determine a reasonable population size based on dimensionality and budget.
        # DE typically requires at least 4 members for the rand/1 mutation strategy.
        self.pop_size = int(min(budget, max(6, 2 * dim)))
        if self.pop_size > 50:
            self.pop_size = 50

    def __call__(self, func):
        """
        Executes the minimization search.
        
        Args:
            func (callable): The objective function to minimize.
        
        Returns:
            tuple: (best_x, best_y) representing the best solution found.
        """
        # Determine bounds from provided function attributes
        try:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        except AttributeError:
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)

        # Initialize population
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        fitness = np.full(self.pop_size, np.inf)

        # Initial evaluation
        best_y = np.inf
        best_x = None

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            fitness[i] = func(pop[i])
            self.eval_count += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()

        # Evolutionary loop
        # Hyperparameters for DE/rand/1/bin
        cr = 0.9  # Crossover probability

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # 1. Mutation: Select 3 random indices distinct from i
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                
                # Use dithering for the scale factor F to improve robustness
                f_scale = 0.4 + 0.6 * np.random.rand()
                mutant = pop[r1] + f_scale * (pop[r2] - pop[r3])

                # 2. Binomial Crossover
                cross_points = np.random.rand(self.dim) < cr
                # Ensure at least one dimension is inherited from the mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])

                # 3. Boundary Handling (Clip)
                trial = np.clip(trial, lb, ub)

                # 4. Selection
                trial_y = func(trial)
                self.eval_count += 1

                if trial_y <= fitness[i]:
                    fitness[i] = trial_y
                    pop[i] = trial
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()

        return best_x, best_y
