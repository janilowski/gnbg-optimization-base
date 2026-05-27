# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adaptive Differential Evolution (jDE) variant designed for robust black-box minimization.
# Search state: A population of candidate solution vectors, their objective values, and individual-specific mutation (F) and crossover (CR) parameters.
# Candidate generation: Uses the DE/rand/1/bin strategy, where a mutant vector is generated from three random population members and combined with the parent vector via binomial crossover.
# Selection and replacement: Greedy selection where a trial vector replaces its parent in the population only if it yields a lower or equal objective value.
# Adaptation: Control parameters F and CR are adapted per individual using a stochastic update rule (jDE style) to balance exploration and exploitation dynamically.
# Exploration mechanisms: Differential mutation driven by the population variance and parameter randomization to prevent stagnation.
# Exploitation mechanisms: Greedy selection and the contraction of the population around local minima.
# Boundary handling: Hard clipping of candidate vectors to the hyperrectangle defined by the function bounds.
# Budget strategy: A strict evaluation counter ensures the search terminates exactly when the budget is exhausted, even if mid-generation.
# Closest known influences: jDE (Brest et al., 2006), standard Differential Evolution (Storn & Price).
# Novelty or unusual aspects: Highly compact implementation of jDE with per-evaluation budget monitoring.
# Failure modes: May converge prematurely on highly multimodal landscapes if the population size is too small relative to the complexity.
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

    def __call__(self, func):
        """
        Executes the optimization process.
        :param func: The objective function to minimize.
        :return: (best_x, best_y) - the best solution found.
        """
        # Extract bounds from func
        try:
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        except AttributeError:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)

        # Ensure bounds are arrays of correct dimension
        if lb.ndim == 0: lb = np.full(self.dim, lb)
        if ub.ndim == 0: ub = np.full(self.dim, ub)

        # Hyperparameters for the jDE adaptation
        fl, fu = 0.1, 0.9  # Mutation factor range
        tau1, tau2 = 0.1, 0.1 # Probabilities for parameter update
        
        # Population size: heuristic balance between diversity and budget
        pop_size = min(self.budget, max(10, int(10 + 2 * np.log(self.dim))))
        
        # Initialization
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.full(pop_size, np.inf)
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        evals = 0
        best_x = None
        best_y = np.inf

        # Evaluate initial population
        for i in range(pop_size):
            if evals >= self.budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()

        # Evolution loop
        while evals < self.budget:
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Adaptation of F and CR (jDE logic)
                fi = F[i] if np.random.rand() > tau1 else (fl + np.random.rand() * fu)
                cri = CR[i] if np.random.rand() > tau2 else np.random.rand()
                
                # Mutation: DE/rand/1
                indices = [idx for idx in range(pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                mutant = pop[r1] + fi * (pop[r2] - pop[r3])
                
                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) <= cri
                if not np.any(cross_points): # Ensure at least one component is changed
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary handling: Clipping
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                f_trial = func(trial)
                evals += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    F[i] = fi
                    CR[i] = cri
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_y
