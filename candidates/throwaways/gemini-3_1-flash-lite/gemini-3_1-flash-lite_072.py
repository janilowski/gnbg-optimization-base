# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (Rao-1) inspired population-based metaheuristic.
# Search state: A population of vectors maintained in a numpy array of shape (pop_size, dim).
# Candidate generation: Generates new agents by linearly combining the difference between the current best and worst agents with the current agent's position.
# Selection and replacement: Greedy selection; a new candidate replaces the current agent if it yields a lower function value.
# Adaptation: None; parameters are static to ensure budget efficiency.
# Exploration mechanisms: Utilizes randomized weights to scale contributions from the best and worst individuals.
# Exploitation mechanisms: Attracts agents toward the current global best through the difference vector.
# Boundary handling: Clamping to the search space defined by the objective function's bounds.
# Budget strategy: Static population size adjusted based on dimensionality to ensure sufficient iterations.
# Closest known influences: Rao Algorithm (Physics-based optimization).
# Novelty or unusual aspects: Extremely compact implementation of a population-based search for black-box minimization.
# Failure modes: Susceptible to local optima in highly multi-modal landscapes with limited budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Population size scaling: small enough for low budget, large enough for exploration
        self.pop_size = min(50, max(10, 2 * dim))
        self.max_iter = max(1, budget // (self.pop_size + 1))

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        y = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(y)
        best_x = pop[best_idx].copy()
        best_y = y[best_idx]
        
        evals = self.pop_size
        
        # Iteration loop
        for _ in range(self.max_iter):
            if evals >= self.budget:
                break
                
            best_agent = pop[np.argmin(y)]
            worst_agent = pop[np.argmax(y)]
            
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Rao-inspired candidate generation: x_new = x + r1*(best - worst)
                r1 = np.random.rand(self.dim)
                candidate = pop[i] + r1 * (best_agent - worst_agent)
                
                # Clamping
                candidate = np.clip(candidate, lb, ub)
                
                # Evaluation
                cand_y = func(candidate)
                evals += 1
                
                # Greedy selection
                if cand_y < y[i]:
                    pop[i] = candidate
                    y[i] = cand_y
                    
                    if cand_y < best_y:
                        best_y = cand_y
                        best_x = candidate.copy()
                        
        return best_x, best_y
