import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A dithered Differential Evolution (DE/rand/1/bin) algorithm designed for robust black-box optimization.
# Search state: A population of candidate solutions and their corresponding fitness values, alongside the current best solution found.
# Candidate generation: Mutation is performed using the 'rand/1' strategy (one random base vector plus a scaled difference of two other random vectors), followed by binomial crossover with the target vector.
# Selection and replacement: A greedy replacement strategy is used where a trial vector replaces its parent in the population only if it yields a lower or equal objective value.
# Adaptation: The mutation scaling factor (F) and the crossover probability (Cr) are dithered (randomly sampled) for each trial to maintain diversity and handle various search scales.
# Exploration mechanisms: Initial uniform random population sampling and the stochastic nature of the differential mutation operator provide global search capabilities.
# Exploitation mechanisms: Greedy selection ensures the population moves towards better regions, while the differential nature of the mutation naturally reduces step sizes as the population converges.
# Boundary handling: Trial vectors are clipped to the feasible region defined by the problem's lower and upper bounds.
# Budget strategy: A strict evaluation counter is maintained; the algorithm terminates immediately when the evaluation budget is exhausted, even during population initialization or mid-iteration.
# Closest known influences: The original Differential Evolution algorithm by Storn and Price, with parameter dithering techniques common in modern DE variants like JADE.
# Novelty or unusual aspects: Uses a simplified parameter dithering approach to achieve robustness across different problem types without the computational overhead of complex adaptive feedback loops.
# Failure modes: May converge prematurely on highly multi-modal landscapes if the population size is too small, or may progress slowly on extremely high-dimensional problems with very limited budgets.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the optimization algorithm.
        
        Args:
            budget (int): Total number of function evaluations allowed.
            dim (int): Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        """
        Runs the minimization process on the provided objective function.
        
        Args:
            func: The objective function to minimize.
        
        Returns:
            tuple: (best_x, best_y) representing the best solution found.
        """
        # Determine problem bounds
        if hasattr(func, 'lower') and func.lower is not None:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback bounds if not provided
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)

        best_x = None
        best_y = float('inf')

        def safe_eval(x):
            """Evaluates the function while respecting the budget."""
            nonlocal best_x, best_y
            if self.eval_count >= self.budget:
                return None
            
            y = func(x)
            self.eval_count += 1
            
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Heuristic for population size: at least 4 for DE/rand/1, capped for efficiency
        pop_size = max(4, min(10 * self.dim, 50))
        if self.budget < pop_size:
            pop_size = self.budget

        # Handle extremely low budget cases with simple random sampling
        if pop_size < 4:
            while self.eval_count < self.budget:
                x = lb + np.random.rand(self.dim) * (ub - lb)
                safe_eval(x)
            return best_x, best_y

        # Initialize population
        pop = []
        fitness = []
        for i in range(pop_size):
            x = lb + np.random.rand(self.dim) * (ub - lb)
            y = safe_eval(x)
            if y is not None:
                pop.append(x)
                fitness.append(y)
            else:
                break

        # Main Evolution Loop
        while self.eval_count < self.budget:
            for i in range(len(pop)):
                # Mutation: DE/rand/1
                # Select 3 distinct indices from population excluding the current index i
                idxs = [idx for idx in range(len(pop)) if idx != i]
                if len(idxs) < 3:
                    # Not enough individuals to mutate, break to outer loop/randomize
                    break
                
                abc_idxs = np.random.choice(idxs, 3, replace=False)
                a, b, c = pop[abc_idxs[0]], pop[abc_idxs[1]], pop[abc_idxs[2]]

                # Parameter Dithering (Adaptation)
                F = np.random.uniform(0.4, 1.0)
                Cr = np.random.uniform(0.5, 0.9)

                # Create mutant vector
                mutant = a + F * (b - c)

                # Binomial Crossover
                cross_points = np.random.rand(self.dim) < Cr
                # Ensure at least one dimension is exchanged
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary Handling: Clipping
                trial = np.clip(trial, lb, ub)

                # Evaluation and Greedy Selection
                y_trial = safe_eval(trial)
                if y_trial is not None:
                    if y_trial <= fitness[i]:
                        pop[i] = trial
                        fitness[i] = y_trial
                else:
                    # Budget exhausted
                    break
            
            if self.eval_count >= self.budget:
                break

        return best_x, best_y
