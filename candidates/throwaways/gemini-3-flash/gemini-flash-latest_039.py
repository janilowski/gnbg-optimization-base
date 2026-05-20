import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm implements a self-adaptive Differential Evolution (jDE) variant. It maintains a population of solutions and adjusts its mutation (F) and crossover (CR) parameters for each individual based on their success.
# Search state: The state consists of a population of vectors (candidate solutions), their corresponding objective function values, and per-individual control parameters (F and CR).
# Candidate generation: Offspring are generated using the DE/rand/1 mutation strategy (base vector plus scaled difference of two others) followed by binomial crossover.
# Selection and replacement: The algorithm uses a greedy one-to-one selection mechanism where an offspring replaces its parent in the population only if its fitness is less than or equal to the parent's fitness.
# Adaptation: Parameters F and CR are updated using a probabilistic self-adaptation scheme (tau strategy), where new values are sampled if a random threshold is met, allowing the search to adapt to the local landscape.
# Exploration mechanisms: Exploration is driven by the differential mutation strategy and the diversity of the initial population, as well as the stochastic adaptation of the scaling factor F.
# Exploitation mechanisms: Exploitation is facilitated by the crossover operator, which combines dimensions of successful candidates, and the greedy selection process.
# Boundary handling: Offspring vectors that fall outside the search space are clipped to the hyper-rectangle defined by the lower and upper bounds.
# Budget strategy: The algorithm tracks evaluations strictly using a counter and terminates exactly when the budget is exhausted, ensuring the best solution found is returned.
# Closest known influences: The jDE algorithm by Brest et al. (2006).
# Novelty or unusual aspects: The implementation includes a fallback to random search for extremely low budgets where the population-based DE cannot be initialized.
# Failure modes: The algorithm may converge prematurely on highly multi-modal landscapes if the population size is too small or the budget is insufficient to escape local optima.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the Differential Evolution algorithm.
        
        Args:
            budget: Total number of function evaluations allowed.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        
        # Determine population size: heuristic 10*dim capped at 50 to maintain efficiency.
        # If the budget is very small, we scale the population size down.
        self.np_size = int(max(4, min(10 * dim, 50)))
        if self.np_size * 2 > self.budget:
            self.np_size = max(1, self.budget // 2)

    def __call__(self, func):
        """
        Executes the optimization process.
        
        Args:
            func: Objective function to minimize.
            
        Returns:
            (best_x, best_y): The best solution found and its objective value.
        """
        # Retrieve bounds from the function object
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        else:
            lb, ub = np.zeros(self.dim), np.ones(self.dim)

        evals = 0
        best_y = float('inf')
        best_x = lb + (ub - lb) / 2.0  # Fallback initialization

        if self.budget <= 0:
            return best_x, best_y

        # Fallback for very low budgets: simple random sampling
        if self.np_size < 4:
            while evals < self.budget:
                x = lb + np.random.rand(self.dim) * (ub - lb)
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y, best_x = y, x
            return best_x, best_y

        # Initialize population and parameters
        pop = lb + np.random.rand(self.np_size, self.dim) * (ub - lb)
        # params stores [F, CR] for each individual
        params = np.column_stack([
            np.full(self.np_size, 0.5), # Initial F
            np.full(self.np_size, 0.9)  # Initial CR
        ])
        fitness = np.zeros(self.np_size)

        # Initial evaluation of the population
        for i in range(self.np_size):
            if evals >= self.budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_y:
                best_y, best_x = fitness[i], pop[i].copy()

        # Adaptation hyperparameters
        tau1 = 0.1 # Probability to change F
        tau2 = 0.1 # Probability to change CR
        F_l, F_u = 0.1, 0.9 # Range for F

        # Main evolution loop
        while evals < self.budget:
            for i in range(self.np_size):
                if evals >= self.budget:
                    break
                
                # Self-adaptation of parameters F and CR
                f_i, cr_i = params[i]
                if np.random.rand() < tau1:
                    f_i = F_l + np.random.rand() * F_u
                if np.random.rand() < tau2:
                    cr_i = np.random.rand()
                
                # Selection of three distinct random parents other than i
                candidates = [idx for idx in range(self.np_size) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                
                # Mutation (DE/rand/1)
                mutant = pop[r1] + f_i * (pop[r2] - pop[r3])
                
                # Crossover (Binomial)
                cross_points = np.random.rand(self.dim) < cr_i
                if not np.any(cross_points):
                    # Ensure at least one component is changed
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary handling: clip to hyper-rectangle
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                f_trial = func(trial)
                evals += 1
                
                # Greedy selection
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    params[i] = [f_i, cr_i]
                    if f_trial < best_y:
                        best_y, best_x = f_trial, trial.copy()
                        
        return best_x, best_y
