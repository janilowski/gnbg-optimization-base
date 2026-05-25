import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE) rand/1/bin with adaptive dither, fixed crossover rate, and reflection boundary handling.
# Search state: A population of candidate solutions stored as a 2D numpy array (pop) and their objective values (fitness).
# Candidate generation: For each target vector, a mutant is created by adding a scaled difference of two random vectors to a third (all distinct and different from the target). A trial vector is produced by binomial crossover (CR=0.9) between mutant and target.
# Selection and replacement: Greedy one-to-one replacement: trial replaces target if its objective value is lower (minimization).
# Adaptation: The scaling factor F is dithering (randomly chosen per mutation from [0.5, 1.0]).
# Exploration mechanisms: Mutation uses random differential variation; crossover randomly mixes components.
# Exploitation mechanisms: Greedy selection drives convergence toward better solutions.
# Boundary handling: Reflection: out‑of‑bound coordinates are reflected back inside the feasible space.
# Budget strategy: Initial population evaluated, then generations run until the evaluation budget is exhausted, one trial per generation per individual.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: Simple dithering for F instead of a fixed value; no additional archive or ensemble.
# Failure modes: May stagnate on highly multimodal or deceptive landscapes; performance sensitive to CR and population size.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Prepare the optimizer for a given budget and search dimension.
        budget: maximum number of objective function evaluations.
        dim: number of decision variables.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimizer on the provided test function.
        func should expose bounds via func.lower / func.upper or func.bounds.lb / func.bounds.ub.
        Returns (best_x, best_y) where best_y is the minimal objective value found.
        """
        # --------------------------------------------------------------------
        # Read bounds
        # --------------------------------------------------------------------
        try:
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.array(func.bounds.lb, dtype=float)
                ub = np.array(func.bounds.ub, dtype=float)
            except AttributeError:
                raise AttributeError("Could not read bounds from func.lower/upper or func.bounds.lb/ub")

        dim = self.dim
        budget = self.budget

        # --------------------------------------------------------------------
        # Handle extremely low budget with pure random search
        # --------------------------------------------------------------------
        if budget < 4:
            best_x = None
            best_y = np.inf
            for _ in range(budget):
                x = lb + (ub - lb) * np.random.rand(dim)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # --------------------------------------------------------------------
        # Population size for DE
        # --------------------------------------------------------------------
        # Common heuristic: 10*dim, but clamp to be at most budget/3 (to allow some generations)
        # and at least 4 (required for mutation).
        pop_size = max(4, min(10 * dim, budget // 3))

        # --------------------------------------------------------------------
        # Helper: reflection boundary handler
        # --------------------------------------------------------------------
        def reflect(x):
            """Reflect any component outside [lb, ub] back into the domain."""
            # Lower bound reflection
            lower_violation = x < lb
            x[lower_violation] = lb[lower_violation] + (lb[lower_violation] - x[lower_violation])
            # Upper bound reflection
            upper_violation = x > ub
            x[upper_violation] = ub[upper_violation] - (x[upper_violation] - ub[upper_violation])
            return x

        # --------------------------------------------------------------------
        # Initialisation
        # --------------------------------------------------------------------
        pop = lb + (ub - lb) * np.random.rand(pop_size, dim)
        fitness = np.full(pop_size, np.inf)
        best_y = np.inf
        best_x = None
        evaluations_used = 0

        for i in range(pop_size):
            y = func(pop[i])
            evaluations_used += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        remaining = budget - evaluations_used

        # --------------------------------------------------------------------
        # Main DE loop (rand/1/bin with dithering)
        # --------------------------------------------------------------------
        crossover_rate = 0.9
        # Continue until budget exhausted
        while remaining > 0:
            for i in range(pop_size):
                if remaining <= 0:
                    break

                # Select three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                # F dithers uniformly between 0.5 and 1.0 per mutation
                F = 0.5 + 0.5 * np.random.rand()
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Crossover: binomial
                r = np.random.rand(dim)
                j_rand = np.random.randint(dim)
                mask = (r < crossover_rate) | (np.arange(dim) == j_rand)
                trial = np.where(mask, mutant, pop[i])

                # Boundary reflection
                trial = reflect(trial)

                # Evaluate trial
                trial_fitness = func(trial)
                remaining -= 1

                # Greedy selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
