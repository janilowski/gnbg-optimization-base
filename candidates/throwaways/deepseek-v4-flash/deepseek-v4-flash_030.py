import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple differential evolution (DE/rand/1/bin) minimizer for GNBG benchmarks.
# Search state: a population of candidate vectors, each of dimension dim, stored in array pop,
# and their corresponding objective values fpop. The current best individual and its value are
# tracked externally.
# Candidate generation: for each target vector, a mutant is created by adding the scaled
# difference of two random distinct population members to a third random member (rand/1).
# The scale factor F is drawn uniformly from [0.5, 1.0] per mutation.
# Selection and replacement: binomial crossover between mutant and target uses crossover
# probability CR = 0.9. The trial replaces the target if it yields a lower (minimization)
# objective value.
# Adaptation: F and CR are fixed across the run (F sampled uniformly each mutation within
# a fixed range, CR constant). No adaptation of control parameters.
# Exploration mechanisms: large F values and random parent selection promote exploration;
# population diversity is maintained by replacing inferior individuals.
# Exploitation mechanisms: selection pressure pushes population toward better regions;
# crossover allows mixing of good parameters.
# Boundary handling: trial vectors that violate bounds are reflected back into the domain.
# Budget strategy: population size is chosen so that the number of generations (budget / popsize)
# is roughly 30–200, ensuring a reasonable number of iterations while staying within budget.
# The budget is tracked precisely; evaluation stops as soon as the budget is exhausted.
# Closest known influences: classic DE (Storn & Price, 1997).
# Novelty or unusual aspects: none – this is a straightforward implementation of a well-known
# algorithm, chosen for robustness and simplicity.
# Failure modes: may converge prematurely on highly multimodal or deceptive landscapes,
# especially with limited budget. The fixed CR and F may be suboptimal for some problems.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Prepare the optimizer for a black-box minimization task.
        
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimension of the search space.
        """
        self.budget = budget
        self.dim = dim

        # Determine population size.
        # We want enough individuals for diversity but not too many that
        # generations are too few.  A heuristic: popsize around 4*dim but
        # no less than 10 and no more than 100, and also ensure that
        # at least 20 generations can be run (or use all budget if dim is small).
        ideal_size = int(4 * dim)
        # Clamp to sensible range
        self.popsize = max(10, min(100, ideal_size))
        # If the budget is so small that even one generation would exceed it,
        # reduce popsize to at most budget // 2 (at least 2 evaluations).
        if self.popsize > self.budget:
            self.popsize = max(2, self.budget // 2)
        # Now compute the effective number of generations:
        # each generation costs popsize evaluations (except the initial one).
        # We will do floor((budget - popsize) / popsize) generations after initial.
        # But we can also allow a fractional last generation? We'll stop exactly when budget exhausted.
        # So we set max_generations = (budget - popsize) // popsize  (if positive)
        self.max_generations = (budget - self.popsize) // self.popsize if budget > self.popsize else 0

    def __call__(self, func) -> tuple[np.ndarray, float]:
        """
        Run the optimization algorithm on the given function.

        Parameters
        ----------
        func : callable
            The objective function. Must return a scalar float.
            Bounds are read from func.lower / func.upper or (fallback)
            func.bounds.lb / func.bounds.ub.

        Returns
        -------
        best_x : np.ndarray
            Best found solution.
        best_y : float
            Objective value at best_x.
        """
        # ---------- Determine bounds ----------
        try:
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        except AttributeError:
            try:
                b = func.bounds
                lb = np.array(b.lb, dtype=float)
                ub = np.array(b.ub, dtype=float)
            except AttributeError:
                raise RuntimeError("Cannot determine bounds from func")
        # Ensure arrays
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        # ---------- Helper: reflect into bounds ----------
        def reflect(x):
            """Reflect coordinate back into [lb, ub]."""
            # Reflect in a loop until inside (usually one or two reflections suffice)
            # But we do it as a simple while for each coordinate:
            for i in range(self.dim):
                while True:
                    if x[i] < lb[i]:
                        x[i] = 2 * lb[i] - x[i]
                    elif x[i] > ub[i]:
                        x[i] = 2 * ub[i] - x[i]
                    else:
                        break
            return x

        # ---------- Initialize population ----------
        pop = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        fpop = np.full(self.popsize, np.inf)
        best_x = None
        best_y = np.inf
        evals = 0

        # Evaluate initial population
        for i in range(self.popsize):
            if evals >= self.budget:
                # Not enough budget; we'll fill remaining with Inf and break
                fpop[i] = np.inf
                continue
            y = func(pop[i])
            evals += 1
            fpop[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # ---------- Main generation loop ----------
        generation = 0
        while generation < self.max_generations and evals < self.budget:
            generation += 1
            # For each target vector, create trial via DE/rand/1/bin
            for i in range(self.popsize):
                if evals >= self.budget:
                    break

                # Choose three random indices distinct from each other and from i
                r = np.random.choice(self.popsize, 3, replace=False)
                while i in r:   # very unlikely, but ensure
                    r = np.random.choice(self.popsize, 3, replace=False)
                a, b, c = r

                # Mutation scale factor
                F = np.random.uniform(0.5, 1.0)

                # Mutant
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover with probability CR
                CR = 0.9
                trial = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary handling
                trial = reflect(trial)

                # Evaluate if budget allows
                if evals >= self.budget:
                    break
                y_trial = func(trial)
                evals += 1
                if y_trial < fpop[i]:
                    pop[i] = trial
                    fpop[i] = y_trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()

            # Optional: if we ran out of budget during the inner loop, break outer
            if evals >= self.budget:
                break

        # In case no evaluation happened (should not happen if budget > 0)
        if best_x is None:
            # Fallback: return the first individual
            if self.popsize > 0 and evals > 0:
                best_x = pop[0].copy()
                best_y = fpop[0]
            else:
                best_x = np.random.uniform(lb, ub)
                best_y = np.inf

        return best_x, best_y
