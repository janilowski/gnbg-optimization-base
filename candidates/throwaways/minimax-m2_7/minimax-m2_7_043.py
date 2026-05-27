# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact real‑coded Genetic Algorithm (GA) that evolves a population of candidates using tournament selection, arithmetic crossover, and Gaussian mutation. The algorithm maintains an elite best‑seen solution and adapts the mutation step size σ based on the number of improvements observed in each generation.
# Search state: Population array (pop) of size μ × dim together with a parallel list of objective values (pop_vals). The global best candidate (best_x) and its objective (best_y) are tracked continuously.
# Candidate generation: Two parents are chosen by 3‑player tournament selection. An offspring is created by a per‑gene arithmetic blend (α·p1 + (1‑α)·p2) and mutated by adding σ·N(0,1). The child is clipped to the problem bounds.
# Selection and replacement: (μ+λ) replacement – parents and offspring are concatenated, then the best μ individuals are kept for the next generation.
# Adaptation: σ is increased by a factor 1.2 when >20 % of the offspring improve the global best; otherwise σ is decreased by 0.8. σ is clamped between 0.1 % and 50 % of the average bound range.
# Exploration mechanisms: Large initial σ, crossover between diverse parents, and tournament selection promote exploration.
# Exploitation mechanisms: Elite preservation of the best solution and σ reduction during the final budget consumption focus search around promising regions.
# Boundary handling: All candidate vectors are clipped to the problem’s lower/upper bounds after mutation/crossover.
# Budget strategy: The evaluation budget is first spent on a random initial population (μ = max(10,4·dim) or smaller if the budget is insufficient). Then the algorithm runs generational loops that each consume μ evaluations, until the remaining budget is too small for a full generation. Any leftover evaluations are used for random perturbations around the current best solution.
# Closest known influences: Classical GA with tournament selection and self‑adaptive mutation step size reminiscent of Evolution Strategies.
# Novelty or unusual aspects: The self‑adaptation of σ uses an improvement‑rate feedback rather than a fixed schedule, providing a simple but effective balance between exploration and exploitation.
# Failure modes: If the provided budget is too low to fill the initial population, convergence may be poor. In highly multi‑modal landscapes the static crossover parameters may not preserve sufficient diversity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Simple real‑coded Genetic Algorithm for black‑box minimization."""

    def __init__(self, budget, dim):
        """Initialize the algorithm.

        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the problem.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """Run the algorithm on the given objective function.

        Parameters
        ----------
        func : callable
            A black‑box objective to be minimized. The function accepts a
            1‑D NumPy array of length dim and returns a scalar.

        Returns
        -------
        best_x : np.ndarray
            Best solution found (vector of length dim).
        best_y : float
            Objective value at best_x.
        """
        budget = self.budget
        dim = self.dim

        # -----------------------------------------------------------------
        # Determine search bounds
        # -----------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # No bounds attribute → default to [0,1] hypercube
            lb = np.zeros(dim, dtype=float)
            ub = np.ones(dim, dtype=float)

        # Ensure bounds are the correct shape
        if lb.shape != (dim,):
            lb = np.resize(lb, dim)
        if ub.shape != (dim,):
            ub = np.resize(ub, dim)

        # -----------------------------------------------------------------
        # Population size (μ) – at least 10, otherwise scaled by dim
        # -----------------------------------------------------------------
        pop_size = max(10, dim * 4)
        if budget < pop_size:
            pop_size = budget  # cannot allocate more than the remaining budget

        # -----------------------------------------------------------------
        # Helper to evaluate a candidate while respecting the budget
        # -----------------------------------------------------------------
        best_x = np.empty(dim, dtype=float)
        best_y = np.inf

        def evaluate(candidate):
            """Evaluate one candidate, update global best, decrement budget."""
            nonlocal best_x, best_y, budget
            y = func(candidate)
            budget -= 1
            if y < best_y:
                best_y = y
                best_x = candidate.copy()
            return y

        # -----------------------------------------------------------------
        # Initial random population
        # -----------------------------------------------------------------
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        pop_vals = [evaluate(pop[i]) for i in range(pop_size)]

        # If budget exhausted after the initial seed, return immediately
        if budget <= 0:
            return best_x, best_y

        # -----------------------------------------------------------------
        # Mutation step size self‑adaptation parameters
        # -----------------------------------------------------------------
        range_mean = np.mean(ub - lb)
        sigma = range_mean * 0.1               # initial step size
        min_sigma = range_mean * 1e-3
        max_sigma = range_mean * 0.5

        improve_thr = int(0.2 * pop_size)      # expected improvements per generation

        # -----------------------------------------------------------------
        # Tournament selection helper
        # -----------------------------------------------------------------
        def tournament_selection(values, tsize=3):
            """Return index of the best among a random tournament."""
            indices = np.random.choice(len(values), size=tsize, replace=False)
            best_idx = indices[0]
            best_val = values[best_idx]
            for idx in indices[1:]:
                if values[idx] < best_val:
                    best_val = values[idx]
                    best_idx = idx
            return best_idx

        # -----------------------------------------------------------------
        # Generational evolution loop
        # -----------------------------------------------------------------
        while budget >= pop_size:
            offspring = []
            offspring_vals = []
            improvements = 0

            for _ in range(pop_size):
                # ---- Selection ----
                p1_idx = tournament_selection(pop_vals)
                p2_idx = tournament_selection(pop_vals)
                p1, p2 = pop[p1_idx], pop[p2_idx]

                # ---- Crossover (arithmetic blend) ----
                alpha = np.random.rand(dim)
                child = alpha * p1 + (1.0 - alpha) * p2

                # ---- Mutation ----
                child += sigma * np.random.randn(dim)

                # ---- Boundary handling ----
                child = np.clip(child, lb, ub)

                # ---- Evaluation ----
                val = evaluate(child)
                offspring.append(child)
                offspring_vals.append(val)
                if val < best_y:
                    improvements += 1

            # ---- (μ+λ) Replacement ----
            combined_pop = np.vstack([pop, offspring])
            combined_vals = pop_vals + offspring_vals

            # Keep the best μ individuals
            sorted_indices = np.argsort(combined_vals)
            pop = combined_pop[sorted_indices[:pop_size]]
            pop_vals = [combined_vals[i] for i in sorted_indices[:pop_size]]

            # ---- Adapt σ based on improvement rate ----
            if improvements > improve_thr:
                sigma = min(sigma * 1.2, max_sigma)
            else:
                sigma = max(sigma * 0.8, min_sigma)

        # -----------------------------------------------------------------
        # Utilise any remaining budget with random perturbations around best
        # -----------------------------------------------------------------
        while budget > 0:
            candidate = best_x + sigma * np.random.randn(dim)
            candidate = np.clip(candidate, lb, ub)
            evaluate(candidate)
            # Slowly reduce
