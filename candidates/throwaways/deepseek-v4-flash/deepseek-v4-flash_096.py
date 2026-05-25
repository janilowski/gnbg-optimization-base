import numpy as np
from typing import Tuple, Any

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A black-box minimization algorithm based on Differential Evolution (DE)
#          with reflection boundary handling, adaptive mutation/crossover parameters,
#          and a simple restart mechanism to avoid stagnation.
# Search state: A population of candidate solutions (real vectors) and their
#               corresponding objective values, plus an elite candidate (best so far).
# Candidate generation: For each parent, a mutant is created using the DE/rand/1
#                       scheme. F (scale factor) and CR (crossover rate) are adapted
#                       based on the recent success of generating improvements.
# Selection and replacement: Each parent is greedily replaced by its trial offspring
#                            if the offspring has a lower (better) objective value.
# Adaptation: F and CR are adjusted dynamically: when the best value has not improved
#             for a given number of evaluations, F is increased and CR decreased to
#             promote exploration; otherwise they are tightened toward
#             standard values (F=0.8, CR=0.9) to encourage exploitation.
# Exploration mechanisms: The DE/rand/1 mutation introduces diversity through
#                         random vector differences; the restart mechanism
#                         reinitializes the population (except the best) to
#                         explore distant regions after stagnation.
# Exploitation mechanisms: Greedy parent–offspring replacement and the gradual
#                          tightening of F and CR around standard values favour
#                          local refinement near promising areas.
# Boundary handling: Reflected points are used when a mutant coordinate falls
#                    outside the domain (periodic reflection using the lower/upper bounds).
# Budget strategy: The algorithm terminates exactly when the evaluation budget is exhausted
#                 (evaluation counter checked before each call to the objective function).
# Closest known influences: Classic DE (Storn & Price), self-adaptive DE variants,
#                           and restart strategies (e.g., IPOP-CMA-ES).
# Novelty or unusual aspects: Simple online adaptation of F and CR based on a
#                             sliding-window success rate.
# Failure modes: May be slow on highly multimodal landscapes if the budget is very low;
#                reflection boundary handling can alias solutions near corners.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the algorithm with a given budget and dimension.

        Args:
            budget: Maximum number of function evaluations allowed.
            dim:    Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func: Any) -> Tuple[np.ndarray, float]:
        """
        Run the algorithm on the provided objective function.

        Args:
            func: An object that implements the __call__(x) interface
                  and exposes lower/upper bounds via either
                  `func.lower` / `func.upper` or `func.bounds.lb` / `func.bounds.ub`.

        Returns:
            (best_x, best_y): The best found solution and its objective value.
        """
        # ------------------------------------------------------------------ #
        # 1. Extract bounds
        # ------------------------------------------------------------------ #
        try:
            lb = np.atleast_1d(np.asarray(func.lower, dtype=float))
            ub = np.atleast_1d(np.asarray(func.upper, dtype=float))
        except AttributeError:
            lb = np.atleast_1d(np.asarray(func.bounds.lb, dtype=float))
            ub = np.atleast_1d(np.asarray(func.bounds.ub, dtype=float))

        # Ensure bounds are arrays of length dim
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)
        elif lb.shape[0] != self.dim:
            # If only one pair given, broadcast
            lb = np.broadcast_to(lb, (self.dim,))
            ub = np.broadcast_to(ub, (self.dim,))

        # Compute domain widths (used for reflection)
        width = ub - lb

        # ------------------------------------------------------------------ #
        # 2. Algorithm parameters
        # ------------------------------------------------------------------ #
        n_evals = 0
        # Population size: 4*dim, capped at 100 and at least 10
        NP = max(10, min(100, 4 * self.dim))
        # Base mutation and crossover rates
        F_base = 0.8
        CR_base = 0.9
        # Current adaptive values – start with the base values
        F = F_base
        CR = CR_base
        # Stagnation detection
        max_stag_evals = max(10, int(0.15 * self.budget))
        stag_counter = 0
        best_y = np.inf
        best_x = None
        # Sliding window for success rate (last 5*NP evaluations)
        success_window_size = int(5 * NP)
        success_count = 0
        eval_count_since_reset = 0

        # ------------------------------------------------------------------ #
        # 3. Initialisation
        # ------------------------------------------------------------------ #
        pop = np.random.uniform(lb, ub, size=(NP, self.dim))
        pop_y = np.full(NP, np.inf)
        for i in range(NP):
            pop_y[i] = func(pop[i])
            n_evals += 1
            if pop_y[i] < best_y:
                best_y = pop_y[i]
                best_x = pop[i].copy()

        # ------------------------------------------------------------------ #
        # 4. Main loop
        # ------------------------------------------------------------------ #
        while n_evals < self.budget:
            # --- 4a. Generate offspring population ------------------------- #
            for i in range(NP):
                # Stop if budget exhausted
                if n_evals >= self.budget:
                    break

                # Choose three distinct random indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Mutation (DE/rand/1)
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Boundary reflection (periodic)
                # For each coordinate that is out of bounds, reflect symmetrically
                # inside the domain.
                for d in range(self.dim):
                    if mutant[d] < lb[d]:
                        diff = lb[d] - mutant[d]
                        # Reflect around the lower bound
                        mutant[d] = lb[d] + (diff % width[d])
                    elif mutant[d] > ub[d]:
                        diff = mutant[d] - ub[d]
                        mutant[d] = ub[d] - (diff % width[d])
                # Ensure no numeric drift
                mutant = np.clip(mutant, lb, ub)

                # Binomial crossover
                cross_points = np.random.rand(self.dim) < CR
                # At least one component is replaced
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Evaluate trial
                trial_y = func(trial)
                n_evals += 1
                eval_count_since_reset += 1

                # Selection: greedy replacement
                if trial_y <= pop_y[i]:
                    pop[i] = trial
                    pop_y[i] = trial_y
                    success_count += 1
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()
                        stag_counter = 0
                    else:
                        stag_counter += 1
                else:
                    stag_counter += 1

            # --- 4b. Stagnation check and possible restart ----------------- #
            if stag_counter > max_stag_evals:
                # Restart: reinitialize all but the best individual
                # Keep best_x as is, but we will reinsert it after reinitialisation
                new_pop = np.random.uniform(lb, ub, size=(NP, self.dim))
                # Replace one random index with the current best
                idx_best = np.random.randint(0, NP)
                new_pop[idx_best] = best_x.copy()
                # Evaluate new individuals if not already evaluated
                for i in range(NP):
                    if n_evals >= self.budget:
                        break
                    if i == idx_best:
                        continue  # best already evaluated
                    new_y = func(new_pop[i])
                    n_evals += 1
                    eval_count_since_reset += 1
                    if new_y < best_y:
                        best_y = new_y
                        best_x = new_pop[i].copy()
                    pop_y[i] = new_y
                pop = new_pop
                # Reset stagnation counter and success counter
                stag_counter = 0
                success_count = 0
                eval_count_since_reset = 0
                # Reset F and CR to default
                F = F_base
                CR = CR_base
                continue

            # --- 4c. Adaptive parameter update (every window evaluations) -- #
            if eval_count_since_reset >= success_window_size:
                # Success rate
                succ_rate = success_count / max(eval_count_since_reset, 1)
                if succ_rate < 0.2:
                    # Too few successes: increase exploration
                    F = min(1.5, F * 1.1)
                    CR = max(0.3, CR * 0.9)
                elif succ_rate > 0.5:
                    # Many successes: encourage exploitation
                    F = max(0.3, F * 0.95)
                    CR = min(0.95, CR * 1.05)
                else:
                    # Drift back toward base values
                    F += (F_base - F) * 0.1
                    CR += (CR_base - CR) * 0.1
                # Reset window counters
                success_count = 0
                eval_count_since_reset = 0

        return best_x, best_y
