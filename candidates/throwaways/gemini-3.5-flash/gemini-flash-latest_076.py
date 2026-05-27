# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm is an adaptive memetic optimization framework designed to handle black-box minimization tasks robustly across varying dimensionalities and evaluation budgets. It dynamically switches between a specialized adaptive random-walk local search (for extremely low budgets or high-dimensional bottlenecks) and a Self-Adaptive Differential Evolution (jDE) hybrid with periodic Local Search (for standard budgets).
# Search state: The search state consists of a population of candidate solutions, their corresponding fitness values, self-adaptive control parameters (scaling factor F and crossover rate CR) for each individual, the global best solution (best_x, best_y), and the running count of objective function evaluations.
# Candidate generation: In the DE mode, candidate generation uses a "current-to-pbest/1" mutation strategy combined with binomial crossover. If the mutated parameters fall out of bounds, a bounce-back strategy is employed. In the local search mode, candidates are generated using an adaptive-step-size random walk along random search hyperspheres and coordinate directions.
# Selection and replacement: Selection is elitist. A trial vector replaces its parent in the DE population if its objective function value is strictly better. The global best tracker is greedily updated whenever any evaluation yields a lower objective value.
# Adaptation: Control parameters F and CR are self-adapted on an individual basis. With a small probability (e.g., 10%), they are re-sampled to allow the search to adapt to different landscapes (flat, rugged, etc.). The local search step-size (sigma) dynamically expands (multiplied by 1.2) upon a successful improvement and contracts (multiplied by 0.5) upon failure.
# Exploration mechanisms: Exploration is maintained via the differential mutation vectors, self-adaptive CR values enabling larger coordinate changes, and randomized restarts or direction shifts in the local search when improvement stalls.
# Exploitation mechanisms: Exploitation is driven by the "current-to-pbest/1" mutation topology, directing search towards the best-performing regions, and is heavily reinforced by the direct, adaptive-step-size local search executed periodically on the current best individual.
# Boundary handling: To prevent stagnation and projection clustering on boundaries, we implement a soft bounce-back strategy. If a trial vector violates a boundary, it is re-projected slightly inside the feasible region rather than hard-clipped.
# Budget strategy: The algorithm strictly monitors its budget. To prevent waste, population size is scaled dynamically with the budget and dimension. If the budget is too small to support a stable DE population (e.g., budget < 15 * dim), the algorithm bypasses DE entirely and uses an intensive, restart-capable local search.
# Closest known influences: Self-adaptive Differential Evolution (jDE by Brest et al.), SHADE, and pattern search / local random-walk optimization algorithms.
# Novelty or unusual aspects: The seamless transitions between DE and adaptive random-walk local search based on the ratio of remaining budget to dimension, and the integration of a highly reactive bounce-back boundary handler.
# Failure modes: High-dimensional highly non-separable landscapes with extremely tight budgets where cooperative co-evolution or coordinate-aligned steps are strictly required.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.best_x = None
        self.best_y = float('inf')

    def _safe_eval(self, x, func, lb, ub):
        """Evaluates the objective function while ensuring budget limits are respected."""
        if self.evals >= self.budget:
            return float('inf')
        
        # Ensure the candidate is within bounds (safety clip)
        x_clipped = np.clip(x, lb, ub)
        try:
            val = func(x_clipped)
        except Exception:
            val = float('inf')
            
        self.evals += 1
        
        if val < self.best_y:
            self.best_y = val
            self.best_x = x_clipped.copy()
            
        return val

    def _local_search(self, func, lb, ub, max_evals):
        """Performs an adaptive-step random walk around the current best solution."""
        if self.best_x is None:
            return

        step_size = 0.05 * (ub - lb)
        consecutive_failures = 0
        
        while self.evals < self.budget and max_evals > 0:
            # Generate a candidate step
            direction = np.random.randn(self.dim)
            norm = np.linalg.norm(direction)
            if norm < 1e-30:
                direction = np.zeros(self.dim)
                direction[np.random.randint(0, self.dim)] = 1.0
            else:
                direction /= norm
            
            trial = self.best_x + step_size * direction
            # Soft boundary handling (bounce-back)
            out_ub = trial > ub
            out_lb = trial < lb
            if np.any(out_ub) or np.any(out_lb):
                trial[out_ub] = ub[out_ub] - np.random.rand(np.sum(out_ub)) * (ub[out_ub] - lb[out_ub]) * 0.05
                trial[out_lb] = lb[out_lb] + np.random.rand(np.sum(out_lb)) * (ub[out_lb] - lb[out_lb]) * 0.05
            
            val = self._safe_eval(trial, func, lb, ub)
            max_evals -= 1
            
            if val < self.best_y:
                # Success: expand step size slightly and reset failure counter
                step_size *= 1.2
                consecutive_failures = 0
            else:
                # Failure: shrink step size
                step_size *= 0.5
                consecutive_failures += 1
                
                # Try a localized step in the exact opposite direction as a quick check
                if self.evals < self.budget and max_evals > 0:
                    trial_opp = self.best_x - 0.5 * step_size * direction
                    trial_opp = np.clip(trial_opp, lb, ub)
                    val_opp = self._safe_eval(trial_opp, func, lb, ub)
                    max_evals -= 1
                    if val_opp < self.best_y:
                        step_size *= 1.1
                        consecutive_failures = 0
            
            # If step size becomes negligible, reset it to encourage exploration
            if np.max(step_size / (ub - lb + 1e-30)) < 1e-7 or consecutive_failures > 15:
                step_size = 0.05 * (ub - lb) * (0.1 + 0.9 * np.random.rand())
                consecutive_failures = 0

    def __call__(self, func):
        self.evals = 0
        self.best_x = None
        self.best_y = float('inf')

        # 1. Retrieve search bounds safely
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.atleast_1d(func.lower)
            ub = np.atleast_1d(func.upper)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.atleast_1d(func.bounds.lb)
            ub = np.atleast_1d(func.bounds.ub)
        else:
            lb = np.full(self.dim, -10.0)
            ub = np.full(self.dim, 10.0)

        # Ensure bounds have correct dimensionality and no infinite boundaries
        lb = np.where(np.isinf(lb), -100.0, lb)
        ub = np.where(np.isinf(ub), 100.0, ub)
        if len(lb) != self.dim:
            lb = np.full(self.dim, lb[0] if len(lb) > 0 else -10.0)
        if len(ub) != self.dim:
            ub = np.full(self.dim, ub[0] if len(ub) > 0 else 10.0)

        # 2. Strategy selection based on available budget
        # If budget is extremely tight, use direct adaptive search immediately
        if self.budget < 15 * self.dim or self.budget < 40:
            # Initial random sampling to locate a reasonable starting point
            initial_samples = max(2, min(self.budget // 4, 10))
            for _ in range(initial_samples):
                if self.evals >= self.budget:
                    break
                sample = np.random.uniform(lb, ub)
                self._safe_eval(sample, func, lb, ub)
            
            # Fallback to center point if we haven't evaluated it
            if self.evals < self.budget:
                self._safe_eval(0.5 * (lb + ub), func, lb, ub)
            
            # Local search with the remaining budget
            remaining = self.budget - self.evals
            if remaining > 0:
                self._local_search(func, lb, ub, remaining)
                
            return self.best_x, self.best_y

        # Memetic DE Strategy for normal/large budgets
        pop_size = max(10, min(50, 3 * self.dim))
        pop_size = min(pop_size, self.budget // 3)
        if pop_size < 4:
            pop_size = 4

        # Initialize Population
        pop = np.zeros((pop_size, self.dim))
        pop_y = np.zeros(pop_size)
        
        # Latin Hypercube-like initialization or stratified random
        for i in range(pop_size):
            pop[i] = lb + (ub - lb) * (i + np.random.rand(self.dim)) / pop_size
            pop[i] = np.clip(pop[i], lb, ub)
            pop_y[i] = self._safe_eval(pop[i], func, lb, ub)

        # Self-adaptive parameter initialization (jDE style)
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)

        gen = 0
        while self.evals < self.budget:
            # Periodic Local Search around the best individual
            if gen % 10 == 0 and self.evals < self.budget:
                # Dedicate up to 10% of remaining budget or 2 * dim, whichever is larger, to Local Search
                ls_budget = max(2 * self.dim, (self.budget - self.evals) // 10)
                ls_budget = min(ls_budget, self.budget - self.evals)
                if ls_budget > 0:
                    self._local_search(func, lb, ub, ls_budget)
                    # Update population's best if it got improved
                    best_idx = np.argmin(pop_y)
                    if self.best_y < pop_y[best_idx]:
                        pop[best_idx] = self.best_x.copy()
                        pop_y[best_idx] = self.best_y

            for i in range(pop_size):
                if self.evals >= self.budget:
                    break
                
                # Parameter adaptation
                if np.random.rand() < 0.1:
                    F[i] = 0.1 + 0.9 * np.random.rand()
                if np.random.rand() < 0.1:
                    CR[i] = np.random.rand()

                # Mutation: Current-to-pbest/1 approximation
                # We select the best in the current population
                best_idx = np.argmin(pop_y)
                candidates = [idx for idx in range(pop_size) if idx != i]
                if len(candidates) < 2:
                    r1, r2 = i, best_idx
                else:
                    r1, r2 = np.random.choice(candidates, 2, replace=False)

                mutant = pop[i] + F[i] * (pop[best_idx] - pop[i]) + F[i] * (pop[r1] - pop[r2])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])

                # Soft Boundary Handling
                out_ub = trial > ub
                out_lb = trial < lb
                if np.any(out_ub) or np.any(out_lb):
                    trial[out_ub] = ub[out_ub] - np.random.rand(np.sum(out_ub)) * (ub[out_ub] - lb[out_ub]) * 0.1
                    trial[out_lb] = lb[out_lb] + np.random.rand(np.sum(out_lb)) * (ub[out_lb] - lb[out_lb]) * 0.1

                # Selection
                val = self._safe_eval(trial, func, lb, ub)
                if val <= pop_y[i]:
                    pop[i] = trial
                    pop_y[i] = val

            gen += 1

        return self.best_x, self.best_y
