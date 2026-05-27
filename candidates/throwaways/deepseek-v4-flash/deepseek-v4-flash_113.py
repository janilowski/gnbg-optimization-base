# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1)-Evolution Strategy with step-size adaptation using Rechenberg's 1/5 rule. It is designed for black-box minimization and works robustly across dimensions.
# Search state: A single current candidate point (parent) and its associated objective value. The step size sigma is a scalar that is adapted periodically.
# Candidate generation: Offspring is created by adding isotropic Gaussian noise scaled by sigma to the parent. The noise vector is drawn from np.random.normal(0,1,dim).
# Selection and replacement: The offspring replaces the parent if it yields lower (better) objective value (greedy selection). The best-so-far solution is updated accordingly.
# Adaptation: Every floor(budget / (20 * dim + 1)) generations, sigma is adjusted: if the success rate (fraction of offspring that improved) exceeds 1/5, sigma = sigma * 1.5; if below 1/5, sigma = sigma / 1.5. This follows Rechenberg's rule.
# Exploration mechanisms: The multiplicative step-size adaptation allows the algorithm to explore more when progress is easy and to shrink when fine-tuning is needed.
# Exploitation mechanisms: Greedy selection and the eventual reduction of sigma when the success rate is low concentrate search near the current best.
# Boundary handling: Offspring points that fall outside the box bounds are reflected inward (mirror reflection) to stay feasible. This avoids waste and preserves uniformity.
# Budget strategy: The algorithm evaluates exactly one new candidate per iteration, using the entire evaluation budget. The initial parent evaluation consumes one budget.
# Closest known influences: (1+1)-ES with Rechenberg's 1/5 rule is a classic mutation-based evolutionary strategy.
# Novelty or unusual aspects: The implementation is minimal and uses reflection boundary handling. The adaptation period depends on dimension and budget to ensure enough samples for reliable success rate estimation.
# Failure modes: May converge prematurely on multimodal landscapes if step size shrinks too fast or if the initial point is poor. The 1/5 rule can oscillate. Very high budgets relative to dimension might cause unnecessary fine-tuning.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """(1+1)-Evolution Strategy with step-size adaptation for black-box minimization.
    
    Public interface:
        __init__(self, budget: int, dim: int)
        __call__(self, func) -> tuple[np.ndarray, float]
            func provides .lower, .upper or .bounds.lb, .bounds.ub
    """
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot read bounds from func: missing .lower/.upper or .bounds.lb/.bounds.ub")

        dim = self.dim
        # Initial parent: uniform random inside bounds
        parent = lb + np.random.rand(dim) * (ub - lb)
        evals = 1
        best_x = parent.copy()
        best_y = func(parent)

        # Step size: initial value = 10% of the diagonal length
        diagonal = np.sqrt(np.sum((ub - lb)**2))
        sigma = 0.1 * diagonal if diagonal > 0 else 1.0

        # Adaptation parameters
        # Period: we want enough generations to estimate success rate reliably.
        # We use floor(budget / (20 * dim + 1)) but at least 10.
        period = max(10, self.budget // (20 * dim + 1))
        gen_counter = 0
        success_counter = 0

        # Main loop
        while evals < self.budget:
            # Generate offspring by adding Gaussian noise scaled by sigma
            offspring = parent + sigma * np.random.randn(dim)

            # Reflect boundaries (mirror reflection)
            # For each coordinate, if outside [lb_i, ub_i], reflect inward.
            # Reflection: if x < lb: new = lb + (lb - x); if x > ub: new = ub - (x - ub)
            # This is equivalent to: offset = x - lb; if offset < 0: offset = -offset;
            # and similarly for upper bound, but handled coordinate-wise.
            # Simpler: use np.clip and then reflect? We'll do explicit reflection loop.
            for i in range(dim):
                if offspring[i] < lb[i]:
                    offspring[i] = lb[i] + (lb[i] - offspring[i])
                    # If still out (possible when lower bound > upper bound), clip
                    if offspring[i] > ub[i]:
                        offspring[i] = lb[i]  # fallback to lower bound
                elif offspring[i] > ub[i]:
                    offspring[i] = ub[i] - (offspring[i] - ub[i])
                    if offspring[i] < lb[i]:
                        offspring[i] = ub[i]  # fallback to upper bound

            # Evaluate offspring
            y_off = func(offspring)
            evals += 1
            gen_counter += 1

            # Selection and best update
            if y_off < best_y:
                best_y = y_off
                best_x = offspring.copy()
                success_counter += 1
                # Replace parent (greedy) only if offspring is better
                parent = offspring
            # else parent remains unchanged

            # Step-size adaptation at the end of a period
            if gen_counter >= period:
                success_rate = success_counter / period if period > 0 else 0.0
                if success_rate > 0.2:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma /= 1.5
                # Reset counters
                gen_counter = 0
                success_counter = 0

            # Ensure sigma doesn't shrink too much (optional safety)
            sigma = max(sigma, 1e-10 * diagonal)

        return best_x, best_y
