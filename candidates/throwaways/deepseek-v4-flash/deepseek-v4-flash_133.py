# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1) Evolution Strategy with the classic 1/5 success rule for step-size adaptation.
#          It maintains a single parent solution, generates one offspring per iteration by adding
#          Gaussian noise, and replaces the parent only if the offspring has a better fitness.
#          The step size (sigma) is updated periodically based on the observed success rate to
#          balance exploration and exploitation.
# Search state: A single parent point in the search space and its current objective value.
# Candidate generation: Offspring = parent + sigma * N(0, I), where sigma is a scalar step size.
# Selection and replacement: Deterministic (1+1): the parent is replaced if the offspring yields
#          a strictly lower objective value (minimization). The best solution found is tracked
#          separately.
# Adaptation: Every 5 * dimension evaluations, the success rate over that period is computed.
#          If it exceeds 0.2, sigma is increased by a factor of 1.2; otherwise sigma is decreased
#          by a factor of 1.2. The success counter is reset.
# Exploration mechanisms: Gaussian mutation with step size that can grow when many successes occur.
# Exploitation mechanisms: Step size shrinks when few offspring are successful, focusing search
#          locally around the current parent.
# Boundary handling: Each candidate coordinate is clamped to the variable bounds after mutation.
# Budget strategy: The function is called until the evaluation budget is exhausted; the best
#          solution seen at any point is returned.
# Closest known influences: Rechenberg’s (1+1)-ES with the 1/5 rule.
# Novelty or unusual aspects: No novel mechanisms; this is a canonical textbook ES used for
#          didactic clarity and robustness across dimensions.
# Failure modes: May converge prematurely if the step size collapses too early; lacks population
#          diversity for multimodal or noisy landscapes. The simple 1/5 rule can be misled by
#          plateaus or deceptive gradients.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """(1+1)-ES with 1/5 success rule for black-box minimization."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ----- Extract bounds -----
        if hasattr(func, 'lower'):
            low = np.asarray(func.lower, dtype=float)
            high = np.asarray(func.upper, dtype=float)
        else:
            low = np.asarray(func.bounds.lb, dtype=float)
            high = np.asarray(func.bounds.ub, dtype=float)

        # ----- Initialisation -----
        dim = self.dim
        parent = np.random.uniform(low, high, dim)
        parent_y = func(parent)
        evals = 1

        best_x = parent.copy()
        best_y = parent_y

        # Step size: start at 20% of the average variable range
        sigma = 0.2 * np.mean(high - low)

        # Adaptation parameters
        adapt_period = max(5 * dim, 10)   # check success every this many evaluations
        success_counter = 0
        evals_since_last_adapt = 0

        # Main loop
        while evals < self.budget:
            # Generate offspring
            offspring = parent + sigma * np.random.randn(dim)
            # Clamp to bounds
            offspring = np.clip(offspring, low, high)

            offspring_y = func(offspring)
            evals += 1

            # Update best solution
            if offspring_y < best_y:
                best_x = offspring.copy()
                best_y = offspring_y

            # (1+1) selection: replace parent if offspring is strictly better
            if offspring_y < parent_y:
                parent = offspring
                parent_y = offspring_y
                success_counter += 1

            evals_since_last_adapt += 1

            # Adapt step size periodically
            if evals_since_last_adapt >= adapt_period:
                success_rate = success_counter / adapt_period
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma /= 1.2
                # Reset
                success_counter = 0
                evals_since_last_adapt = 0

        return best_x, best_y
