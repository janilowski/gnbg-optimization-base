import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a simple evolution strategy with isotropic Gaussian mutations and step-size adaptation using the 1/5 success rule. The algorithm maintains a single parent mean and adapts step size to balance exploration and exploitation.
# Search state: mean vector (current best point after selection), step size sigma, number of successful offspring in last generation, and overall best-so-far solution.
# Candidate generation: Each generation, lambda offspring are sampled from N(mean, sigma^2 I). Generated points are clipped to bounds.
# Selection and replacement: The mu best offspring (mu = floor(lambda/2)) are used to compute the new mean via arithmetic average (equal weights). If no offspring improves over the previous mean, the mean remains unchanged? Typically we update mean to weighted average of mu best, regardless. So that's what we do: new mean = average of mu best candidates.
# Adaptation: Step size sigma is adjusted after each generation based on the success rate (fraction of offspring that improved over the previous mean's fitness). If success rate > 0.2, sigma is increased (multiply by 1.1); if < 0.2, sigma is decreased (multiply by 0.9). Sigma is clamped to a minimum (1e-10) and maximum (half the average range) to prevent extinction or too large jumps.
# Exploration mechanisms: Offspring are generated with isotropic Gaussian mutations, providing exploratory sampling. The step size adaptation ensures that exploration scale adjusts to the landscape.
# Exploitation mechanisms: The mean is updated toward the best mu candidates, focusing search in promising regions. The best solution seen so far is tracked and returned.
# Boundary handling: New candidate points are clipped component-wise to [lb, ub].
# Budget strategy: The algorithm operates in generations; the population size is set as lambda = max(4, 4+3*log(dim)). The number of generations is determined by budget // lambda. If budget is not exhausted, remaining evaluations are used for additional local search around the best (or just fill with random perturbations). I'll add a final local search if budget left: generate individual random perturbations and update best.
# Closest known influences: This algorithm is a simple (mu, lambda) evolution strategy with step-size adaptation via the 1/5 rule, as described in Rechenberg's work. It's essentially a basic ES without recombination.
# Novelty or unusual aspects: None; simple and well-known baseline.
# Failure modes: May get stuck in local optima if step size decays too quickly or if initial mean is far from optimum; may not converge precisely due to isotropic steps; may perform poorly on highly non-separable or ill-conditioned functions.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.array(b.lb, dtype=float)
            ub = np.array(b.ub, dtype=float)
        else:
            raise ValueError("Function must provide .lower/.upper or .bounds.lb/.bounds.ub")

        # Ensure lb and ub are 1D arrays of dimension dim
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)

        # Initialization
        dim = self.dim
        initial_range = ub - lb
        # Start mean at a random point inside bounds
        mean = lb + np.random.rand(dim) * initial_range
        # Initial step size: 20% of average range, scaled by 1/sqrt(dim) to account for isotropic noise
        sigma = 0.2 * np.mean(initial_range) / np.sqrt(dim)
        sigma_min = 1e-10
        sigma_max = 0.5 * np.mean(initial_range)

        # Population size: simple scaling with dimension
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2  # number of parents

        # Track best solution
        best_y = func(mean)  # first evaluation
        evaluations = 1
        best_x = mean.copy()

        # Main loop: generations
        while evaluations < self.budget:
            # Determine how many offspring we can produce this generation
            pop_size = min(lambda_, self.budget - evaluations)
            if pop_size < 1:
                break

            # Candidate generation
            candidates = np.zeros((pop_size, dim))
            for i in range(pop_size):
                candidate = mean + np.random.randn(dim) * sigma
                # Clip to bounds
                candidate = np.clip(candidate, lb, ub)
                candidates[i] = candidate

            # Evaluate
            values = np.array([func(c) for c in candidates])
            evaluations += pop_size

            # Update best overall
            best_idx = np.argmin(values)
            if values[best_idx] < best_y:
                best_y = values[best_idx]
                best_x = candidates[best_idx].copy()

            # Selection: choose the mu best offspring
            if mu > 0 and pop_size >= mu:
                sorted_indices = np.argsort(values)
                selected = candidates[sorted_indices[:mu]]
                # New mean = average of selected parents
                new_mean = np.mean(selected, axis=0)
            else:
                # Fallback: keep the best candidate as new mean
                new_mean = candidates[best_idx].copy() if pop_size > 0 else mean.copy()

            # Count successes: offspring that are better than the previous mean's fitness
            # (compare to previous mean's fitness, which we can compute or approximate)
            # To avoid extra evaluation, we use the value of the best candidate as reference.
            # A "success" is an offspring that is better than the current best (mean) value.
            # However the mean might not have been evaluated exactly. We'll compare to best_y
            # but that is a strict criterion. Alternatively, compare to the median or the previous mean's
            # value. We'll compute the previous mean's value exactly if not too costly.
            # Simpler: use success definition: offspring value < previous mean's value.
            # We can compute previous mean value if we haven't evaluated it? We'll evaluate it.
            # To avoid extra eval, we'll use the best from previous generation as proxy.
            # Robust approach: re-evaluate the mean with a spare evaluation if budget allows.
            # We'll just keep it simple: success if value < best_y (which is the best ever seen).
            # This is a bit harsh but still works.
            # Alternatively, we can compute success by comparing to the best candidate of previous generation,
            # but we don't have that stored. I'll use a different approach: after selection, compute
            # the new mean and then set success count based on improvement over the old mean's fitness.
            # Compute old mean's fitness once per generation (cost 1 eval). This is acceptable.
            # Here's a better scheme: evaluate old mean at start of generation (or after update).
            old_mean = mean.copy()
            # Evaluate old mean only once per generation; we already have its fitness from first time? not.
            # We'll do: if first generation, we have it; otherwise we need to compute.
            # Let's just track previous_mean_fitness.
            if not hasattr(self, '_prev_mean_fit'):
                self._prev_mean_fit = func(old_mean)
                evaluations += 1
            # Count successes: offspring with value < previous_mean_fitness
            successes = np.sum(values < self._prev_mean_fit)
            success_rate = successes / pop_size if pop_size > 0 else 0.0

            # Update mean
            mean = new_mean
            # Evaluate new mean for next generation
            # (but we might have already evaluated it if it came from an offspring? We'll re-evaluate to be consistent)
            if evaluations < self.budget:
                mean_fit = func(mean)
                evaluations += 1
                if mean_fit < best_y:
                    best_y = mean_fit
                    best_x = mean.copy()
                self._prev_mean_fit = mean_fit
            else:
                # No budget left for mean evaluation; just use the old prev_mean_fit
                pass

            # Step size adaptation: 1/5 rule
            target = 0.2
            if success_rate > target:
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, sigma_min, sigma_max)

        # Final fallback: if we still have budget, perform extra local perturbations
        while evaluations < self.budget:
            candidate = best_x + np.random.randn(dim) * sigma * 0.1
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            evaluations += 1
            if val < best_y:
                best_y = val
                best_x = candidate.copy()

        return best_x, best_y
