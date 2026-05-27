import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1)-Evolution Strategy with Rechenberg's 1/5 rule for step size adaptation.
# Search state: A single current point (best_x) and its objective value (best_y). Step size sigma is adapted.
# Candidate generation: Add isotropic Gaussian noise scaled by sigma to the current point.
# Selection and replacement: If the candidate is strictly better (lower objective) than the current point, it replaces it.
# Adaptation: The step size sigma is adjusted every adaptation window (10 evaluations) based on the success rate of offspring. A success rate > 1/5 increases sigma; < 1/5 decreases sigma.
# Exploration mechanisms: The Gaussian mutation provides exploration. A larger sigma when success rate is high encourages broader search.
# Exploitation mechanisms: When success rate is low, sigma shrinks, focusing search near the current best.
# Boundary handling: Candidates are reflected back into the feasible domain after mutation.
# Budget strategy: The loop explicitly counts evaluations and stops before exceeding the budget.
# Closest known influences: Classic (1+1)-ES with Rechenberg's 1/5 rule, as described in Schwefel (1995).
# Novelty or unusual aspects: Uses a fixed adaptation window rather than cumulative step-size adaptation; designed for simplicity and robustness.
# Failure modes: May converge prematurely to local optima if the initial step size is inappropriate or if the function is highly multimodal. Restarts are not implemented, so performance can degrade on pathological landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from function interface
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lo = np.asarray(func.lower, dtype=float)
            hi = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lo = np.asarray(func.bounds.lb, dtype=float)
            hi = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot read bounds from func")

        # Ensure bounds are 1D arrays
        lo = np.ravel(lo)
        hi = np.ravel(hi)
        dim = self.dim
        if dim != lo.size:
            # Dimension mismatch: trust func? But we use self.dim
            # If sizes differ, use the smaller and extend?
            # Keep it simple: assume they match
            pass

        # Initialise current point uniformly in bounds
        best_x = lo + np.random.rand(dim) * (hi - lo)
        best_y = func(best_x)
        evals = 1

        # Step size initialisation: 0.2 * typical range
        sigma = 0.2 * (hi - lo).mean()
        if sigma == 0.0:
            sigma = 0.1  # fallback for constant bounds

        # Parameters for Rechenberg's rule
        adapt_interval = 10
        success_count = 0
        total_count = 0

        while evals < self.budget:
            # Candidate generation with Gaussian mutation
            candidate = best_x + sigma * np.random.randn(dim)

            # Boundary handling: reflect inbound
            # Reflect for each coordinate
            for i in range(dim):
                while candidate[i] < lo[i] or candidate[i] > hi[i]:
                    if candidate[i] < lo[i]:
                        candidate[i] = lo[i] + (lo[i] - candidate[i])
                    if candidate[i] > hi[i]:
                        candidate[i] = hi[i] - (candidate[i] - hi[i])

            # Evaluate
            cand_y = func(candidate)
            evals += 1

            # Selection
            if cand_y < best_y:
                best_x, best_y = candidate, cand_y
                success_count += 1
            total_count += 1

            # Adaptation every adapt_interval evaluations
            if total_count == adapt_interval:
                success_rate = success_count / adapt_interval
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                # else unchanged
                success_count = 0
                total_count = 0

            # Protect against sigma collapse
            if sigma < 1e-12:
                sigma = 0.2 * (hi - lo).mean()

        return best_x, best_y
