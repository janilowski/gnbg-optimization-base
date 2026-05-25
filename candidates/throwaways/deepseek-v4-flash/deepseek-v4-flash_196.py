# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1) Evolution Strategy with the 1/5th success rule for step-size adaptation, designed for continuous black-box minimization. It uses a single parent and generates one offspring per iteration via isotropic Gaussian mutation.
# Search state: A single solution vector (parent) and a global step size sigma.
# Candidate generation: Offspring = parent + sigma * N(0,I). Vector is clipped to bounds.
# Selection and replacement: Greedy (elitist): offspring replaces parent only if it has lower objective value.
# Adaptation: Step size sigma is adapted using the 1/5th success rule based on a sliding window of recent trials. If the success rate exceeds 1/5, sigma is increased; if below, decreased.
# Exploration mechanisms: Large sigma encourages exploration; adaptation can increase sigma when many successes occur, allowing escape from local optima.
# Exploitation mechanisms: Small sigma fine-tunes near promising regions. The greedy selection ensures convergence.
# Boundary handling: Clipping to lower and upper bounds.
# Budget strategy: Evaluates exactly budget-1 candidates after initial evaluation; stops when budget is exhausted.
# Closest known influences: Traditional (1+1)-ES with Rechenberg's 1/5 rule.
# Novelty or unusual aspects: None; a straightforward implementation.
# Failure modes: May get stuck in local optima on highly multimodal landscapes or converge prematurely if sigma collapses. Performance sensitive to initial sigma and adaptation parameters.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # --- Read bounds from the function object ---
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds: need func.lower/upper or func.bounds.lb/ub")

        # Ensure bounds are 1D arrays of correct length
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)
        else:
            lb = lb.flatten()
            ub = ub.flatten()

        # --- Initialisation ---
        parent = lb + np.random.rand(self.dim) * (ub - lb)
        parent_y = func(parent)
        evals = 1

        best_x = parent.copy()
        best_y = parent_y

        # Step size: start with 20% of the mean axis range
        range_scale = np.mean(ub - lb)
        sigma = 0.2 * range_scale

        # --- Parameters for 1/5 rule ---
        window_size = max(10, self.dim)   # length of success history
        success_history = []              # list of booleans
        sig_min = 1e-10 * range_scale
        sig_max = 1e+2 * range_scale

        # --- Main loop ---
        while evals < self.budget:
            # Generate candidate via Gaussian mutation
            candidate = parent + sigma * np.random.randn(self.dim)
            # Boundary clipping
            candidate = np.clip(candidate, lb, ub)

            # Evaluate
            candidate_y = func(candidate)
            evals += 1

            # Selection (greedy / elitist)
            success = candidate_y < parent_y
            if success:
                parent = candidate
                parent_y = candidate_y
                if candidate_y < best_y:
                    best_x = candidate.copy()
                    best_y = candidate_y

            # Update success history
            success_history.append(success)
            if len(success_history) > window_size:
                success_history.pop(0)

            # Adapt sigma using 1/5 rule over the window
            if len(success_history) == window_size:
                succ_rate = np.mean(success_history)
                if succ_rate > 0.2:
                    sigma *= 1.2
                elif succ_rate < 0.2:
                    sigma *= 0.85
                # Clamp sigma to avoid extreme values
                sigma = np.clip(sigma, sig_min, sig_max)

        return best_x, best_y
