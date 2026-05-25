import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1)-Evolution Strategy with isotropic Gaussian mutations and cumulative step-size adaptation (1/5 success rule). It is designed for black-box minimization of functions with box constraints, working well in low to moderate dimensions.
# Search state: Single best solution (x_best) and its objective value (best_y). A scalar step size controls mutation amplitude.
# Candidate generation: Add isotropic Gaussian noise scaled by step size to x_best, then clip to the feasible box.
# Selection and replacement: Greedy – the new candidate replaces x_best if its objective value is strictly lower; otherwise it is discarded.
# Adaptation: Step size is updated after each generation using a simple exponential rule: on success multiply by exp(1/dim), on failure multiply by exp(-0.5/dim). This implements the 1/5 success rule implicitly and maintains a reasonable success rate.
# Exploration mechanisms: Isotropic Gaussian mutations with dynamic step size allow both local refinement and occasional larger jumps.
# Exploitation mechanisms: The greedy selection focuses the search around the best solution found so far. As the step size shrinks on repeated failures, the search becomes increasingly local.
# Boundary handling: Candidate coordinates are clipped to the box bounds (lower, upper) after mutation. No reflection or projection is used.
# Budget strategy: The algorithm uses exactly one function evaluation per iteration (the initial evaluation plus the loop). It stops as soon as the budget is exhausted, without any restarts or extra evaluations.
# Closest known influences: (1+1)-ES with 1/5 success rule (Rechenberg, 1973; Schwefel, 1981).
# Novelty or unusual aspects: None – the implementation is a straightforward textbook variant. The only adjustment is a per‑generation exponential update rather than a window‑based success rate.
# Failure modes: On very high‑dimensional (e.g., >50) or extremely rugged landscapes the single isotropic step size may adapt too slowly or get stuck. Also, because no restarts are used, the method can converge prematurely to a local optimum.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Minimization of a black-box function using a (1+1)-Evolution Strategy."""
    
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from either func.lower/upper or func.bounds.lb/ub
        try:
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lower = np.asarray(func.bounds.lb, dtype=float)
                upper = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Function must provide bounds via .lower/.upper or .bounds.lb/.bounds.ub")
        if lower.shape != (self.dim,) or upper.shape != (self.dim,):
            raise ValueError("Bounds must have shape (dim,)")

        # Initialise
        rng = np.random.default_rng()
        x_best = rng.uniform(lower, upper)
        best_y = func(x_best)
        evaluations = 1

        # Mean range for initial step size (10% of average range)
        mean_range = np.mean(upper - lower)
        step_size = 0.1 * mean_range
        # Prevent step from becoming too small
        min_step = 1e-10 * mean_range

        while evaluations < self.budget:
            # Generate candidate
            z = rng.normal(0.0, step_size, size=self.dim)
            x_candidate = x_best + z
            # Clip to bounds
            x_candidate = np.clip(x_candidate, lower, upper)

            y_candidate = func(x_candidate)
            evaluations += 1

            # Greedy selection
            if y_candidate < best_y:
                x_best = x_candidate
                best_y = y_candidate
                success = True
            else:
                success = False

            # Step size adaptation (1/5 rule)
            if success:
                step_size *= np.exp(1.0 / self.dim)
            else:
                step_size *= np.exp(-0.5 / self.dim)
            if step_size < min_step:
                step_size = min_step

        return x_best, best_y
