import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (1+1)-Evolution Strategy with isotropic Gaussian mutation
#          and step-size adaptation using Rechenberg's 1/5 success rule.
# Search state: A single parent candidate (vector) and its objective value.
# Candidate generation: Parent is perturbed by zero-mean Gaussian noise scaled
#   by the current step size and the per‑dimension domain width.
# Selection and replacement: Deterministic elitist replacement: the child
#   replaces the parent only if it yields a strictly better objective value.
# Adaptation: Step size is adjusted after every generation based on the success
#   frequency over a sliding window (last max(10, dim) trials). If the
#   success rate exceeds 0.2, sigma is increased (×1.2); if below 0.2, sigma
#   is decreased (×0.8).
# Exploration mechanisms: The global step size adapts to the landscape,
#   enabling wider exploration when many improvements are found.
# Exploitation mechanisms: Only improving moves are accepted; the parent
#   always moves towards better regions.
# Boundary handling: Generated points are clipped to the lower and upper bounds.
# Budget strategy: Every call to func() is counted. The loop stops as soon as
#   the full budget is consumed. No evaluations are wasted.
# Closest known influences: Classic (1+1)-ES with 1/5 success rule (Rechenberg).
# Novelty or unusual aspects: None. Straightforward textbook implementation.
# Failure modes: Can stagnate in local optima when step size collapses;
#   no restart mechanism; performance degrades on highly multimodal functions
#   or when the optimum lies on the boundary.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the (1+1)-ES minimizer on the given black-box function.

        Parameters
        ----------
        func : callable
            Objective function with attributes giving the domain bounds.
            Expected to provide either:
                func.bounds.lb, func.bounds.ub  (arrays)
            or
                func.lower, func.upper           (arrays)

        Returns
        -------
        best_x : np.ndarray
            Best point found.
        best_y : float
            Objective value at best_x.
        """
        # -------------------- Read domain bounds --------------------
        if hasattr(func, 'bounds'):
            # Coco / BBOB style
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            # Simple interface
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)

        span = ub - lb
        # Handle possible zero-length dimensions (fallback to 1.0)
        span[span < 1e-12] = 1.0

        # -------------------- Initialisation --------------------
        rng = np.random.default_rng()
        # Start step size as a fraction of the mean domain span
        sigma = 0.2 * np.mean(span)

        # First candidate (uniform random inside bounds)
        parent = lb + rng.random(self.dim) * span
        evals = 0
        parent_y = func(parent)
        evals += 1

        best_x = parent.copy()
        best_y = parent_y

        # Sliding window for success rate
        window_size = max(10, self.dim)
        successes = []  # list of 0/1 for recent generations

        # -------------------- Main optimization loop --------------------
        while evals < self.budget:
            # Generate child by isotropic Gaussian mutation
            child = parent + sigma * rng.normal(0, 1, self.dim) * span
            # Enforce bounds
            child = np.clip(child, lb, ub)

            # Evaluate
            child_y = func(child)
            evals += 1

            # Keep track of global best
            if child_y < best_y:
                best_x = child.copy()
                best_y = child_y

            # Selection: replace parent if child is better
            if child_y < parent_y:
                parent = child
                parent_y = child_y
                successes.append(1)
            else:
                successes.append(0)

            # Maintain window size
            if len(successes) > window_size:
                successes.pop(0)

            # Adapt step size after window is filled
            if len(successes) == window_size:
                success_rate = np.mean(successes)
                # Rechenberg's 1/5 rule
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                # Keep sigma in a reasonable range to avoid collapse
                sigma = max(1e-8, min(1.0, sigma))

        return best_x, best_y
