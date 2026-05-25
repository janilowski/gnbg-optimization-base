import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: The algorithm is an adaptive random search that performs hill‑climbing with stochastic perturbations and periodic restarts. It maintains a current solution and a scalar step‑size (sigma) that scales the allowed exploration range. The step‑size is adapted online based on success/failure and the algorithm restarts when the step‑size becomes negligible, ensuring continued exploration while never exceeding the evaluation budget.
#
# Search state: best solution (best_x, best_y) seen so far, current solution (current_x), and a step‑size factor (sigma_factor) that multiplies the bound range.
#
# Candidate generation: A candidate is formed by adding Gaussian noise to the current solution; the noise magnitude is sigma_factor * (upper_bound - lower_bound) per dimension. The candidate is clipped back into the feasible region.
#
# Selection and replacement: After evaluating the candidate, if it improves the best known value the algorithm updates best_x, best_y and the current solution; otherwise the current solution stays unchanged.
#
# Adaptation: After each evaluation the step‑size factor is increased by 20% on success and decreased by 30% on failure, bounded between a minimum (1e‑6) and a maximum (1.0). When the factor hits the minimum the algorithm performs a restart: a new random point is sampled, evaluated, and the step‑size factor is reset to its initial value (0.5).
#
# Exploration mechanisms: Random perturbations around the current point provide exploration; periodic restarts ensure the search does not get trapped when the step‑size becomes too small.
#
# Exploitation mechanisms: By always moving to a better candidate and adjusting the step‑size, the algorithm concentrates on promising regions while retaining some exploratory ability.
#
# Boundary handling: All points are clipped to the problem's lower/upper bounds, guaranteeing feasibility.
#
# Budget strategy: A counter tracks evaluations; the main loop terminates as soon as the counter reaches the supplied budget, guaranteeing never to exceed it.
#
# Closest known influences: The method is similar to a (1+1) evolution strategy with the 1/5 rule for step‑size adaptation and to classic adaptive random search algorithms.
#
# Novelty or unusual aspects: Deliberately minimal – only a scalar step‑size and Gaussian perturbations – making it easy to understand and robust across a wide range of dimensions.
#
# Failure modes: In highly multi‑modal landscapes the simple hill‑climbing may become stuck in local minima; restarts mitigate this but do not guarantee global optimality.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    """
    Simple adaptive random search for black‑box minimization.

    The algorithm maintains a current solution and a step‑size factor.
    Perturbations are drawn from a normal distribution scaled by the factor
    and the problem bounds. The step‑size is adapted online and the search
    restarts when the step‑size becomes negligible.
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the decision space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the adaptive random search on the provided objective function.

        Parameters
        ----------
        func : callable
            A black‑box objective function that accepts a numpy array
            (single candidate) and returns a scalar (objective value).

        Returns
        -------
        best_x : numpy.ndarray
            The best (lowest) solution found.
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------------
        # Retrieve problem bounds (support both attribute styles)
        # ------------------------------------------------------------------
        try:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        except AttributeError:
            # Assume func has a `bounds` attribute with .lb and .ub
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)

        # Ensure they are numpy arrays of correct shape
        lb = np.atleast_1d(lb)
        ub = np.atleast_1d(ub)
        # If the problem supplies scalar bounds, broadcast to dim
        if lb.shape[0] == 1:
            lb = np.tile(lb, self.dim)
        if ub.shape[0] == 1:
            ub = np.tile(ub, self.dim)

        bound_range = ub - lb

        # ------------------------------------------------------------------
        # Helper: generate a random initial point inside the bounds
        # ------------------------------------------------------------------
        def random_point():
            return lb + np.random.rand(self.dim) * bound_range

        # ------------------------------------------------------------------
        # Initialisation
        # ------------------------------------------------------------------
        evals = 0
        best_x = None
        best_y = np.inf

        # First evaluation – if budget is zero we skip the loop
        if self.budget > 0:
            start_x = random_point()
            start_y = func(start_x)
            evals += 1
            best_x = start_x.copy()
            best_y = start_y
            current_x = best_x.copy()
        else:
            # No evaluations allowed – return whatever we have (None)
            return best_x, best_y

        # ------------------------------------------------------------------
        # Adaptive random search main loop
        # ------------------------------------------------------------------
        # Initial step‑size factor (relative to the bound range)
        sigma_factor = 0.5
        min_sigma_factor = 1e-6
        max_sigma_factor = 1.0
        success_mult = 1.2      # factor on successful improvement
        failure_mult = 0.7      # factor on failure

        while evals < self.budget:
            # -------------------------------------------------
            # Generate a candidate by perturbing current_x
            # -------------------------------------------------
            sigma = sigma_factor * bound_range          # element‑wise scaling
            noise = np.random.randn(self.dim)
            candidate = current_x + sigma * noise

            # Clip to bounds to keep feasible
            candidate = np.clip(candidate, lb, ub)

            # -------------------------------------------------
            # Evaluate candidate
            # -------------------------------------------------
            cand_y = func(candidate)
            evals += 1

            # -------------------------------------------------
            # Selection & step‑size adaptation
            # -------------------------------------------------
            if cand_y < best_y:
                # Improvement found – accept candidate
                best_y = cand_y
                best_x = candidate.copy()
                current_x = candidate.copy()
                sigma_factor = min(max_sigma_factor, sigma_factor * success_mult)
            else:
                # No improvement – shrink step‑size
                sigma_factor = max(min_sigma_factor, sigma_factor * failure_mult)

            # -------------------------------------------------
            # Restart when step‑size becomes too small
            # -------------------------------------------------
            if sigma_factor <= min_sigma_factor:
                # Restart: draw a new random point, evaluate, and reset sigma_factor
                # Make sure we still have budget for this evaluation
                if evals >= self.budget:
                    break
                new_point = random_point()
                new_y = func(new_point)
                evals += 1

                # Update best if new point is better
                if new_y < best_y:
                    best_y = new_y
                    best_x = new_point.copy()

                # Reset current point and step‑size factor
                current_x = new_point.copy()
                sigma_factor = 0.5   # restore initial factor

        return best_x, best_y
