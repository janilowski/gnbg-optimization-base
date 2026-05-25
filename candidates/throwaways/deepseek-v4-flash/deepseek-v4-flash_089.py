import numpy as np
from math import exp, isfinite

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Simulated Annealing with adaptive step size and reannealing restarts.
#   Designed for continuous black-box minimization under a limited evaluation budget.
#   Handles arbitrary dimension by scaling step sizes to the search domain.
# Search state: A single current solution point (x_current) and its objective value.
#   Best-so-far (x_best, y_best) is tracked separately.
# Candidate generation: Add a Gaussian perturbation scaled by the current step size
#   (which is adapted online) to the current solution.
# Selection and replacement: Accept candidate if it has lower objective (greedy) or
#   with a probability based on the Metropolis criterion, where temperature follows
#   an exponential cooling schedule.
# Adaptation: Step size is adapted every L evaluations based on the acceptance rate
#   (target between 0.2 and 0.5). Temperature is reduced by a factor every L evaluations.
# Exploration mechanisms: High temperature early on accepts many bad moves; random
#   reannealing restarts when no improvement is seen for a long time introduce new
#   areas of the search space.
# Exploitation mechanisms: Low temperature later rejects most bad moves, focusing on
#   local refinement. Step size shrinks when acceptance rate is low, enabling finer search.
# Boundary handling: Candidate coordinates are clipped to the lower/upper bounds.
# Budget strategy: The algorithm uses exactly the given budget evaluation by evaluation,
#   stopping immediately when the budget is exhausted.
# Closest known influences: Classic simulated annealing (Kirkpatrick et al.) combined
#   with adaptive step size control (Corana et al.) and reannealing as used in ASA.
# Novelty or unusual aspects: A fixed cooling schedule with online step size adaptation
#   and reannealing triggered by stagnation; no complicated parameter tuning.
# Failure modes: On extremely rugged or high-dimensional landscapes with very limited
#   budget, the algorithm may converge prematurely to a poor local optimum because it
#   only maintains a single candidate. The reannealing attempts mitigate this but are
#   not guaranteed to escape deep local minima.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """Initialize the optimizer with a fixed evaluation budget and dimension."""
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """Run the optimization on the given black‑box function.

        Parameters
        ----------
        func : callable
            The objective function to minimize. It must provide bounds via
            ``func.lower`` / ``func.upper`` or ``func.bounds.lb`` / ``func.bounds.ub``.

        Returns
        -------
        best_x : np.ndarray
            Best found point (dim,).
        best_y : float
            Objective value at best_x.
        """
        # ---- extract bounds -------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.atleast_1d(np.asarray(func.lower, dtype=float))
            upper = np.atleast_1d(np.asarray(func.upper, dtype=float))
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lower = np.atleast_1d(np.asarray(b.lb, dtype=float))
            upper = np.atleast_1d(np.asarray(b.ub, dtype=float))
        else:
            raise AttributeError("Function must provide lower/upper or bounds.lb/ub")
        lower = np.broadcast_to(lower, (self.dim,)).copy()
        upper = np.broadcast_to(upper, (self.dim,)).copy()
        width = upper - lower

        # ---- internal parameters --------------------------------------------
        # cooling schedule: temperature decays every L evaluations
        L = max(1, self.dim * 5)               # temperature update period
        initial_temp = 1.0                     # initial temperature
        temp = initial_temp
        cooling_factor = 0.95                  # exponential cooling per period
        # step size adaptation
        step = 0.2 * width                     # initial step per dimension
        step_min = 1e-12 * width               # minimum step to avoid collapse
        step_max = 0.5 * width                 # maximum step
        # acceptance rate tracking
        window_size = L
        accepted_count = 0
        eval_count = 0
        # stagnation detection
        patience = max(100, 10 * L)            # evaluations without improvement until reanneal
        no_improve_count = 0

        # ---- initial point --------------------------------------------------
        # random uniform in bounds
        x = lower + np.random.rand(self.dim) * width
        y = func(x)
        eval_count = 1

        best_x = x.copy()
        best_y = y
        current_x = x.copy()
        current_y = y

        # ---- main loop ------------------------------------------------------
        while eval_count < self.budget:
            # 1. generate candidate
            # Gaussian perturbation, scale by current step
            perturbation = np.random.randn(self.dim) * step
            cand = current_x + perturbation
            # clip to bounds
            cand = np.clip(cand, lower, upper)

            # 2. evaluate
            cand_y = func(cand)
            eval_count += 1

            # 3. acceptance?
            if cand_y < current_y:
                accepted = True
            else:
                # Metropolis probability
                delta = cand_y - current_y
                # avoid overflow for large negative delta / small temp
                if isfinite(temp) and temp > 0.0:
                    prob = exp(-delta / temp)
                else:
                    prob = 0.0
                accepted = np.random.rand() < prob

            if accepted:
                current_x = cand
                current_y = cand_y
                accepted_count += 1

            # 4. update best
            if current_y < best_y:
                best_x = current_x.copy()
                best_y = current_y
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 5. periodic adaptation (every L evaluations)
            if eval_count % L == 0 and eval_count > 0:
                # temperature cooling
                temp *= cooling_factor
                # step size adaptation based on acceptance rate over last L moves
                rate = accepted_count / window_size
                if rate < 0.2:
                    step *= 0.85
                elif rate > 0.5:
                    step *= 1.15
                # clamp step size
                step = np.clip(step, step_min, step_max)
                accepted_count = 0

            # 6. reannealing (restart) if no improvement for too long
            if no_improve_count >= patience:
                # reset current to best (or to a new random point with some perturbation)
                # here we reset to best and introduce a random jump from best
                jump = 0.5 * width * np.random.uniform(-1, 1, self.dim)
                current_x = np.clip(best_x + jump, lower, upper)
                current_y = func(current_x)
                eval_count += 1
                # reset temperature and step size
                temp = initial_temp
                step = 0.2 * width
                accepted_count = 0
                no_improve_count = 0
                # if best is still better, keep best
                if current_y < best_y:
                    best_x = current_x.copy()
                    best_y = current_y

            # safety exit if budget exhausted inside an iteration
            if eval_count >= self.budget:
                break

        return best_x, best_y
