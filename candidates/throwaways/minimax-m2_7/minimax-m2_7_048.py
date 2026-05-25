# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (1+1)-evolution strategy (hill climbing) with adaptive step size. The method starts by sampling a few random points to obtain an initial best solution, then iteratively proposes new candidates by adding a Gaussian perturbation scaled by sigma. sigma is adapted based on success (1/5th rule).
# Search state: Keeps track of the current best solution (best_x) and its fitness (best_y), as well as the current step size sigma.
# Candidate generation: New candidate = best_x + sigma * N(0, I). sigma is adapted after each candidate evaluation.
# Selection and replacement: Greedy selection – if the new candidate's fitness is less than (or equal to) the current best, it replaces the best.
# Adaptation: sigma is multiplied by 1.2 on successful moves and by 0.9 on unsuccessful moves (1/5th rule). It is clamped to stay within a small minimum and not exceed the maximal bound range.
# Exploration mechanisms: Initial random sampling across the whole domain provides global exploration; large sigma early on encourages broad search.
# Exploitation mechanisms: After a good region is found, the algorithm fine‑tunes using smaller sigma values.
# Boundary handling: Candidates are clipped to the provided lower/upper bounds before evaluation.
# Budget strategy: Evaluates a small initial random batch (up to 10*dim points, limited by budget) to quickly locate a promising region; then performs as many iterative proposals as remaining budget allows.
# Closest known influences: Classic (1+1)-ES, simple hill‑climbing with adaptive step size, Covariance Matrix Adaptation (CMA-ES) inspiration in the sigma adaptation.
# Novelty or unusual aspects: The 1/5th rule is implemented in a lightweight way without maintaining a covariance matrix, making it very compact while still being adaptive.
# Failure modes: For highly multi‑modal or rugged landscapes the algorithm may converge to a local optimum; sigma may shrink to near‑zero if no improvements are found, reducing further exploration.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ------------------------------------------------------------------
        # Retrieve problem bounds (lower, upper). Accept either
        #   func.lower / func.upper  or  func.bounds.lb / func.bounds.ub.
        # Fall back to [-5,5]^dim if bounds are not exposed.
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Sensible default when bounds are not provided
            lower = np.full(self.dim, -5.0)
            upper = np.full(self.dim, 5.0)

        # --------------------------------------------------------------
        # Initial random sampling to get a reasonable starting point.
        # Sample at most 10*dim points, but never exceed the budget.
        # --------------------------------------------------------------
        init_size = min(self.budget, max(10 * self.dim, 1))
        eval_count = 0

        best_x = None
        best_y = np.inf

        for _ in range(init_size):
            x = np.random.uniform(lower, upper)          # uniform random point
            y = func(x)                                 # evaluate
            eval_count += 1
            if y < best_y:
                best_y = y
                best_x = x
                if eval_count >= self.budget:            # budget already exhausted
                    return best_x, best_y

        # --------------------------------------------------------------
        # Initial step size: a fraction of the overall domain width.
        # --------------------------------------------------------------
        sigma = np.max(upper - lower) * 0.2
        sigma = max(sigma, 1e-8)                         # avoid degenerate step

        # --------------------------------------------------------------
        # Iterative (1+1)-ES with 1/5‑th success rule.
        # --------------------------------------------------------------
        while eval_count < self.budget:
            # ---- generate candidate ----
            candidate = best_x + sigma * np.random.randn(self.dim)
            candidate = np.clip(candidate, lower, upper)  # stay inside bounds

            # ---- evaluate candidate ----
            y_candidate = func(candidate)
            eval_count += 1

            # ---- greedy selection ----
            if y_candidate <= best_y:
                best_x = candidate
                best_y = y_candidate
                success = True
            else:
                success = False

            # ---- adapt step size (1/5th rule) ----
            if success:
                sigma *= 1.2
            else:
                sigma *= 0.9

            # ---- keep sigma in a reasonable range ----
            sigma = np.clip(sigma, 1e-8, np.max(upper - lower))

        return best_x, best_y
