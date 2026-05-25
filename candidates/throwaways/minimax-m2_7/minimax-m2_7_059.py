# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A minimalist (1+1)-evolution strategy that performs a single‑candidate hill‑climb with a step‑size that adapts using the classic 1/5 success rule.
# Search state: The algorithm keeps the current candidate solution and the best solution observed so far.
# Candidate generation: A new point is created by adding a Gaussian perturbation (scaled by a step‑size sigma) to the current point.
# Selection and replacement: If the perturbed point yields a lower objective value it becomes the new current point; otherwise the current point is retained.
# Adaptation: After each trial sigma is multiplied by a success‑factor (>1) on improvement and by a failure‑factor (<1) otherwise. Additionally, a sliding window of recent trials implements the 1/5 rule: if the success ratio exceeds 20 % sigma is enlarged, otherwise it is shrunk.
# Exploration mechanisms: The search starts from a uniformly random point covering the whole feasible region, and large sigma values early on explore broadly.
# Exploitation mechanisms: As sigma shrinks, the algorithm narrows its focus to fine‑grained refinement around the current solution.
# Boundary handling: After perturbation each coordinate is clipped to the feasible box (component‑wise min/max). This simple truncation prevents any boundary violations.
# Budget strategy: The algorithm counts each call to the objective function and stops immediately when the evaluation counter reaches the supplied budget, guaranteeing the budget is never exceeded.
# Closest known influences: The classic (1+1)-ES with Rechenberg’s 1/5 step‑size adaptation.
# Novelty or unusual aspects: Implementation is deliberately small, relying only on NumPy and the Python standard library, making it easy to embed in any benchmark harness.
# Failure modes: Because only a single solution is maintained, the method can become trapped in local minima on highly multi‑modal landscapes. It also does not handle noisy objectives.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple (1+1)-evolution strategy with a 1/5 success rule for step‑size adaptation.
    Designed to be compact, readable, and to respect the evaluation budget supplied
    by the benchmark harness.
    """

    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the decision space (number of variables).
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func) -> tuple:
        """
        Run the optimisation.

        Parameters
        ----------
        func : callable
            Black‑box objective function. Must support:
                - `func(x)` where x is a NumPy array (returns a scalar),
                - Either attributes `func.lower` / `func.upper` or
                  `func.bounds.lb` / `func.bounds.ub` that define the box constraints.

        Returns
        -------
        best_x : np.ndarray
            Best (lowest‑valued) decision vector found.
        best_y : float
            Corresponding objective value.
        """
        # ------------------------------------------------------------------
        # 1. Retrieve box constraints (lower and upper bounds).
        # ------------------------------------------------------------------
        try:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        except AttributeError:
            # Assume a `bounds` attribute with `lb` and `ub`.
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)

        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            raise ValueError(
                f"Dimension mismatch: expected bounds of length {self.dim}, "
                f"got {lb.shape} and {ub.shape}"
            )

        # ------------------------------------------------------------------
        # 2. Guard against zero or negative budgets.
        # ------------------------------------------------------------------
        if self.budget < 1:
            # No evaluations allowed – return a placeholder.
            return None, None

        # ------------------------------------------------------------------
        # 3. Initialise search: random start inside the box.
        # ------------------------------------------------------------------
        range_vals = ub - lb
        sigma = 0.2 * range_vals.mean()  # initial step size (scalar)

        x = lb + np.random.rand(self.dim) * range_vals
        y = func(x)
        evals = 1

        best_x = x.copy()
        best_y = y

        # ------------------------------------------------------------------
        # 4. Adaptation parameters for step‑size control.
        # ------------------------------------------------------------------
        inc_factor = 1.2    # multiply sigma on success
        dec_factor = 0.8    # multiply sigma on failure
        window_size = 10 + int(2 * self.dim)  # sliding window for 1/5 rule

        successes = 0
        failures = 0

        # ------------------------------------------------------------------
        # 5. Main optimisation loop (1+1)-ES with 1/5 rule.
        # ------------------------------------------------------------------
        while evals < self.budget:
            # --- Candidate generation ------------------------------------
            # Perturb current point with isotropic Gaussian noise.
            x_cand = x + sigma * np.random.randn(self.dim)

            # --- Boundary handling (clipping) ----------------------------
            x_cand = np.clip(x_cand, lb, ub)

            # --- Evaluate candidate --------------------------------------
            y_cand = func(x_cand)
            evals += 1

            # --- Update best known solution -----------------------------
            if y_cand < best_y:
                best_x = x_cand.copy()
                best_y = y_cand

            # --- Acceptance decision ------------------------------------
            if y_cand < y:
                # Success: move to the candidate.
                x, y = x_cand, y_cand
                successes += 1
                sigma *= inc_factor
            else:
                failures += 1
                sigma *= dec_factor

            # --- 1/5 rule over sliding window ----------------------------
            if successes + failures >= window_size:
                success_ratio = successes / (successes + failures)
                if success_ratio > 0.2:
                    sigma *= inc_factor
                elif success_ratio < 0.2:
                    sigma *= dec_factor
                # Reset counters for the next window.
                successes = 0
                failures = 0

            # Prevent sigma from becoming negligibly small.
            sigma = max(sigma, 1e-6 * range_vals.mean())

        # ------------------------------------------------------------------
        # 6. Return best solution found.
        # ------------------------------------------------------------------
        return best_x, best_y
