# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1)-Evolution Strategy with step-size adaptation via the 1/5th success rule. Designed for robustness across dimensions in the GNBG black-box minimization benchmark.
# Search state: Single best candidate (best_x) and its objective value (best_y). A scalar step size (sigma) controls the search radius; a success/failure history (recent successes) determines adaptation.
# Candidate generation: Gaussian perturbation: x_cand = best_x + sigma * N(0, I). Candidate is clamped to the box constraints.
# Selection and replacement: Greedy: if f(x_cand) < f(best_x), replace best_x and record a success; otherwise record a failure.
# Adaptation: After each evaluation, the success rate over a sliding window of length W (default 50) is computed. If rate > 0.2, sigma is increased by factor exp(0.2); if rate < 0.2, sigma is decreased by factor exp(-0.2). This implements the classic 1/5th rule.
# Exploration mechanisms: Large sigma early on due to initialization; adaptation increases sigma when too many successes (i.e., large steps are beneficial).
# Exploitation mechanisms: Small sigma when few successes occur; the deterministic greedy selection focuses on the best point.
# Boundary handling: Simple clamping to [lb, ub] per coordinate. This can cause efficiency loss near boundaries but is simple and robust.
# Budget strategy: Iterate until evaluation count reaches budget. No restarts or complex budget allocation.
# Closest known influences: (1+1)-ES with cumulative step-size adaptation (CSA) or the simpler 1/5th rule. The code is essentially a minimal implementation of the latter.
# Novelty or unusual aspects: None; the algorithm is intentionally standard and minimal. The main goal is correctness and clarity to serve as a baseline.
# Failure modes: Premature convergence on multimodal functions if sigma collapses. Slow progress on ill-conditioned or highly non-separable problems due to isotropic steps. May be outperformed by population-based methods.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Minimal (1+1)-Evolution Strategy with 1/5th rule step-size adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Window size for success rate (clamped to budget/10, at least 1)
        self.window = max(1, min(50, int(self.budget / 10)))
        self.history = np.zeros(self.window, dtype=bool)  # recent success flags
        self.hist_idx = 0

    def __call__(self, func):
        # --- Read bounds ---
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb, dtype=float)
                ub = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Cannot read bounds from the function object.")
        if lb.ndim == 0:  # scalar bounds – broadcast to dim
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)

        # --- Initialization ---
        best_x = lb + (ub - lb) * np.random.rand(self.dim)
        best_y = func(best_x)
        evals = 1

        # Step size: 20% of the average range (scalar, isotropic)
        avg_range = np.mean(ub - lb)
        sigma = 0.2 * avg_range
        sigma_min = 1e-10 * avg_range   # minimal step size for safety

        # Main loop (stop when evaluations used up)
        while evals < self.budget:
            # Generate candidate
            x_cand = best_x + sigma * np.random.randn(self.dim)
            # Clip to bounds
            x_cand = np.clip(x_cand, lb, ub)

            # Evaluate
            y_cand = func(x_cand)
            evals += 1

            # Success?
            success = y_cand < best_y
            self.history[self.hist_idx] = success
            self.hist_idx = (self.hist_idx + 1) % self.window
            if success:
                best_x = x_cand
                best_y = y_cand

            # Step-size adaptation (1/5th rule) after at least one window fill
            if evals >= self.window:
                success_rate = np.mean(self.history)
                if success_rate > 0.2:
                    sigma *= np.exp(0.2)
                elif success_rate < 0.2:
                    sigma *= np.exp(-0.2)
                # Enforce minimum step size to avoid complete stop
                sigma = max(sigma, sigma_min)

        return best_x, best_y
