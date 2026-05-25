# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm for GNBG-style
# problems using a derivative-free evolution strategy with adaptive step size.
# Search state: Maintains a current best solution x_best, its value y_best,
# a global step size sigma, and a count of function evaluations used so far.
# Candidate generation: Each iteration samples a small population of Gaussian
# perturbations around the current mean (x_best), evaluates them, and also
# includes a few “directional” probes along random unit vectors to improve
# robustness in varying coordinate systems.
# Selection and replacement: Selects the best candidate among the evaluated
# offspring as the new x_best if it improves the objective; otherwise it
# may still adapt sigma downward to reflect lack of progress.
# Adaptation: Uses a simple success-based update rule for sigma (increase on
# improvement, decrease on stagnation), plus a mild shrink schedule as budget
# depletes.
# Exploration mechanisms: Random Gaussian steps and occasional directional
# probes encourage exploration, especially early in the budget.
# Exploitation mechanisms: The search is centered on the current best point,
# so successful improvements concentrate sampling around promising regions.
# Boundary handling: Candidates are projected back into box bounds using
# clipping; bounds are read from func.lower/func.upper or func.bounds.lb/ub.
# Budget strategy: Strictly caps total objective evaluations to the provided
# budget; the loop dynamically reduces offspring count if needed.
# Closest known influences: Inspired by the classic (1+λ)-ES and simple
# success-based step-size adaptation, adapted for strict evaluation budgets.
# Novelty or unusual aspects: Combines ES-style offspring sampling with a small
# directional probing component and an evaluation-budget-aware offspring count.
# Failure modes: If the objective is extremely noisy or highly non-stationary,
# the success rule may lead to premature shrinking or occasional wasted probes;
# projection can also reduce effective exploration near tight bounds.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # ---- Read bounds robustly ----
        lb = ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub.")

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Ensure valid bounds ordering
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo
        span = np.where(span > 0, span, 1.0)  # avoid zeros for sigma initialization

        # ---- Evaluation budget tracking ----
        max_evals = self.budget
        evals = 0

        def clipped(x):
            return np.clip(x, lo, hi)

        def eval_obj(x):
            nonlocal evals
            if evals >= max_evals:
                # Should never happen if the logic is correct.
                # Return a very bad value to avoid crashes.
                return float("inf")
            x = clipped(x)
            y = float(func(x))
            evals += 1
            return y

        # ---- Initialization ----
        # Start from the center; if span is huge or irregular, this is still safe.
        x_best = clipped(0.5 * (lo + hi))
        y_best = eval_obj(x_best)

        # Initialize step size proportional to the typical scale of the search space.
        # Choose a conservative starting sigma for stability.
        sigma = 0.3 * np.mean(span)
        sigma = max(sigma, 1e-12)

        # Population size: keep small to reduce evaluation overhead.
        # Use a heuristic based on dimension but cap to avoid budget waste.
        # (1+λ)-like behavior can be simulated by keeping λ modest.
        lam_base = max(4, min(24, int(4 + np.sqrt(self.dim) * 2)))
        # Directional probes count (small constant).
        dir_probes = 2

        # Success-based adaptation parameters
        # Increase factor slightly on improvement, decrease factor on failure.
        inc = 1.18
        dec = 0.82

        # Progress-based minimum sigma to prevent numerical issues.
        sigma_min = 1e-15

        # ---- Main loop ----
        # We already used 1 evaluation for x_best. Remaining are for iterations.
        while evals < max_evals:
            remaining = max_evals - evals
            if remaining <= 0:
                break

            # Choose offspring count given remaining budget.
            # We also include a couple of directional probes each iteration.
            lam = min(lam_base, remaining - min(dir_probes, remaining))
            if lam <= 0:
                break

            # Additional candidates from directional probes (can overlap with Gaussian samples,
            # but they are deterministic given random draws below).
            probes = min(dir_probes, max(0, remaining - lam))
            total_candidates = lam + probes

            # Mutation: gaussian perturbations around x_best.
            # Use standard normal for numerical stability and speed.
            # Shape: (total, dim)
            z = np.random.randn(total_candidates, self.dim)
            # Candidate vectors
            candidates = x_best + sigma * z

            # Directional probes: move along random unit vectors with a larger step.
            # This helps when progress is mostly along some direction.
            if probes > 0:
                # Replace last 'probes' candidates by directional probes
                U = np.random.randn(probes, self.dim)
                norms = np.linalg.norm(U, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                U = U / norms
                # Use a step size that is a bit larger than gaussian typical size
                step = sigma * (1.5 + 1.0 * np.random.rand(probes, 1))
                # Random sign per probe
                sign = np.where(np.random.rand(probes, 1) < 0.5, -1.0, 1.0)
                candidates[lam:lam + probes, :] = x_best + sign * step * U

            # Evaluate candidates
            improved = False
            best_local_x = x_best
            best_local_y = y_best

            # Evaluate carefully to not exceed remaining budget.
            for i in range(total_candidates):
                if evals >= max_evals:
                    break
                y = eval_obj(candidates[i])
                if y < best_local_y:
                    best_local_y = y
                    best_local_x = clipped(candidates[i])
                    improved = True

            # Update best solution if any candidate improved
            if improved:
                x_best = best_local_x
                y_best = best_local_y
                sigma = max(sigma_min, sigma * inc)
            else:
                sigma = max(sigma_min, sigma * dec)

            # Mild schedule: as budget depletes, shrink sigma a bit to focus exploitation.
            # This is budget-aware without tracking iterations explicitly.
            progress = evals / max_evals
            sigma *= (1.0 - 0.15 * progress)
            sigma = max(sigma_min, sigma)

            # Edge case: if sigma becomes too small relative to bounds, still continue
            # until budget is exhausted (or sigma min ensures progress isn't impossible
            # due to numeric issues).
            if evals >= max_evals:
                break

        return np.asarray(x_best, dtype=float), float(y_best)
