# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer using a
# budget-aware evolution strategy with a diagonal Gaussian mutation model.
# Search state: Maintains a current best solution x_best and its fitness
# y_best, plus a running step-size sigma that controls mutation scale.
# Candidate generation: Each iteration samples a small population from a
# Normal(x_best, sigma) distribution (independently per dimension), using a
# mirrored (antithetic) pairing to reduce variance.
# Selection and replacement: Evaluates all candidates, picks the best one
# (lowest objective value), and replaces x_best if an improvement is found.
# Adaptation: Updates sigma using a simple success rule based on how often
# candidates improve within an iteration, making sigma shrink on failures
# and grow slightly on successes.
# Exploration mechanisms: Population sampling and adaptive sigma provide global
# exploration early on and local search later.
# Exploitation mechanisms: Centering the search distribution at the current best
# progressively focuses sampling around the best-so-far point.
# Boundary handling: Clips candidates to the provided bounds after mutation to
# ensure feasible points; clipping is applied consistently before evaluation.
# Budget strategy: Consumes exactly the provided evaluation budget by tracking
# remaining evaluations and truncating the last batch if needed.
# Closest known influences: Similar in spirit to evolution strategies (ES) and
# mirrored sampling used in derivative-free optimization; simplified to keep
# the implementation compact.
# Novelty or unusual aspects: Uses a mirrored population and a budget-aware
# batch evaluation that may terminate early without exceeding the budget.
# Failure modes: If the objective is highly noisy or deceptive, simple success
# adaptation may oscillate; clipping can also bias search near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # --- Read bounds from func ---
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("func must provide bounds via (lower, upper) or func.bounds.lb/ub")

        if lb.shape == () or ub.shape == ():
            # Scalar bounds (broadcast to dim)
            lb = np.full(self.dim, float(lb))
            ub = np.full(self.dim, float(ub))
        else:
            lb = np.broadcast_to(lb, (self.dim,)).astype(float, copy=False)
            ub = np.broadcast_to(ub, (self.dim,)).astype(float, copy=False)

        # Ensure valid bounds order
        if np.any(ub < lb):
            raise ValueError("Invalid bounds: require upper >= lower elementwise")

        dim = self.dim
        budget = max(0, int(self.budget))
        if budget == 0:
            # No evaluations allowed: return a valid point anyway.
            # (Harness likely expects y from evaluations, but requirement is to never exceed budget.)
            x0 = 0.5 * (lb + ub)
            return x0, float("inf")

        # --- Budget tracking ---
        evals_used = 0

        def clamp(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_at(x):
            nonlocal evals_used
            if evals_used >= budget:
                # Must not exceed budget; guard to avoid overflow on last batch.
                return float("inf")
            x = clamp(np.asarray(x, dtype=float))
            y = float(func(x))
            evals_used += 1
            return y

        # --- Initialization ---
        # Choose a starting point near the center; add tiny jitter to avoid exact ties.
        x_best = 0.5 * (lb + ub)
        x_best = clamp(x_best)
        # Use one evaluation for x_best (if budget permits).
        y_best = eval_at(x_best)

        # Step-size: start at a fraction of the diagonal range.
        span = ub - lb
        # If span is zero everywhere, search space is constant; return immediately after initial eval.
        if not np.any(span > 0):
            return x_best, y_best

        # Robust initial sigma: proportional to typical coordinate range.
        # Use geometric mean-like scaling but stable.
        positive_span = np.where(span > 0, span, 1.0)
        sigma = 0.3 * np.mean(positive_span)

        # --- Search loop ---
        # Population size: keep it small to reduce wasted evaluations.
        # Use an even number to allow mirrored sampling; ensure >= 2.
        base_pop = 8
        pop = min(base_pop, budget - evals_used) if budget - evals_used > 0 else 0
        pop = max(2, int(pop))
        if pop % 2 == 1:
            pop -= 1
        if pop < 2:
            pop = 2
        # We'll adjust final batch sizes dynamically.

        # Success-based adaptation parameters.
        shrink = 0.85
        grow = 1.15
        min_sigma = 1e-12 * max(1.0, np.mean(positive_span))
        max_sigma = 2.0 * max(positive_span)

        while evals_used < budget:
            remaining = budget - evals_used
            # Determine batch size: even for mirrored sampling.
            batch = min(pop, remaining)
            if batch % 2 == 1:
                batch -= 1
            if batch < 2:
                break  # not enough budget for a mirrored pair

            half = batch // 2

            # Generate mirrored perturbations.
            # eps shape: (half, dim)
            eps = np.random.normal(0.0, 1.0, size=(half, dim))
            # Candidates: x_best + sigma*eps and x_best - sigma*eps
            c1 = x_best + sigma * eps
            c2 = x_best - sigma * eps
            # Stack into evaluation list
            candidates = np.vstack([c1, c2])  # shape (batch, dim)

            # Evaluate candidates (clip inside eval_at).
            ys = np.empty(batch, dtype=float)
            for i in range(batch):
                ys[i] = eval_at(candidates[i])

            # Select best candidate
            idx = int(np.argmin(ys))
            y_min = float(ys[idx])
            x_candidate = clamp(candidates[idx])

            # Success heuristic: count how many improved over current best
            # (ties treated as not improving).
            improvements = int(np.sum(ys < y_best))
            # Adapt sigma based on number of improvements (success rate)
            success_rate = improvements / float(batch)
            if success_rate > 0.2:
                sigma = min(max_sigma, sigma * grow)
            elif success_rate < 0.1:
                sigma = max(min_sigma, sigma * shrink)

            # Replacement: always keep best-so-far
            if y_min < y_best:
                x_best = x_candidate
                y_best = y_min

            # If sigma becomes extremely small, still keep going until budget ends
            # but the search will effectively be local.
            if sigma <= min_sigma and evals_used < budget:
                # Continue with very small sigma; do not early exit to respect budget logic.
                sigma = min_sigma

        return x_best, y_best
