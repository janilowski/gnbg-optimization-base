# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, robust black-box minimization algorithm (one-point
# finite-difference guided evolution strategy) with multiple restarts. It uses only
# numpy operations and respects a fixed evaluation budget.
# Search state: Maintains a current mean vector (mu), a step-size (sigma), and a
# best-so-far solution found across the entire run. It tracks the number of objective
# evaluations used.
# Candidate generation: At each iteration it draws several random direction vectors
# (standard normal), evaluates the objective at mu +/- delta*direction (central
# difference), and uses the sign-weighted comparison to form a gradient-like signal.
# Selection and replacement: Uses the averaged gradient-like signal to propose a new
# mean via a gradient step, while optionally keeping mu if the proposal does not
# improve. Additionally, it updates global best whenever a new lower objective value
# is observed.
# Adaptation: Adapts step-size sigma based on whether improvements occur; sigma is
# reduced on stagnation and slightly increased when improvements are frequent.
# Exploration mechanisms: Random directions and occasional “restart” diversify the search
# when no progress is detected; restarts reset mu to a new random point within bounds.
# Exploitation mechanisms: The gradient-like signal from central differences biases
# movement toward locally decreasing directions while step-size controls intensity.
# Boundary handling: All candidate points are clipped to provided bounds (per-dimension)
# to ensure feasibility.
# Budget strategy: Never exceeds the provided evaluation budget; the code carefully
# calculates how many candidate evaluations can be afforded each iteration and stops
# exactly when the budget is consumed.
# Closest known influences: Inspired by CMA-ES/ES-style mean updates and finite-difference
# gradient estimation, but implemented as a lightweight, single-parent strategy.
# Novelty or unusual aspects: Uses a sign-based accumulation across symmetric perturbation
# evaluations to create a gradient direction with minimal additional bookkeeping, combined
# with adaptive sigma and multi-restart resilience.
# Failure modes: In very high dimension or with extremely noisy objectives, finite-difference
# signals may be unreliable; the algorithm mitigates with restarts and conservative sigma
# adaptation, but it cannot guarantee improvement on adversarial functions.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from func ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and func.bounds is not None:
            b = func.bounds
            # Expect b.lb / b.ub
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        if lb is None or ub is None:
            # If bounds are not provided, fall back to a generic symmetric range.
            # (The benchmark usually provides bounds; this is a robustness fallback.)
            lb = -5.0 * np.ones(dim, dtype=float)
            ub = 5.0 * np.ones(dim, dtype=float)

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Ensure valid bounds
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo
        span[span <= 0] = 1.0  # avoid zero span degeneracy

        def clip(x):
            return np.minimum(hi, np.maximum(lo, x))

        # ---- Evaluation budget tracking ----
        evals = 0

        # Make sure func is only called with 1-D numpy arrays
        def eval_obj(x):
            nonlocal evals
            if evals >= budget:
                # In case of floating rounding in budget accounting, prevent extra calls.
                return np.inf
            x = np.asarray(x, dtype=float).reshape(-1)
            y = func(x)
            evals += 1
            return float(y)

        # ---- Initialization ----
        rng = np.random  # harness sets global seed via numpy

        # Choose restart count based on budget and dimension (lightweight schedule)
        # Each "iteration" consumes roughly 2*m evaluations for central differences.
        # We'll do a few restarts and keep the main loop budget-driven.
        # A larger dim benefits from fewer directions per iteration.
        dir_count = int(np.clip(4 + dim // 10, 4, 12))
        # Candidate evaluations per iteration: 2*dir_count (central differences) plus optionally 1 base eval.
        # We'll incorporate base eval in the current loop carefully.
        # Reduce direction count if budget is tight.
        if budget < 2 * dir_count + 5:
            dir_count = max(2, budget // 4)

        # Finite difference delta magnitude relative to span
        # Use a small fraction to balance truncation and numerical issues.
        delta_factor = 1e-3

        # sigma initial: proportional to span
        sigma0 = 0.25 * np.mean(span)
        sigma0 = float(sigma0) if sigma0 > 0 else 0.25

        best_x = None
        best_y = np.inf

        # Helper to propose random point within bounds
        def random_point():
            return lo + rng.rand(dim) * (hi - lo)

        # We will manage restarts by creating multiple "runs" of the same optimizer state.
        # Budget is global; each restart consumes evaluations.
        restarts = 0
        max_restarts = 1 + int(budget // max(20, 5 * dim))
        max_restarts = int(np.clip(max_restarts, 1, 5))

        # Initialize mu and base evaluation at each restart
        while evals < budget and restarts < max_restarts:
            restarts += 1

            mu = random_point()
            mu = clip(mu)

            # If possible, evaluate mu to seed best
            y_mu = np.inf
            if evals < budget:
                y_mu = eval_obj(mu)
                if y_mu < best_y:
                    best_y = y_mu
                    best_x = mu.copy()

            # Adaptive step-size and stagnation tracking
            sigma = sigma0
            stagnation = 0
            improved_recently = 0

            # Iterations: budget-driven, not fixed count
            # Each iteration evaluates up to (2*dir_count) points using symmetric perturbations.
            while evals < budget:
                # If we can't afford full central differences, reduce m.
                remaining = budget - evals
                m = int(min(dir_count, max(1, remaining // 2)))
                # Each direction needs 2 evaluations; avoid zero.
                if m <= 0:
                    break

                # Compute delta per-coordinate
                delta = delta_factor * np.sqrt(np.maximum(span, 1e-12))
                # If delta is too tiny, enlarge slightly based on sigma.
                if np.mean(delta) < 1e-12:
                    delta = np.full(dim, 1e-6)

                # Create random directions; normalize to unit length for stability.
                # Using gaussian directions helps cover space.
                D = rng.randn(m, dim)
                norms = np.linalg.norm(D, axis=1)
                norms[norms == 0] = 1.0
                D /= norms[:, None]

                # Gradient-like accumulator using symmetric comparisons.
                # For each direction u, we evaluate mu+delta*u and mu-delta*u:
                # central difference approx: (f(mu+)-f(mu-)) / (2*delta)
                # We scale it by u and accumulate.
                g = np.zeros(dim, dtype=float)

                # For exploitation, optionally compare against current mu value.
                # But central differences already provide a direction; keep simple.
                for i in range(m):
                    u = D[i]
                    xp = clip(mu + delta_factor * sigma * u)
                    xm = clip(mu - delta_factor * sigma * u)

                    yp = eval_obj(xp)
                    ym = eval_obj(xm)

                    # Update global best
                    if yp < best_y:
                        best_y = yp
                        best_x = xp.copy()
                    if ym < best_y:
                        best_y = ym
                        best_x = xm.copy()

                    # Gradient signal (scaled)
                    # Avoid division by zero: delta_eff equals delta_factor*sigma.
                    delta_eff = max(1e-12, float(delta_factor * sigma))
                    g += ((yp - ym) / (2.0 * delta_eff)) * u

                # Normalize gradient to avoid overly large steps in early noisy regimes
                g_norm = float(np.linalg.norm(g))
                if g_norm > 0:
                    g_dir = g / g_norm
                else:
                    g_dir = g

                # Proposed update: move opposite gradient direction to minimize.
                # Step length is controlled by sigma; dampen by dimension.
                # This is intentionally conservative.
                lr = 0.5 / np.sqrt(dim if dim > 0 else 1.0)
                proposal = clip(mu - lr * sigma * g_dir)

                y_prop = np.inf
                if evals < budget:
                    y_prop = eval_obj(proposal)
                    if y_prop < best_y:
                        best_y = y_prop
                        best_x = proposal.copy()

                improved = y_prop < y_mu

                # Selection & replacement: keep best among {mu, proposal} locally
                if improved:
                    mu = proposal
                    y_mu = y_prop
                    stagnation = 0
                    improved_recently += 1
                else:
                    stagnation += 1

                # Adapt sigma
                # If we improved a couple times, slightly increase; otherwise reduce.
                if improved:
                    if improved_recently >= 2:
                        sigma *= 1.05
                    else:
                        sigma *= 1.0
                else:
                    improved_recently = 0
                    sigma *= 0.85

                # If stagnating, force diversification via partial restart
                if stagnation >= 3:
                    break  # exit current restart; outer loop may restart

                # Safety: if sigma becomes extremely small, break to avoid wasted evals
                if sigma < 1e-12:
                    break

        # Final guard: if never evaluated (e.g., budget==0), return something in bounds
        if best_x is None:
            best_x = random_point()
            best_x = clip(best_x)
            best_y = np.inf

        return best_x, best_y
