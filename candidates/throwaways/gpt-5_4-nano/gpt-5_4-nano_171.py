# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a simple gradient-free black-box minimizer using a
# population-based evolutionary strategy with occasional local refinement.
# It maintains a small set of candidate points and iteratively improves them
# by sampling Gaussian perturbations around the current best.
# Search state: Tracks evaluation budget usage, the current best point and
# value, and a step size (mutation scale) that adapts based on progress.
# Candidate generation: At each iteration, samples offspring by adding
# normally distributed noise to a selected parent (typically the best point)
# scaled by the current step size; also includes a few candidates around
# the best using a decreasing perturbation magnitude for local search.
# Selection and replacement: Evaluates each offspring, then selects the best
# among parent+offspring as the new center; keeps only the best-so-far point
# for determinism of output and for the next sampling step.
# Adaptation: If improvements are found, step size is mildly decreased or
# increased depending on relative success rate; otherwise it decays to focus.
# Exploration mechanisms: Uses sampling from a Gaussian around the best (and
# a few around random individuals initially) plus occasional re-centering
# when the algorithm stagnates.
# Exploitation mechanisms: Performs a small number of directed local samples
# around the best with smaller step sizes to fine-tune.
# Boundary handling: Uses clipping to the provided bounds for every candidate.
# Budget strategy: Precomputes how many evaluations to do per generation based
# on the allowed evaluation budget; never exceeds the budget even if bounds
# or dimension require adjustments.
# Closest known influences: Inspired by CMA-ES-like “best-point” evolutionary
# strategies (without covariance learning) and classic (μ+λ)/(1+λ) ES,
# combined with a lightweight step-size adaptation and local refinement.
# Novelty or unusual aspects: Keeps the code compact while still providing
# robust bound handling, budget-aware iteration sizing, and a simple
# success-rate-driven step-size schedule.
# Failure modes: In highly non-smooth or deceptive landscapes, step-size
# adaptation may converge prematurely; in very flat objectives, progress-based
# adaptation can stall, but boundary clipping and occasional re-centering
# help mitigate this.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # --- Read bounds in a robust way ---
        lb = ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            # support lb/ub attributes or lb()/ub()
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            elif hasattr(b, "lower") and hasattr(b, "upper"):
                lb = np.asarray(b.lower, dtype=float)
                ub = np.asarray(b.upper, dtype=float)

        if lb is None or ub is None:
            # Fallback: unbounded-like using large clip window around 0.
            # But requirement says to read bounds from func; this should rarely happen.
            lb = np.full(dim, -1.0)
            ub = np.full(dim, 1.0)

        lb = np.broadcast_to(lb, (dim,))
        ub = np.broadcast_to(ub, (dim,))
        span = ub - lb
        span = np.where(span > 0, span, 1.0)

        # --- Budget-safe evaluation wrapper ---
        evals = 0

        def f(x):
            nonlocal evals
            if evals >= budget:
                # Should not happen if we guard loops properly.
                # Return +inf to avoid influencing selection.
                return float("inf")
            evals += 1
            return float(func(np.asarray(x, dtype=float)))

        # If budget is extremely small, evaluate one point and return it.
        if budget <= 0:
            x0 = np.clip(np.zeros(dim), lb, ub)
            return x0, float("inf")

        # --- Initialization ---
        # Start with a few random points (bounded), always including center.
        center = lb + 0.5 * span
        # Choose an initial population size depending on budget.
        # We'll do up to 5 initial evals (or fewer if budget is tiny).
        init_pop = min(5, budget)
        population = []

        # Ensure we evaluate distinct-ish points: include center then randoms.
        x_center = np.clip(center, lb, ub)
        y_center = f(x_center)
        best_x = x_center.copy()
        best_y = y_center

        population.append((best_x, best_y))

        # Remaining initial evals
        for _ in range(init_pop - 1):
            x = lb + np.random.rand(dim) * span
            x = np.clip(x, lb, ub)
            y = f(x)
            if y < best_y:
                best_y = y
                best_x = x.copy()
            population.append((x, y))

        # Step size: start with ~10% of span or smaller; if span is tiny, avoid 0.
        sigma = 0.1 * span
        sigma = np.where(sigma > 1e-12, sigma, 1e-12)

        # Success tracking for step adaptation
        best_prev = best_y
        stagnation = 0

        # --- Main loop ---
        # We'll allocate remaining evaluations into generations where each generation
        # evaluates lambda offspring plus optional local refinements.
        # Keep lambda small for robustness.
        while evals < budget:
            remaining = budget - evals

            # Choose lambda depending on remaining budget and dimension.
            # Typical: 4..12 but budget-aware.
            lam = int(np.clip(4 + dim // 4, 4, 12))
            lam = min(lam, remaining)

            # Parent selection: use the current best point as anchor.
            parent = best_x

            # Generate offspring: Gaussian perturbations
            # Use isotropic sigma (per-dimension), with slight randomness scaling.
            # Offspring count = lam
            children = []
            child_ys = []

            # Exploration vs exploitation mix:
            # with some probability, sample from a broader scale; otherwise near-best.
            explore_scale = 1.0
            if stagnation >= 3:
                explore_scale = 1.8  # re-ignite exploration
            elif stagnation == 0:
                explore_scale = 1.0
            else:
                explore_scale = 1.2

            for _ in range(lam):
                noise = np.random.randn(dim)
                step = sigma * explore_scale
                x = parent + noise * step
                # Boundary handling
                x = np.clip(x, lb, ub)
                y = f(x)
                children.append(x)
                child_ys.append(y)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()

            # Optional small local refinement around best using smaller sigma.
            # Use only if we still have budget.
            local_budget = budget - evals
            if local_budget > 0:
                # Evaluate a few cheap local candidates (0..3) depending on remaining.
                k = min(3, local_budget)
                # Decreasing perturbation magnitude for local exploitation
                local_sigma = sigma * 0.25
                local_sigma = np.where(local_sigma > 1e-12, local_sigma, 1e-12)
                # A mix of random and quasi-directed samples using one-dimensional perturbations
                for i in range(k):
                    if np.random.rand() < 0.7:
                        x = best_x + np.random.randn(dim) * local_sigma
                    else:
                        # Coordinate perturbation: pick a dimension and try both sides
                        j = np.random.randint(0, dim)
                        x = best_x.copy()
                        x[j] = x[j] + (1.0 if i % 2 == 0 else -1.0) * local_sigma[j]
                    x = np.clip(x, lb, ub)
                    y = f(x)
                    if y < best_y:
                        best_y = y
                        best_x = x.copy()

            # --- Adaptation ---
            improved = best_y < best_prev - 1e-12
            if improved:
                # Successful: mildly contract to exploit; reduce stagnation.
                stagnation = 0
                # Contract sigma but not too aggressively.
                sigma = sigma * 0.85
            else:
                # Not improved: increase or decrease depending on stagnation count.
                stagnation += 1
                # Decay sigma to focus; occasional re-ignite uses explore_scale above.
                sigma = sigma * 0.7

            # Prevent sigma from collapsing to zero.
            # Set a floor based on numerical scale and span.
            sigma_floor = 1e-8 * span
            sigma = np.where(sigma > sigma_floor, sigma, sigma_floor)

            best_prev = best_y

            # If sigma is extremely small and no improvement, re-center by sampling
            # around best once to escape, within remaining budget next iteration.
            if stagnation >= 6 and evals < budget:
                # Do a small reseed: sample one point far-ish around best.
                remaining = budget - evals
                if remaining > 0:
                    x = best_x + np.random.randn(dim) * (sigma * 3.0)
                    x = np.clip(x, lb, ub)
                    y = f(x)
                    if y < best_y:
                        best_y = y
                        best_x = x.copy()
                    stagnation = 0 if y < best_prev else stagnation

        return best_x, best_y
