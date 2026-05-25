# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm that works with only function
# evaluations. It is derivative-free, population-based, and uses a combination of global exploration and local
# coordinate-wise exploitation. It maintains a small set of candidate points, keeps the current best, and
# iteratively generates new candidates around good points.
# Search state: The algorithm tracks current best (best_x, best_y) and a population of points with their
# objective values. It also tracks an adaptive step size (sigma) for sampling around promising solutions.
# Candidate generation: Each iteration proposes new candidates using (1) random exploration from the whole
# box and (2) local perturbations around the current best and around the top fraction of the population.
# Perturbations combine isotropic noise scaled by sigma with occasional axis-aligned moves for faster
# local progress in different dimensions.
# Selection and replacement: Candidates are evaluated, then the population is updated by keeping the best
# points (elitism). The global best is updated whenever a better y is found.
# Adaptation: The step size sigma is adapted based on whether new evaluations improved the global best.
# If an improvement happens, sigma is gently decreased (focused search); otherwise it is decreased more
# slowly or partially reset to encourage escape.
# Exploration mechanisms: Uniform random sampling across the bounds is used early and intermittently
# (driven by remaining budget and by lack of improvements) to escape local minima.
# Exploitation mechanisms: Local sampling around best/top points uses Gaussian perturbations plus occasional
# coordinate-wise mutations to better align with separable structure.
# Boundary handling: All candidate points are clipped into the feasible box after sampling.
# Budget strategy: The algorithm strictly respects the evaluation budget by computing how many evaluations
# remain and only evaluating that many candidates per iteration.
# Closest known influences: The design is inspired by small-population evolutionary strategies and
# one-point/best-centered random search with adaptive step sizing and elitist selection.
# Novelty or unusual aspects: The candidate set mixes isotropic and coordinate-wise moves, making it robust
# across differing dimensionalities without needing gradient or problem-specific heuristics.
# Failure modes: If the objective is extremely deceptive or has very narrow feasible basins, the adaptive
# sigma may shrink too quickly; periodic exploratory resets mitigate this. If budget is tiny, the method
# behaves like a mostly random search around a few initial points.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            raise ValueError("budget must be positive")

        # Read bounds from func (supports both APIs).
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lo = np.asarray(func.lower, dtype=float)
            hi = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            lo = np.asarray(getattr(b, "lb"), dtype=float)
            hi = np.asarray(getattr(b, "ub"), dtype=float)
        else:
            raise AttributeError("func must provide bounds via lower/upper or func.bounds.lb/ub")

        # Ensure shape correctness.
        lo = np.broadcast_to(lo, (dim,)).copy()
        hi = np.broadcast_to(hi, (dim,)).copy()
        if np.any(hi < lo):
            raise ValueError("Invalid bounds: upper must be >= lower for all dimensions")

        # Evaluate helper with strict budget accounting.
        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                # Should never happen if budget management is correct.
                return None, True
            y = func(x)
            evals += 1
            return y, False

        # Random number generator uses global numpy seed set by the harness.
        rng = np.random

        # Initial population size. Kept small for evaluation budget constraints.
        # At least 2 to form a meaningful top set, but never exceed budget.
        pop_size = int(min(max(2, dim + 1), max(2, budget)))
        # Determine how many initial points we can afford.
        n_init = min(pop_size, budget)

        # Step size: fraction of average box width.
        box_width = hi - lo
        # Avoid zero-width dimensions producing sigma=0; use small epsilon.
        mean_width = float(np.mean(box_width)) if np.all(box_width >= 0) else float(np.mean(np.maximum(box_width, 0.0)))
        sigma0 = 0.3 * (mean_width if mean_width > 0 else 1.0)
        sigma_min = 1e-12 * (mean_width if mean_width > 0 else 1.0)

        # Allocate population.
        X = np.empty((n_init, dim), dtype=float)
        Y = np.empty(n_init, dtype=float)

        # Sample initial candidates uniformly.
        for i in range(n_init):
            if evals >= budget:
                break
            x = lo + (hi - lo) * rng.rand(dim)
            x = np.clip(x, lo, hi)
            y, done = evaluate(x)
            if done:
                break
            X[i] = x
            Y[i] = y

        # If budget was extremely small, handle partial fill.
        valid = evals
        if valid == 0:
            # Should not happen because budget>0 and we start evaluating, but just in case.
            raise RuntimeError("No evaluations performed.")
        X = X[:valid]
        Y = Y[:valid]

        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])

        # Population parameters for subsequent iterations.
        # We keep a fixed pop size for simplicity; if we started with smaller, we adapt.
        pop_size = max(pop_size, valid)
        pop_cap = max(pop_size, valid)

        # Track best improvement to adapt sigma.
        improved_recently = False

        # Main loop: each iteration evaluates a small batch of candidates.
        # Batch size chosen to amortize overhead while respecting budget.
        # If budget is small, batch will naturally shrink.
        max_iters = 1 + (budget - evals)  # loose upper bound; actual evals stop earlier
        iteration = 0

        while evals < budget and iteration < max_iters:
            iteration += 1

            # Remaining evaluations determine batch size.
            rem = budget - evals
            if rem <= 0:
                break

            # Number of candidates to evaluate this iteration.
            # Use min of rem and a small fraction of pop/hidden constant.
            batch = int(min(rem, max(2, pop_cap // 2)))
            # Ensure at least 1 candidate if rem > 0.
            batch = max(1, batch)

            # Sort current population by fitness.
            order = np.argsort(Y)
            # Take top elites for local moves.
            elite_count = int(max(1, min(len(order), max(2, pop_cap // 4))))
            elites = X[order[:elite_count]]

            # Probability of global exploration vs local exploitation.
            # Increase global exploration when no improvement recently or later in the run.
            progress = evals / max(1, budget)
            p_explore = 0.25 + 0.4 * (1.0 - progress)
            if not improved_recently:
                p_explore += 0.15
            p_explore = float(min(0.75, max(0.05, p_explore)))

            # Adapt sigma based on improvement.
            if improved_recently:
                sigma = max(sigma_min, sigma0 * (0.75 ** iteration))
            else:
                # Slight decay; occasionally reset to escape.
                sigma = max(sigma_min, sigma0 * (0.90 ** iteration))
                # Soft reset if stuck for a long time (based on progress).
                if (iteration % 7) == 0 and progress > 0.25:
                    sigma = max(sigma, 0.5 * sigma0)

            # Generate and evaluate candidates.
            new_X = []
            new_Y = []
            for _ in range(batch):
                if evals >= budget:
                    break

                if rng.rand() < p_explore:
                    # Global exploration: uniform across the whole box.
                    x = lo + (hi - lo) * rng.rand(dim)
                else:
                    # Local exploitation:
                    # Choose a center: sometimes best, often elites.
                    if rng.rand() < 0.7:
                        center = best_x
                    else:
                        center = elites[rng.randint(0, len(elites))]

                    # Create a perturbation:
                    # - Isotropic Gaussian component.
                    # - Occasional coordinate-aligned move.
                    noise = rng.randn(dim) * sigma
                    x = center + noise

                    # Coordinate-wise mutation to diversify local search
                    if dim >= 2 and rng.rand() < 0.35:
                        k = rng.randint(0, dim)
                        # Move along one coordinate using box width scale.
                        width_k = float(hi[k] - lo[k])
                        step_k = (0.5 + rng.rand()) * (width_k if width_k > 0 else 1.0)
                        direction = -1.0 if rng.rand() < 0.5 else 1.0
                        x[k] = center[k] + direction * 0.15 * step_k

                    # Slightly bias toward bounds (helps when optimum is near boundary).
                    if rng.rand() < 0.10:
                        # Pick dimension and push toward whichever bound is closer.
                        j = rng.randint(0, dim)
                        if abs(center[j] - lo[j]) < abs(center[j] - hi[j]):
                            x[j] = lo[j] + 0.05 * (hi[j] - lo[j]) * rng.rand()
                        else:
                            x[j] = hi[j] - 0.05 * (hi[j] - lo[j]) * rng.rand()

                # Boundary handling: clip into feasible region.
                x = np.clip(x, lo, hi)
                y, done = evaluate(x)
                if done:
                    break
                new_X.append(x)
                new_Y.append(y)

                # Update best immediately to drive sigma adaptation.
                if y < best_y:
                    best_y = float(y)
                    best_x = np.array(x, copy=True)
                    improved_recently = True

            if len(new_X) == 0:
                # No new evaluations done due to budget edge-case.
                break

            improved_any = improved_recently

            # Merge old population with new candidates, then truncate by fitness (elitism).
            X = np.vstack([X, np.asarray(new_X, dtype=float)])
            Y = np.concatenate([Y, np.asarray(new_Y, dtype=float)])

            # Truncate to pop_cap to keep computations bounded.
            order = np.argsort(Y)
            keep = min(pop_cap, len(order))
            X = X[order[:keep]]
            Y = Y[order[:keep]]

            # If we didn't improve in this iteration, decay improved flag.
            if not improved_any:
                improved_recently = False
            else:
                # Keep improved flag for one iteration.
                improved_recently = True

        return best_x, best_y
