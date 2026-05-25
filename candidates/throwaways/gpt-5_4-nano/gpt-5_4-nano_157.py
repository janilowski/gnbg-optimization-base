# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# (derivative-free) suitable for GNBG-style benchmarks. It mixes several
# population-free strategies: random sampling, local "coordinate-like"
# perturbations around the current best, and occasional larger re-scales to
# recover from stagnation.
#
# Search state: The algorithm maintains a single incumbent best point
# (best_x, best_y) plus a step-size (sigma) that controls perturbation scale.
# It also tracks how many evaluations remain and a simple stagnation counter.
#
# Candidate generation: Each iteration generates a batch of candidates by
# sampling normally-distributed perturbations around best_x using sigma.
# Candidates are clipped to bounds. Additionally, it injects a small number of
# "global" candidates sampled uniformly across the full bounds to encourage
# exploration.
#
# Selection and replacement: The best candidate among the generated set is
# evaluated, and if it improves the incumbent then it replaces best_x/best_y.
# The step-size sigma is adapted depending on whether improvement happens.
#
# Adaptation: Sigma shrinks when progress is made to focus exploitation, and
# it grows when no improvement is observed to widen the search.
#
# Exploration mechanisms: Uniform global samples and occasional larger
# perturbations (via sigma inflation) help escape local minima.
#
# Exploitation mechanisms: Many candidates are generated near best_x with a
# relatively small sigma so the search refines around promising regions.
#
# Boundary handling: All candidate points are clipped to feasible bounds. The
# bounds are read from func.lower/func.upper or func.bounds.lb/func.bounds.ub.
#
# Budget strategy: The algorithm strictly never exceeds the provided evaluation
# budget; it allocates an initial random phase and then iterates until the
# remaining evaluation budget is exhausted.
#
# Closest known influences: This design is reminiscent of simple evolution
# strategies (1/5-like adaptation) and random-restart local search, adapted
# for a single incumbent and a strict evaluation budget.
#
# Novelty or unusual aspects: It uses batch-style candidate creation but only
# evaluates as many as budget allows, with a lightweight stagnation-driven
# schedule to modulate sigma and exploration frequency.
#
# Failure modes: If the objective is extremely noisy or highly irregular, the
# simple improvement-based adaptation may oscillate or stagnate; clipping can
# also reduce effective search near the bounds.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lower, upper = self._read_bounds(func)
        d = self.dim

        # Ensure sane shapes and types
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        if lower.shape[0] != d or upper.shape[0] != d:
            raise ValueError("Bounds dimensionality does not match dim.")
        # In case bounds are degenerate, avoid zero volume issues in step sizing
        span = upper - lower
        span = np.where(span > 0, span, 1.0)

        # Helper to evaluate safely and track budget
        evals = 0
        best_x = None
        best_y = np.inf

        def eval_point(x):
            nonlocal evals, best_x, best_y
            if evals >= self.budget:
                return
            # Ensure 1D numpy array
            x = np.asarray(x, dtype=float).reshape(-1)
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Corner case: budget 0
        if self.budget <= 0:
            # Return something valid
            mid = (lower + upper) / 2.0
            return mid, float("inf")

        # Initialize incumbent with the best of a few random samples
        # (budget-aware).
        rng = np.random
        init_trials = min(max(2, d + 1), self.budget)
        for _ in range(init_trials):
            x = lower + rng.random(d) * (upper - lower)
            eval_point(x)

        if best_x is None:
            best_x = (lower + upper) / 2.0
            best_y = float("inf")

        # Step-size: start with a fraction of the box size, capped for stability.
        # sigma is global scale for perturbations in each coordinate.
        sigma = 0.25 * np.mean(span)
        sigma = float(max(sigma, 1e-12))

        # Stagnation counter: if no improvement for a while, increase exploration.
        no_improve = 0
        # Target batch size for candidate generation
        # Keep small to maintain low variance and budget responsiveness.
        batch_size = min(12, max(3, d))

        # Main loop: strict budget compliance
        while evals < self.budget:
            remaining = self.budget - evals

            # Adaptation schedule parameters
            # - If stagnating, widen the search and add more global samples.
            # - If improving, focus around best_x with smaller sigma.
            if no_improve >= 5:
                explore_frac = 0.35
                sigma = min(2.0 * sigma, 2.0 * np.mean(span))
                inflate = 1.5
            else:
                explore_frac = 0.15
                inflate = 1.2

            # Determine how many candidates we can evaluate in this iteration
            # without exceeding remaining budget.
            k = min(batch_size, remaining)

            # Decide number of "local" and "global" candidates
            n_global = int(round(explore_frac * k))
            n_global = min(n_global, k)
            n_local = k - n_global

            candidates = []

            # Global exploration candidates: uniform samples over bounds
            for _ in range(n_global):
                x = lower + rng.random(d) * (upper - lower)
                candidates.append(x)

            # Local exploitation candidates: Gaussian perturbations around best_x
            # with coordinate-wise noise, scaled by sigma and occasional inflation.
            if n_local > 0:
                # Perturbation matrix: each row is a candidate
                noise = rng.normal(size=(n_local, d))
                step = (sigma * inflate) * noise
                # Broadcast add and clip
                local = best_x[None, :] + step
                local = np.clip(local, lower, upper)
                candidates.extend([local[i] for i in range(n_local)])

            # Evaluate all candidates, but do not exceed budget
            prev_best_y = best_y
            for x in candidates:
                if evals >= self.budget:
                    break
                eval_point(x)

            # Update stagnation counter and sigma based on whether improvement happened
            if best_y < prev_best_y:
                no_improve = 0
                # Shrink sigma to exploit the promising basin
                sigma = max(0.6 * sigma, 1e-12)
            else:
                no_improve += 1
                # Mild expansion when not improving
                sigma = min(1.15 * sigma, 2.0 * np.mean(span))

        return best_x, best_y

    def _read_bounds(self, func):
        # Bounds may be provided as func.lower/func.upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = func.lower
            upper = func.upper
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lower = b.lb
                upper = b.ub
            else:
                raise AttributeError("func.bounds must provide lb and ub.")
        else:
            raise AttributeError("func must provide either (lower, upper) or func.bounds.(lb, ub).")
        return lower, upper
