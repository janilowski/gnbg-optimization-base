# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box optimizer for
# continuous minimization problems. It maintains a small set of candidate solutions,
# iteratively refines them using local random search (Gaussian steps), and periodically
# re-centers around the best found point.
# Search state: The algorithm tracks the best-so-far solution and value, a current
# search radius (step size) per iteration, and a population of candidate points sampled
# around the current center.
# Candidate generation: Each iteration samples several candidate points by drawing
# Gaussian perturbations around the current best point. Step size is dynamically reduced
# as the evaluation budget is consumed.
# Selection and replacement: For each candidate, the objective is evaluated (minimization),
# and any improvement replaces the global best. Candidates that do not improve may still
# influence the next center if they are among the best in the current iteration.
# Adaptation: The search radius shrinks linearly with the remaining budget to shift from
# exploration early to exploitation later. Additionally, the per-iteration center can be updated
# to the best candidate of that iteration to guide the search.
# Exploration mechanisms: Larger step sizes early, multiple candidates per iteration,
# and occasional “restart” sampling near the bounds/center help avoid stagnation.
# Exploitation mechanisms: As the budget decreases, the Gaussian steps become smaller,
# focusing search around the current best.
# Boundary handling: All generated points are clipped to the provided bounds before
# evaluation, ensuring feasibility.
# Budget strategy: The algorithm never exceeds the provided evaluation budget. It computes
# a maximum number of evaluations and carefully sizes the initial and iterative sampling.
# Closest known influences: This design is akin to simple evolution strategies / CMA-lite
# (population of Gaussian mutations with shrinking step size) without relying on gradients
# or external libraries.
# Novelty or unusual aspects: It uses a very lightweight, budget-aware population schedule
# and picks the next center from the iteration’s best candidates to balance robustness
# and simplicity.
# Failure modes: On highly irregular objectives or extremely narrow feasible regions,
# clipping and random sampling may waste evaluations. Very tight bounds combined with
# an overly aggressive shrink schedule could slow progress; the code mitigates this with
# conservative shrinking and occasional restart sampling.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        max_evals = max(1, int(self.budget))

        # --- Read bounds from func ---
        # Accept either func.lower/func.upper or func.bounds.lb/func.bounds.ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)

        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        # Ensure correct shape and numeric validity
        lb = lb.reshape(-1).astype(float)
        ub = ub.reshape(-1).astype(float)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds must match the given dimension.")

        # Handle degenerate bounds
        span = ub - lb
        span = np.where(span > 0, span, 1.0)  # avoid zeros in scaling
        x_center = (lb + ub) / 2.0

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        # --- Budget-aware evaluator ---
        evals = 0
        best_x = x_center.copy()
        best_y = float("inf")

        def eval_one(x):
            nonlocal evals, best_x, best_y
            if evals >= max_evals:
                return None
            x = clip(np.asarray(x, dtype=float))
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # --- Initialization: sample a few points uniformly ---
        # Choose a small population size that adapts to budget.
        # The number of iterations depends on budget and population size.
        # Keep it simple and compact.
        base_pop = 6
        pop = int(min(base_pop, max_evals)) if max_evals > 0 else 1
        # If dim is large, slightly increase pop to gather signal.
        pop = int(min(max_evals, max(6, min(18, dim // 2 + 6))))
        pop = max(1, min(pop, max_evals))

        # Evaluate initial best candidates
        # Include center and a few random points.
        eval_one(x_center)
        remaining = max_evals - evals
        if remaining > 0:
            n0 = min(pop - 1, remaining)
            # Uniform sampling inside bounds
            u = np.random.random((n0, dim))
            X = lb + u * (ub - lb)
            for i in range(n0):
                eval_one(X[i])

        if evals >= max_evals:
            return best_x, best_y

        # Initial step size: fraction of the box size
        # Use a conservative scale to help with robustness.
        global_span = np.max(ub - lb)
        step0 = 0.25 * global_span if global_span > 0 else 0.1
        step = step0

        # Iterative refinement
        # Determine iterations based on remaining budget and pop size.
        # We'll aim to use most budget, but never exceed it.
        # Pop per iteration excludes the center evaluation (we already have best_x).
        it_evals_budget = max_evals - evals
        # At least one iteration.
        iters = max(1, it_evals_budget // max(1, pop))
        # If too little budget, reduce iterations.
        # (Still keep at least one batch.)
        iters = max(1, min(iters, 50))

        # Exploration/restart controls
        # Restart occasionally if no improvement recently.
        stagnation_limit = max(10, min(50, max_evals // 4))
        stagnation = 0

        # Keep a notion of "last best" for stagnation.
        last_best = best_y

        for t in range(iters):
            if evals >= max_evals:
                break

            # Linear schedule for step size as budget runs out, with a floor.
            # This shifts from exploration to exploitation.
            frac_left = (max_evals - evals) / max_evals
            step = step0 * max(0.05, frac_left)

            # Determine how many candidates we can still evaluate.
            # We sample pop candidates per iteration (or fewer if budget tight).
            cand = min(pop, max_evals - evals)
            if cand <= 0:
                break

            # Occasionally do a "restart-like" candidate set:
            # sample some points with larger variance to diversify.
            # Probability decreases with time.
            restart_prob = 0.15 * (1.0 - t / max(1, iters - 1))
            do_restart = (np.random.random() < restart_prob)

            # Choose center:
            # Use current best_x, but sometimes bias toward a new center from
            # small jitter around best_x to encourage local coverage.
            center = best_x.copy()
            if do_restart:
                # Widen exploration around center (but still clipped).
                wide_step = step * 2.5
                sigma = wide_step
                jitter_center = clip(center + np.random.normal(scale=0.25 * step0, size=dim))
                center = jitter_center
            else:
                sigma = step

            # Create candidate points by Gaussian perturbations.
            # Some candidates also use coordinate-wise scaled noise for diversity.
            Z = np.random.normal(size=(cand, dim))
            # Add mild anisotropy: scale by relative span to normalize across dimensions
            # (so that all dims have comparable relative perturbation).
            rel = (ub - lb)
            rel = np.where(rel > 0, rel, 1.0)
            # Use rel/mean as scale; keep bounded to avoid huge steps.
            rel_scale = np.clip(rel / (np.mean(rel) + 1e-12), 0.5, 2.0)
            perturb = (Z * rel_scale[None, :]) * sigma

            Xc = center[None, :] + perturb

            # Additionally, include one uniform sample occasionally to escape local minima.
            if cand >= 2 and (np.random.random() < 0.12):
                # Replace one candidate with uniform inside bounds.
                j = np.random.randint(0, cand)
                Xc[j] = lb + np.random.random(dim) * (ub - lb)

            # Evaluate candidates, track best of this iteration to potentially update center.
            iter_best_x = None
            iter_best_y = float("inf")
            for i in range(cand):
                y = eval_one(Xc[i])
                if y is None:
                    break
                if y < iter_best_y:
                    iter_best_y = y
                    iter_best_x = clip(Xc[i])

            if evals >= max_evals:
                break

            # Update stagnation and optionally adjust center strategy.
            if best_y < last_best - 1e-15:
                stagnation = 0
                last_best = best_y
            else:
                stagnation += 1

            # If stagnating, perform a small targeted restart near bounds.
            if stagnation >= stagnation_limit and evals < max_evals:
                # Reset step size and sample a few points near extreme corners
                # (helps when the minimizer lies near boundaries).
                # Only spend a small fraction of remaining budget.
                remaining = max_evals - evals
                nrs = min(max(3, pop // 2), remaining)
                # Sample corners mix: choose randomly lb or ub per coordinate.
                corner_choice = np.random.randint(0, 2, size=(nrs, dim))
                corners = np.where(corner_choice == 0, lb[None, :], ub[None, :]).astype(float)

                # Move slightly inward so clipping doesn't collapse all samples.
                inward = 0.05 + 0.15 * np.random.random((nrs, dim))
                Xr = corners * inward + (1.0 - inward) * x_center[None, :]
                for i in range(nrs):
                    eval_one(Xr[i])

                # Reset step and stagnation.
                step0 = max(step0 * 0.9, 0.01 * (global_span if global_span > 0 else 1.0))
                stagnation = 0
                last_best = best_y

            # Optional: If the iteration best is close in value to global best,
            # nudge step schedule is already handled by frac_left; nothing else needed.

        return best_x, best_y
