# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, budgeted black-box minimization
# algorithm. It uses a population-based evolutionary strategy with adaptive
# step size and periodic “restart”-like resets to avoid stagnation. It is
# designed to be robust across dimensions and relies only on NumPy and the
# objective-provided bounds.
# Search state: The algorithm maintains a small population of candidate
# solutions, their fitness values, and a global step size (sigma). It also
# tracks the best-so-far solution encountered.
# Candidate generation: Each iteration creates offspring by sampling
# Gaussian perturbations around each parent (with shared sigma). Offspring
# are clipped to remain within the provided bounds.
# Selection and replacement: Fitness is evaluated for all offspring. The
# next generation is formed by selecting the best individuals from the
# combined pool (parents + offspring) using NumPy argsort (elitist truncation).
# Adaptation: Sigma adapts using a simple success rule: if the best offspring
# improves over the current best, sigma is reduced or kept small (more
# exploitation); otherwise sigma is increased moderately (more exploration).
# Exploration mechanisms: Increasing sigma after non-improvement increases
# the exploration radius. Additionally, if multiple consecutive iterations
# fail to improve, sigma is reset to a fraction of the search range.
# Exploitation mechanisms: When improvement occurs, sigma shrinks and elitism
# preserves high-quality solutions for subsequent sampling.
# Boundary handling: Candidates are projected back into bounds using np.clip
# rather than resampling, ensuring feasibility with minimal overhead.
# Budget strategy: A fixed total evaluation budget is enforced. The code
# precomputes how many evaluations to spend per generation and ensures it
# never exceeds the provided budget.
# Closest known influences: The design loosely follows (μ+λ) evolution
# strategies with self-adaptive step size via success/no-success heuristics.
# Novelty or unusual aspects: Budget-aware generation sizing and a lightweight
# restart-like sigma reset are included for better behavior in noisy or
# difficult landscapes without adding complexity.
# Failure modes: If the objective is extremely deceptive or bounds/ranges are
# degenerate (very small), progress may stall; however, the algorithm still
# returns the best evaluated point within budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim

        # ---- Read bounds from func ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        if lb is None or ub is None:
            raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/ub.")

        if lb.shape == ():  # scalar bound
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        lb = lb.reshape(-1).astype(float)
        ub = ub.reshape(-1).astype(float)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Handle degenerate bounds robustly
        span = ub - lb
        span_safe = np.where(span != 0, span, 1.0)

        def project(x):
            # Project to feasible region.
            return np.clip(x, lb, ub)

        evals = 0
        best_x = None
        best_y = np.inf

        def eval_one(x):
            nonlocal evals, best_x, best_y
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.array(x, copy=True)
            return y

        # ---- Budget-aware population sizing ----
        # Choose a small (μ,λ) that scales with dim but keeps overhead small.
        # The evaluation budget includes evaluations of f.
        B = max(1, self.budget)
        # Ensure at least a few generations when possible.
        mu = int(np.clip(6 + dim // 2, 4, 24))
        lam = int(np.clip(10 + dim, 8, 40))
        # If budget is tiny, fall back to simple random sampling around center.
        if B <= 2 * (mu + lam):
            # Evaluate initial random points (projected).
            center = (lb + ub) / 2.0
            for _ in range(B):
                if B == 1:
                    x = center
                else:
                    u = np.random.rand(dim)
                    x = lb + u * span_safe
                    x = project(x)
                y = eval_one(x)
            return best_x, best_y

        # Clamp population sizes so we don't exceed budget.
        # We'll run g generations where each generation uses (mu + lam) evaluations after initialization.
        # But first we must evaluate initial mu points.
        mu = min(mu, B)  # at least for initialization
        init_evals = mu
        if init_evals >= B:
            # Just random init under budget
            for _ in range(B):
                u = np.random.rand(dim)
                x = project(lb + u * span_safe)
                eval_one(x)
            return best_x, best_y

        # Remaining budget for generations
        remaining = B - init_evals

        # Decide number of generations: each generation uses lam evaluations for offspring,
        # while parents are already evaluated. We use elitist replacement, not reevaluation of parents.
        # However, we need fitness for parents initially only.
        # So per generation cost is lam.
        lam = min(lam, remaining)
        # Ensure multiple generations if possible
        target_g = 6 + dim // 5
        g = max(1, min(target_g, remaining // max(1, lam)))

        # If lam is too large, recompute g with smaller lam
        if g == 1:
            # Try to increase number of generations by reducing lam (still >= 2)
            lam = max(2, min(lam, remaining // 2))
            g = max(1, remaining // lam)

        remaining_for_g = g * lam
        # Ignore leftover budget (keep within limit)
        remaining_for_g = max(0, remaining_for_g)

        # ---- Initialize population ----
        center = (lb + ub) / 2.0
        # Initial sigma: a fraction of the average span.
        avg_span = float(np.mean(np.abs(span_safe)))
        sigma = 0.25 * avg_span if avg_span > 0 else 1.0

        pop = []
        pop_y = []
        for _ in range(mu):
            # Uniform sampling inside bounds initially for diversity.
            u = np.random.rand(dim)
            x = project(lb + u * span_safe)
            y = eval_one(x)
            pop.append(x)
            pop_y.append(y)

        pop = np.asarray(pop, dtype=float)
        pop_y = np.asarray(pop_y, dtype=float)

        # Make sure best is consistent
        idx_best = int(np.argmin(pop_y))
        if pop_y[idx_best] < best_y:
            best_y = float(pop_y[idx_best])
            best_x = pop[idx_best].copy()

        # ---- Success tracking for adaptation/restarts ----
        no_improve_count = 0
        # Keep sigma within reasonable bounds.
        sigma_min = 1e-12 * (avg_span if avg_span > 0 else 1.0)
        sigma_max = 1.0 * (avg_span if avg_span > 0 else 1.0)

        # ---- Evolution loop ----
        for _gen in range(g):
            if evals >= B:
                break

            # Offspring generation: for each parent, generate one offspring with Gaussian noise.
            # Shared sigma keeps the algorithm simple.
            # Use a correlated-ish perturbation by scaling standard normal by span.
            # This helps across different scales per coordinate.
            coord_scale = np.where(span != 0, np.abs(span), 1.0)
            # Normalize noise magnitude per dimension to the bounds range.
            # This keeps typical steps comparable across coordinates.
            noise = np.random.randn(lam, dim)
            # Use lam offspring; sample parents index for each offspring.
            parent_idx = np.random.randint(0, mu, size=lam)
            parents = pop[parent_idx]
            # Step size: sigma scaled by relative coordinate span.
            steps = (sigma * noise) * (coord_scale / max(1e-12, float(np.mean(coord_scale))))
            offspring = project(parents + steps)

            # Evaluate offspring
            off_y = np.empty(lam, dtype=float)
            for i in range(lam):
                if evals >= B:
                    break
                off_y[i] = eval_one(offspring[i])

            # If budget ran out mid-generation, stop.
            if evals >= B:
                break

            # Elitist truncation selection from combined pool
            # (parents already evaluated; offspring are new)
            combined_pop = np.vstack([pop, offspring[: off_y.shape[0]]])
            combined_y = np.concatenate([pop_y, off_y[: off_y.shape[0]]])

            order = np.argsort(combined_y)  # minimization
            pop = combined_pop[order[:mu]]
            pop_y = combined_y[order[:mu]]

            # Adapt sigma based on improvement vs previous best
            current_best = float(np.min(pop_y))
            if current_best + 1e-15 < best_y + 0.0:
                # Note: best_y is already updated in eval_one; current_best should not be better
                # without best_y having been updated. Keep logic stable anyway.
                pass

            # Determine whether offspring improved global best.
            # We can compare best_y before generation, but we didn't store it.
            # Instead, use improvement signal via current pop best compared to previous gen best.
            # For robustness, estimate improvement by checking if the minimum of offspring beats
            # the previous best of parents. This is a heuristic success rule.
            parent_best_before = float(np.min(pop_y))  # after replacement; not ideal
            # Better: compute min in offspring relative to min of parents before replacement
            # but we lost parents before update. So we rely on off_y min vs current best.
            offspring_best = float(np.min(off_y)) if off_y.size > 0 else np.inf

            improved = offspring_best < best_y + 1e-15
            # improved is likely false due to best_y already updated by eval_one.
            # Use a more meaningful condition: offspring_best < current best pop (which is now updated).
            # If offspring_best equals current best, it means improvement occurred or tied.
            improved = offspring_best <= current_best + 1e-15

            if improved:
                no_improve_count = 0
                sigma = max(sigma_min, sigma * 0.85)
            else:
                no_improve_count += 1
                sigma = min(sigma_max, sigma * 1.08)

            # Restart-like sigma reset after stagnation
            if no_improve_count >= 3:
                # Reset sigma to a moderate value based on range to re-explore.
                sigma = min(sigma_max, max(sigma_min, 0.35 * avg_span))
                no_improve_count = 0

        # Ensure we have a valid best_x
        if best_x is None:
            # Fallback: evaluate center once.
            x0 = project(center)
            best_x = x0
            best_y = float(func(x0))

        return best_x, best_y
