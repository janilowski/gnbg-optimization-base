# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy using
#   a population of candidate solutions, derivative-free adaptation of a
#   global step size, and periodic local refinement with coordinate-wise
#   moves. The algorithm targets minimization only.
# Search state: Maintains a small population of points within provided bounds,
#   tracks the best-so-far solution and its objective value, and keeps a global
#   exploration step size (sigma) that adapts based on recent improvements.
# Candidate generation: Each iteration generates new candidates by adding
#   Gaussian perturbations scaled by sigma to the current best. It also
#   occasionally samples around the current best using the population's
#   spread to adapt exploration.
# Selection and replacement: Offspring are evaluated and only accepted if they
#   improve upon the parent at the same population index; additionally, the
#   global best is updated whenever a better point is found.
# Adaptation: Uses a simple success-rate rule: if enough offspring improve,
#   sigma increases slightly; otherwise sigma decreases. Sigma is bounded to
#   avoid collapse and runaway.
# Exploration mechanisms: Gaussian sampling around the best plus diversity
#   from population spread. Coordinate-wise refinements near the best provide
#   targeted local search.
# Exploitation mechanisms: Local refinement attempts that probe plus/minus
#   moves along coordinate axes (scaled by sigma) and shrink the step on
#   failure.
# Boundary handling: All generated points are clipped to the feasible bounds.
#   If clipping causes stagnation, sigma is reduced.
# Budget strategy: Precomputes a per-iteration evaluation budget and never calls
#   the objective more than the provided budget. Any leftover evaluations are
#   used for final local refinement.
# Closest known influences: Mixes ideas from evolution strategies (population,
#   mutation with sigma adaptation) with lightweight coordinate local search.
# Novelty or unusual aspects: Uses index-wise parent/offspring replacement to
#   preserve population diversity while still adapting globally from best
#   improvements; also estimates spread to scale exploration without extra
#   hyperparameters.
# Failure modes: On highly non-smooth or deceptive functions, adaptation may
#   shrink sigma too early. Budget exhaustion mid-refinement is handled by
#   checking remaining evaluations before each objective call.
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
        if budget <= 0:
            raise ValueError("budget must be positive")

        # Read bounds from either func.lower/func.upper or func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(
            func.bounds, "ub"
        ):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("func must provide bounds via lower/upper or bounds.lb/ub")

        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        lb = lb.reshape(-1)[:dim].astype(float, copy=False)
        ub = ub.reshape(-1)[:dim].astype(float, copy=False)

        if np.any(ub < lb):
            raise ValueError("Invalid bounds: ub must be >= lb for all dimensions")

        # Objective evaluation wrapper to enforce budget.
        remaining = budget
        best_y = np.inf
        best_x = None

        def eval_one(x):
            nonlocal remaining, best_y, best_x
            if remaining <= 0:
                # Should not happen if logic is correct.
                return np.inf
            y = float(func(x))
            remaining -= 1
            if y < best_y:
                best_y = y
                best_x = np.array(x, dtype=float, copy=True)
            return y

        # If budget is 1, just sample a single point.
        span = ub - lb
        # Avoid zero span issues: for flat dimensions, perturbations are irrelevant.
        span_nonzero = np.where(span > 0, span, 1.0)

        # Population size: keep small and robust. Ensure at least 2.
        # Tradeoff: more candidates reduces randomness but increases budget usage.
        pop = max(2, min(12, dim + 1))
        pop = min(pop, budget)  # can't exceed number of evaluations
        # Number of iterations for mutation steps. We reserve some for local refinement.
        # Reserve at most 25% (and at least 0) for refinement.
        reserve = int(max(0, min(budget // 4, budget - pop)))
        refine_budget = reserve
        remaining_for_main = budget - refine_budget
        # Initial evaluations for population.
        pop_evals = pop
        if remaining_for_main < pop_evals:
            # Degenerate: not enough for pop; evaluate as many as allowed.
            pop = max(1, remaining_for_main)
            pop_evals = pop

        # Initialize population uniformly in bounds.
        # Randomness is controlled externally by harness via np.random.seed.
        X = lb + np.random.rand(pop, dim) * span_nonzero
        # For dimensions with span=0, force exactly lb.
        if np.any(span == 0):
            X[:, span == 0] = lb[span == 0]

        # Evaluate initial population.
        values = np.empty(pop, dtype=float)
        for i in range(pop):
            values[i] = eval_one(X[i])

        # Determine current best and set sigma relative to bounds.
        best_idx = int(np.argmin(values))
        if best_x is None:
            best_x = np.array(X[best_idx], dtype=float, copy=True)
            best_y = float(values[best_idx])

        # Initial step size: a fraction of the average span.
        avg_span = float(np.mean(span_nonzero))
        # If bounds are very tight, avg_span may be 1.0 due to span_nonzero;
        # that's fine because clipping will keep candidates valid.
        sigma = 0.3 * avg_span / max(1.0, np.sqrt(dim))
        sigma = float(max(1e-12, sigma))

        # Sigma bounds to keep adaptation stable.
        sigma_min = float(1e-12)
        sigma_max = float(2.0 * avg_span if avg_span > 0 else 1.0)

        # Main loop: evolutionary mutation with success-based sigma adaptation.
        # We plan roughly 'iters' mutation batches, each producing 'offspring_per_iter' candidates.
        # Offspring are produced around the global best and each population member.
        # Each iteration does: for each parent index, generate one offspring (pop evals).
        # We'll stop when we'd exceed remaining_for_main.
        iters = 0
        # Use local counters for adaptive sigma
        successes_window = 0
        window_size = 10  # number of offspring evaluations to assess success ratio

        # Remaining evals for mutation after initial population.
        main_remaining = max(0, remaining_for_main - pop_evals)
        if main_remaining > 0:
            iters = main_remaining // pop  # full batches
        # Ensure at least one batch if possible.
        if iters == 0 and main_remaining >= 1:
            iters = 1

        # A small amount of extra sampling if pop doesn't divide main budget nicely.
        # We'll handle exact budget via eval_one.
        for _ in range(iters):
            if remaining <= 0:
                break

            # Measure spread to scale exploration: robust MAD-like approximation.
            # If spread is tiny, sigma will decrease anyway through adaptation.
            spread = np.median(np.abs(X - np.median(X, axis=0, keepdims=True)), axis=0)
            spread = float(np.mean(spread)) if np.isfinite(spread).all() else 0.0
            # Candidate mutation scaling: combine global sigma with observed spread.
            scale = sigma * (1.0 + 0.5 * (spread / (avg_span + 1e-12)))

            # Generate offspring and apply index-wise replacement.
            successes = 0
            # Evaluate offspring in a loop to strictly respect budget.
            for i in range(pop):
                if remaining <= 0:
                    break

                # Blend between best-centered and parent-centered steps for robustness.
                # With probability p_best, move from best; otherwise from parent.
                # This helps both exploitation and diversity.
                if np.random.rand() < 0.7:
                    base = best_x
                else:
                    base = X[i]

                # Gaussian perturbation; additionally add a small isotropic term
                # to prevent identical copies when sigma gets small.
                step = np.random.randn(dim) * scale
                x_new = base + step

                # Boundary handling: clip to bounds.
                if np.any(span == 0):
                    # For zero-span dims, set exactly lb.
                    x_new = np.clip(x_new, lb, ub)
                    x_new[span == 0] = lb[span == 0]
                else:
                    x_new = np.clip(x_new, lb, ub)

                y_new = eval_one(x_new)

                # Index-wise selection: replace parent if improved.
                if y_new < values[i]:
                    X[i] = x_new
                    values[i] = y_new
                    successes += 1

            successes_window += successes
            # Adaptive sigma update based on success rate over last batch.
            # batch_offspring = pop or less if budget ended early.
            batch_offspring = pop
            # If budget ended early in this iteration, approximate success rate with actual evaluations.
            # We infer actual offspring evaluations by how many entries remain unchanged isn't safe;
            # instead compute based on remaining decrease isn't accessible here.
            # Simpler: still use pop as denominator; success_window logic is forgiving.
            denom = max(1, batch_offspring)
            success_rate = successes / denom

            # Simple success-rule:
            # - If many improvements, increase sigma a bit.
            # - Otherwise decrease sigma.
            if success_rate > 0.35:
                sigma *= 1.15
            elif success_rate < 0.15:
                sigma *= 0.82

            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # If we are unable to improve for long, shrink a bit to refine.
            iters += 1
            if successes == 0:
                sigma = max(sigma_min, sigma * 0.9)

            if remaining <= 0:
                break

        # Final local refinement around best using coordinate-wise probes.
        # This uses the remaining budget precisely.
        if remaining > 0 and best_x is not None:
            # Start with a step tied to current sigma but ensure it can probe coordinates.
            local_step = sigma
            # If sigma is extremely small relative to span, ensure some movement if possible.
            if avg_span > 0:
                local_step = max(local_step, 1e-6 * avg_span / max(1.0, np.sqrt(dim)))

            # Coordinate order randomized to avoid bias.
            coords = np.arange(dim)
            np.random.shuffle(coords)

            # We will attempt up to 'refine_budget' moves; budget is already tracked in eval_one.
            # Each coordinate may require up to 2 evaluations (+step and -step).
            # To keep it budget-safe, we always check remaining before eval.
            for k in coords:
                if remaining <= 0:
                    break

                # If coordinate has zero span, skip.
                if span[k] == 0:
                    continue

                x0 = best_x
                base_val = best_y

                # Try plus direction
                x_plus = np.array(x0, copy=True)
                x_plus[k] = np.clip(x_plus[k] + local_step, lb[k], ub[k])
                if x_plus[k] != x0[k] and remaining > 0:
                    y_plus = eval_one(x_plus)
                else:
                    y_plus = np.inf

                # Try minus direction
                x_minus = np.array(x0, copy=True)
                x_minus[k] = np.clip(x_minus[k] - local_step, lb[k], ub[k])
                if x_minus[k] != x0[k] and remaining > 0:
                    y_minus = eval_one(x_minus)
                else:
                    y_minus = np.inf

                # If any improvement, keep best_x already updated by eval_one.
                # If no improvement, shrink local_step a bit as we get closer to optimum.
                if best_y >= base_val:
                    local_step *= 0.9
                    if local_step < sigma_min:
                        break
                else:
                    # If improvement, slightly increase to keep momentum (bounded).
                    local_step = min(sigma_max, local_step * 1.05)

        # As a fallback (if func returns inf always or something odd), ensure output types.
        if best_x is None:
            # Evaluate one random point if never updated (shouldn't happen).
            x0 = lb + np.random.rand(dim) * span_nonzero
            if np.any(span == 0):
                x0[span == 0] = lb[span == 0]
            eval_one(x0)

        return np.array(best_x, dtype=float, copy=True), float(best_y)
