# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer based on an
# evolution strategy (ES) with restart and local refinement using multiple
# randomly sampled Gaussian steps around an incumbent.
# Search state: Maintains a current best solution (incumbent) and tracks the
# remaining evaluation budget. The search step size (sigma) controls exploration.
# Candidate generation: Each generation samples a population of candidates as
# incumbent + sigma * N(0, I). After an initial broad phase, it periodically
# performs a small local refinement by probing one-step and two-step
# perturbations around the incumbent.
# Selection and replacement: Evaluates all candidates in the population, takes
# the best (lowest objective value) as the next incumbent. Also uses a simple
# success-based rule to adjust sigma.
# Adaptation: Uses a success-rate heuristic: if the best candidate improves the
# incumbent, sigma is decreased (more exploitation) or increased depending on
# the refinement phase; otherwise sigma is increased to encourage exploration.
# Exploration mechanisms: Population sampling with an adaptive sigma and random
# restart when progress stalls.
# Exploitation mechanisms: Local refinement probes with smaller sigmas around
# the incumbent.
# Boundary handling: Uses vectorized clipping to keep candidates within the
# provided bounds after every mutation step.
# Budget strategy: Never exceeds the provided evaluation budget; every evaluation
# consumes one budget unit. Population size and number of generations are
# computed to fit within the budget.
# Closest known influences: Inspired by canonical evolution strategies (ES) and
# (1+1)-style adaptive random search, combined with restarts and local probing.
# Novelty or unusual aspects: The code blends a global ES sampling phase with
# periodic local “one/two-step” refinement probes, using a lightweight,
# budget-aware scheduling.
# Failure modes: In very high-dimensional problems with extremely narrow or
# deceptive landscapes, progress may stall; restarts and sigma adaptation help,
# but performance is not guaranteed.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = max(0, int(self.budget))
        if budget == 0:
            # Best effort return: choose mid-point if bounds exist, else zeros.
            lower, upper = _get_bounds(func, dim)
            x0 = (lower + upper) * 0.5
            return x0.astype(float), float("inf")

        lower, upper = _get_bounds(func, dim)
        span = np.maximum(upper - lower, 1e-12)
        # Start at a random point in bounds (or mid-point if random fails).
        x_best = lower + span * np.random.rand(dim)
        y_best = float("inf")

        # Evaluate incumbent first to establish baseline.
        evals_used = 0
        x_best = _clip(x_best, lower, upper)
        y_best = _safe_eval(func, x_best)
        evals_used += 1

        # If incumbent is already extremely good, still must respect budget.
        if evals_used >= budget:
            return x_best, y_best

        # Budget-aware ES parameters
        # Population size chosen to be small enough for budget compliance.
        # It scales gently with dimension but stays bounded.
        lam = int(np.clip(4 + dim // 10, 8, 32))
        lam = min(lam, budget - evals_used) if budget - evals_used > 0 else 1
        # Sigma starts as a fraction of the search span.
        sigma0 = 0.2 * float(np.mean(span))
        sigma = max(1e-12, sigma0)

        # Restart/phase scheduling
        # Number of generations roughly fits budget considering lam evaluations each.
        remaining = budget - evals_used
        gens = max(1, remaining // lam)  # number of full population generations
        # Local refinement probes per restart cycle
        refine_every = 3
        max_restarts = 2  # keep compact
        restarts_done = 0

        best_global_x = x_best.copy()
        best_global_y = y_best
        no_improve_gens = 0
        max_no_improve = 4

        # Main loop: generations of (mu=1, lambda) ES with adaptive sigma.
        for g in range(gens):
            if evals_used >= budget:
                break

            # Candidate generation: incumbent + sigma * N(0, I)
            # We sample a whole population to leverage vectorization.
            Z = np.random.randn(lam, dim)
            X = x_best[None, :] + sigma * Z
            X = _clip(X, lower, upper)

            # Evaluate candidates sequentially (func may not support vectorization).
            # Still keeps overhead minimal.
            y_pop = np.empty(lam, dtype=float)
            for i in range(lam):
                if evals_used >= budget:
                    y_pop = y_pop[:i]
                    X = X[:i]
                    lam = i
                    break
                y_pop[i] = _safe_eval(func, X[i])
                evals_used += 1

            if lam <= 0:
                break

            # Selection: pick best candidate (minimization)
            idx = int(np.argmin(y_pop))
            x_candidate = X[idx]
            y_candidate = float(y_pop[idx])

            # Replacement: update incumbent if improved
            improved = y_candidate < y_best
            if improved:
                x_best = x_candidate
                y_best = y_candidate
                if y_best < best_global_y:
                    best_global_y = y_best
                    best_global_x = x_best.copy()
                no_improve_gens = 0
                # Exploit more after success
                sigma *= 0.85
            else:
                no_improve_gens += 1
                # Explore more after failure
                sigma *= 1.08

            # Optional local refinement: probe one-step and two-step around incumbent
            # using smaller sigmas. Triggered periodically or after stalls.
            do_refine = (g % refine_every == refine_every - 1) or (no_improve_gens >= 2)
            if do_refine and evals_used < budget:
                # Probes consume remaining budget but never exceed it.
                # Probe count is small to keep evaluation cost predictable.
                probe_count = min(6, budget - evals_used)
                if probe_count > 0:
                    # One-step probes
                    sigma_local_1 = max(1e-12, 0.05 * float(np.mean(span)) * (0.6 ** restarts_done))
                    # Two-step probes (larger variety)
                    sigma_local_2 = max(1e-12, 0.10 * float(np.mean(span)) * (0.6 ** restarts_done))

                    for k in range(probe_count):
                        step_sigma = sigma_local_1 if (k % 2 == 0) else sigma_local_2
                        u = np.random.randn(dim)
                        x_probe = x_best + step_sigma * u
                        x_probe = _clip(x_probe, lower, upper)
                        y_probe = _safe_eval(func, x_probe)
                        evals_used += 1
                        if y_probe < y_best:
                            x_best = x_probe
                            y_best = y_probe
                            if y_best < best_global_y:
                                best_global_y = y_best
                                best_global_x = x_best.copy()
                            # Successful local move: tighten sigma further
                            sigma *= 0.9
                            no_improve_gens = 0

                        if evals_used >= budget:
                            break

            # Restart if stalled: sample fresh incumbent and reset sigma.
            if no_improve_gens >= max_no_improve and restarts_done < max_restarts and evals_used < budget:
                restarts_done += 1
                no_improve_gens = 0
                # Fresh restart point
                x_best = lower + span * np.random.rand(dim)
                x_best = _clip(x_best, lower, upper)
                y_best = _safe_eval(func, x_best)
                evals_used += 1
                sigma = max(1e-12, 0.25 * float(np.mean(span)))
                if y_best < best_global_y:
                    best_global_y = y_best
                    best_global_x = x_best.copy()

        return best_global_x, best_global_y


def _get_bounds(func, dim):
    """
    Returns (lower, upper) as float numpy arrays of shape (dim,).
    Supports:
      - func.lower / func.upper
      - func.bounds.lb / func.bounds.ub
      - func.bounds is expected to have lb/ub
    """
    lower = None
    upper = None

    if hasattr(func, "lower") and hasattr(func, "upper"):
        lower = np.asarray(getattr(func, "lower"), dtype=float)
        upper = np.asarray(getattr(func, "upper"), dtype=float)
    elif hasattr(func, "bounds"):
        b = getattr(func, "bounds")
        if hasattr(b, "lb") and hasattr(b, "ub"):
            lower = np.asarray(getattr(b, "lb"), dtype=float)
            upper = np.asarray(getattr(b, "ub"), dtype=float)
    # Fallback: try to read attributes directly from func (if func is an object wrapper)
    if lower is None or upper is None:
        # As a last resort, use [-5, 5] box; should rarely happen in proper harnesses.
        lower = -5.0 * np.ones(dim, dtype=float)
        upper = 5.0 * np.ones(dim, dtype=float)

    lower = np.broadcast_to(lower, (dim,)).astype(float, copy=False).copy()
    upper = np.broadcast_to(upper, (dim,)).astype(float, copy=False).copy()
    # Ensure valid order
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    # Avoid degenerate span issues; still allow fixed variables.
    return lo, hi


def _clip(x, lower, upper):
    # Supports both vector (dim,) and matrix (n, dim)
    return np.minimum(np.maximum(x, lower), upper)


def _safe_eval(func, x):
    # Objective is minimization; assume func(x) returns scalar-like.
    y = func(x)
    # Convert to float robustly (handles numpy scalar, Python float, etc.)
    return float(np.asarray(y).item())
