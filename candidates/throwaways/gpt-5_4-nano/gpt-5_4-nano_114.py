# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, robust black-box minimization strategy
# (a bandit-style Evolution Strategy with local coordinate refinement). It maintains
# a small population of candidate solutions and repeatedly proposes new points using
# Gaussian steps, occasionally switching to a local coordinate search around the best
# point found so far.
#
# Search state: The algorithm tracks a current best solution (best_x, best_y),
# the remaining evaluation budget, and dynamic step sizes (global and local),
# along with a "no-improvement" counter to trigger refinement and adjust exploration.
#
# Candidate generation: Each iteration samples offspring by adding Gaussian noise
# with scale tied to the current step size. A fraction of iterations uses a local
# proposal mechanism that perturbs one coordinate at a time near best_x to focus
# on promising directions.
#
# Selection and replacement: Offspring are evaluated and the algorithm applies
# (μ+λ)-style replacement: any offspring that improves upon the global best is used
# to update best_x/best_y, while the population mean (search center) moves slightly
# toward the best offspring.
#
# Adaptation: The algorithm adapts its step size based on whether improvements occur:
# on improvement it reduces step size a bit (favoring exploitation); on stagnation it
# increases step size (favoring exploration) and triggers coordinate refinement.
#
# Exploration mechanisms: Global Gaussian sampling explores the space. Step-size
# growth after stagnation helps escape local minima.
#
# Exploitation mechanisms: Local coordinate refinement samples along single dimensions
# around best_x with decreasing radii, improving accuracy near the current best.
#
# Boundary handling: All candidates are clipped to provided bounds (func.lower/upper
# or func.bounds.lb/ub). This ensures feasibility for box-constrained problems.
#
# Budget strategy: The algorithm strictly decrements evaluations per objective call and
# never exceeds the provided budget. It uses the remaining budget to decide how many
# evaluations to spend on each iteration.
#
# Closest known influences: The design is inspired by basic evolution strategies (ES),
# bandit-like step-size control, and common coordinate-search refinement used in
# black-box optimizers.
#
# Novelty or unusual aspects: The mix of (1) μ+λ global ES updates with (2) a lightweight
# coordinate refinement phase that adaptively activates based on stagnation is intended
# to be effective across many dimensions while remaining compact and budget-aware.
#
# Failure modes: If the function is extremely noisy, step-size adaptation may oscillate
# and coordinate refinement can waste evaluations. If bounds are very tight or the
# optimum lies on a boundary, clipping can reduce effective gradient information;
# however repeated sampling around best_x still allows progress.
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
            # No evaluations allowed; return a deterministic point.
            lb, ub = self._get_bounds(func, dim)
            best_x = np.array(lb, dtype=float)
            if np.any(~np.isfinite(best_x)) or np.any(~np.isfinite(ub)):
                best_x = np.zeros(dim, dtype=float)
            best_x = self._clip(best_x, lb, ub)
            return best_x, float("inf")

        lb, ub = self._get_bounds(func, dim)
        lb = np.asarray(lb, dtype=float).reshape(dim)
        ub = np.asarray(ub, dtype=float).reshape(dim)

        # Robust initialization: sample starting point uniformly inside bounds.
        rng = np.random
        x0 = lb + rng.rand(dim) * (ub - lb)
        x0 = self._clip(x0, lb, ub)

        # Track evaluations.
        evals = 0

        def eval_point(x):
            nonlocal evals
            if evals >= budget:
                return float("inf")
            x = np.asarray(x, dtype=float)
            y = func(x)
            evals += 1
            return float(y)

        best_x = x0.copy()
        best_y = eval_point(best_x)

        # If budget allows, also sample a small initial population around x0.
        # Choose population size based on dimension but keep compact.
        mu = 1
        lam = int(np.clip(4 + dim, 6, 18))  # offspring per macro-iteration
        # Ensure we never overspend.
        lam = max(2, min(lam, budget - 1)) if budget > 1 else 1

        # Initial step size: fraction of bounds range.
        range_ = ub - lb
        # Handle degenerate bounds: avoid zero step sizes.
        global_scale = 0.25 * np.where(range_ > 0, range_, 1.0)
        # Also cap to a reasonable magnitude to avoid too-large jumps.
        global_scale = np.maximum(global_scale, 1e-12)
        global_scale = np.minimum(global_scale, 1e6)

        # Local refinement scale (coordinate steps).
        local_scale = global_scale * 0.5

        no_improve = 0
        # Number of macro-iterations depends on remaining budget.
        # We'll loop until budget is exhausted.
        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break

            # Determine how many offspring we can afford.
            k = min(lam, remaining)

            # Adapt step size based on stagnation.
            # On improvement, shrink a bit; on stagnation, expand.
            if no_improve > 0:
                # If stagnating, explore more aggressively.
                shrink = np.power(0.98, 0.5)
                expand = np.power(1.15, min(no_improve, 8))
                step_scale = global_scale * expand
            else:
                step_scale = global_scale * 0.98

            # Compute a "search center" near the best point.
            # This also helps in constrained spaces due to clipping.
            center = best_x

            # Generate offspring: Gaussian mutations.
            # Candidate generation: x_i = center + step_scale * N(0, I)
            # Using diagonal scaling for simplicity and robustness.
            Z = rng.randn(k, dim)
            X = center[None, :] + Z * step_scale[None, :]

            # Boundary handling: clip all candidates to bounds.
            X = self._clip(X, lb, ub)

            # Evaluate offspring and update best.
            improved = False
            best_off_y = best_y
            best_off_x = best_x

            for i in range(k):
                y = eval_point(X[i])
                if y < best_off_y:
                    best_off_y = y
                    best_off_x = X[i].copy()
                    improved = True

            if improved:
                best_x = best_off_x
                best_y = best_off_y
                no_improve = 0
                # Move global scale slightly smaller to exploit.
                global_scale = np.maximum(global_scale * 0.85, 1e-12)
                local_scale = np.maximum(local_scale * 0.85, 1e-12)
            else:
                no_improve += 1
                # Step-size increases gradually with stagnation.
                global_scale = np.minimum(global_scale * 1.08, (ub - lb + 1.0) * 2.0)
                local_scale = np.minimum(local_scale * 1.06, (ub - lb + 1.0) * 2.0)

            # Exploration/exploitation switch: occasionally perform coordinate refinement.
            # Trigger when we have not improved recently and budget still remains.
            if no_improve >= 3 and evals < budget:
                # Use a small coordinate sampling budget.
                # Allocate remaining evaluations conservatively: 5% or up to dim,
                # but at least 2 evaluations and not exceeding remaining.
                remaining = budget - evals
                coord_trials = int(np.clip(max(2, dim // 2), 2, 6 + dim // 2))
                coord_trials = min(coord_trials, remaining)

                # Choose coordinates: sample a subset of dimensions, focusing on
                # those with larger available range (more room to move).
                range_eff = np.abs(ub - lb)
                # If all ranges are near zero, just keep all identical.
                if np.all(range_eff <= 1e-15):
                    coords = rng.choice(dim, size=min(dim, coord_trials), replace=False)
                else:
                    weights = range_eff / (np.sum(range_eff) + 1e-30)
                    # Weighted choice without replacement isn't directly in numpy for all versions.
                    # We'll approximate by using multinomial and unique filtering.
                    picks = []
                    while len(picks) < min(dim, coord_trials) and len(picks) < dim:
                        c = int(rng.choice(dim, p=weights))
                        if c not in picks:
                            picks.append(c)
                    coords = np.array(picks, dtype=int)

                # Coordinate refinement: try +/- steps with shrinking radii.
                # This is exploitation around best_x.
                # We ensure we evaluate exactly coord_trials points.
                # Each coordinate can contribute up to 2 evaluations.
                trial_points = []
                radii = np.array([0.9, 0.45, 0.2], dtype=float)
                # Build proposals deterministically in order of radii.
                for c in coords:
                    if len(trial_points) >= coord_trials:
                        break
                    for r in radii:
                        if len(trial_points) >= coord_trials:
                            break
                        step = local_scale[c] * r
                        trial_points.append(best_x.copy() + self._unit_dir(c, dim) * step)
                        if len(trial_points) >= coord_trials:
                            break
                        trial_points.append(best_x.copy() - self._unit_dir(c, dim) * step)

                # Evaluate trial points with budget awareness.
                improved_any = False
                for x in trial_points[:coord_trials]:
                    x = self._clip(x, lb, ub)
                    y = eval_point(x)
                    if y < best_y:
                        best_y = y
                        best_x = x.copy()
                        improved_any = True

                if improved_any:
                    # After refinement improvement, reset stagnation and reduce scales.
                    no_improve = 0
                    global_scale = np.maximum(global_scale * 0.7, 1e-12)
                    local_scale = np.maximum(local_scale * 0.7, 1e-12)
                else:
                    # If refinement didn't help, increase global exploration.
                    global_scale = np.minimum(global_scale * 1.12, (ub - lb + 1.0) * 2.0)
                    local_scale = np.minimum(local_scale * 1.08, (ub - lb + 1.0) * 2.0)

        return best_x, best_y

    @staticmethod
    def _get_bounds(func, dim):
        # Accept either func.lower/func.upper or func.bounds.lb/func.bounds.ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
            if lb.size == dim and ub.size == dim:
                return lb, ub

        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float).reshape(-1)
                ub = np.asarray(b.ub, dtype=float).reshape(-1)
                if lb.size == dim and ub.size == dim:
                    return lb, ub

        # Fallback: unconstrained. Use a default box around 0.
        # (Still returned as finite bounds to keep clipping well-defined.)
        lb = -5.0 * np.ones(dim, dtype=float)
        ub = 5.0 * np.ones(dim, dtype=float)
        return lb, ub

    @staticmethod
    def _clip(x, lb, ub):
        # Supports x being shape (dim,) or (n, dim).
        x = np.asarray(x, dtype=float)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        return np.minimum(np.maximum(x, lb), ub)

    @staticmethod
    def _unit_dir(c, dim):
        v = np.zeros(dim, dtype=float)
        v[c] = 1.0
        return v
