# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# based on an iterative variant of random-restart coordinate descent with
# local reflection and adaptive step scaling. It maintains a small set of
# candidate points and repeatedly improves the current best using coordinate
# probes, plus occasional larger random jumps for exploration.
# Search state: Tracks current best position/value, a step-size vector
# (common scale shared across coordinates), and a small memory of recent
# best points to stabilize progress. Also tracks remaining evaluation budget
# and ensures exactly (or fewer) objective calls than allowed.
# Candidate generation: Each iteration samples candidate points by:
#   1) Probing along a subset of coordinates using ±step moves from the best.
#   2) Performing a reflection around the best (best + (best - x) style)
#      to escape shallow local minima.
#   3) Occasionally injecting a random restart around the best with a
#      decaying global radius.
# Selection and replacement: Any candidate that improves the best value
# replaces the best. If no improvement is found, the step size is reduced.
# A small rolling memory of best points can also trigger a restart-like
# region if progress stalls.
# Adaptation: Step scale decays after unsuccessful iterations and can increase
# slightly after consistent improvements, encouraging both refinement and
# occasional escape.
# Exploration mechanisms: Random coordinate subset probing, reflection moves,
# and periodic larger random perturbations.
# Exploitation mechanisms: Intensive coordinate-wise local probing around the
# best and aggressive step reduction upon stagnation.
# Boundary handling: Any candidate is clipped to the provided bounds
# (func.lower/upper or func.bounds.lb/ub). This keeps points valid.
# Budget strategy: The algorithm computes an iteration budget from the total
# allowed evaluations and always checks before each objective call. It stops
# when the budget is exhausted.
# Closest known influences: Similar in spirit to coordinate descent with
# reflection/random restarts (a common pattern in derivative-free optimizers).
# Novelty or unusual aspects: Uses a unified step-size vector, combines
# coordinate probing with reflection proposals around both best and a short
# memory of prior bests, and dynamically adjusts step scaling based on
# success frequency while strictly enforcing the evaluation budget.
# Failure modes: If the objective is highly discontinuous or extremely
# ill-conditioned, coordinate probing may miss improvements; in such cases
# exploration restarts can help but performance is not guaranteed.
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

        # Read bounds from either func.lower/func.upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            b = getattr(func, "bounds", None)
            if b is None or not hasattr(b, "lb") or not hasattr(b, "ub"):
                raise AttributeError("func must provide bounds via lower/upper or bounds.lb/bounds.ub")
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)

        lb = np.broadcast_to(lb, (dim,)).copy()
        ub = np.broadcast_to(ub, (dim,)).copy()
        if np.any(ub <= lb):
            raise ValueError("Invalid bounds: each ub must be > lb")

        rng = np.random

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        # Evaluate objective while strictly staying within budget
        evals = 0

        def eval_obj(x):
            nonlocal evals
            if evals >= budget:
                # Should not happen if checks are correct, but guard anyway.
                raise RuntimeError("Evaluation budget exceeded")
            evals += 1
            return float(func(x))

        # Initialize with a random point and a couple of cheap random candidates
        # to get a reasonable starting best.
        span = ub - lb
        x0 = lb + rng.rand(dim) * span
        best_x = clip(x0)
        best_y = eval_obj(best_x)

        # Memory of recent best points (for reflection around stability region)
        mem_x = [best_x.copy()]
        mem_y = [best_y]

        # Initial step size: fraction of the domain, but not too tiny
        step_scale = 0.25
        step_vec = step_scale * span

        # Choose number of iterations based on budget and per-iteration cost.
        # Each iteration uses up to:
        #   - 2 * k coordinate probes (±step for k coordinates)
        #   - a small number of extra proposals (reflection and occasional restart)
        # We'll adaptively stop when near budget.
        # Start with conservative plan.
        max_iters = max(1, budget // max(1, 2 * min(dim, 8) + 4))

        # Helper: propose coordinate moves around a base point
        def coordinate_probes(base_x, base_y, k):
            nonlocal best_x, best_y
            # Choose a subset of coordinates to probe
            # Ensure deterministic size <= dim
            k = int(min(max(1, k), dim))
            coords = rng.choice(dim, size=k, replace=False)

            improved = False
            for j in coords:
                if evals >= budget:
                    return improved
                # +/- step along coordinate j
                # Use sign selected randomly to reduce bias
                s = step_vec[j]
                # Two candidates, but we may stop early if budget nearly exhausted
                for sign in (1.0, -1.0):
                    if evals >= budget:
                        return improved
                    x = base_x.copy()
                    x[j] = base_x[j] + sign * s
                    y = eval_obj(clip(x))
                    if y < best_y:
                        best_y, best_x = y, x
                        improved = True
                        # Update memory occasionally
                        if len(mem_x) >= 5:
                            mem_x.pop(0)
                            mem_y.pop(0)
                        mem_x.append(best_x.copy())
                        mem_y.append(best_y)
            return improved

        # Main optimization loop
        for it in range(max_iters):
            if evals >= budget:
                break

            # Stagnation check: if no improvement for some time, try reflection/restart
            # We infer by comparing current best against memory minima.
            recent_best = min(mem_y) if mem_y else best_y
            stagnating = (best_y >= recent_best - 1e-15)

            # Determine how many coordinates to probe this iteration
            # Early iterations probe more coordinates.
            frac = 1.0 - (it / max_iters)
            k = int(max(1, min(dim, 2 + int(6 * frac))))
            # Also add a small chance to probe fewer to save evaluations
            if rng.rand() < 0.15:
                k = max(1, k - 1)

            # Exploit: coordinate probes around best
            improved = False
            improved = coordinate_probes(best_x, best_y, k)

            if evals >= budget:
                break

            # Reflection mechanism to escape local minima / plateaus
            # Reflect around best using a prior best (if available) or random point.
            if stagnating or (rng.rand() < 0.35):
                if rng.rand() < 0.7 and len(mem_x) >= 1:
                    base_ref = mem_x[-1]
                else:
                    base_ref = lb + rng.rand(dim) * span

                if evals < budget:
                    # Create reflection-like candidate: best + (best - base_ref)
                    x = best_x + (best_x - base_ref)
                    # Scale reflection magnitude by step size
                    # to avoid too aggressive jumps.
                    x = clip(best_x + 0.7 * (x - best_x))
                    y = eval_obj(x)
                    if y < best_y:
                        best_y = y
                        best_x = x
                        improved = True
                        mem_x.append(best_x.copy())
                        mem_y.append(best_y)
                        if len(mem_x) > 5:
                            mem_x.pop(0)
                            mem_y.pop(0)
                if evals >= budget:
                    break

            # Occasional random exploration / restart around best
            # Radius decays with progress.
            if rng.rand() < (0.25 if dim > 1 else 0.15) or stagnating:
                # Global radius decays as we approach the end (but not to zero).
                global_radius = (0.35 + 0.15 * rng.rand()) * (1.0 - it / max_iters)
                global_radius = max(0.05, global_radius)
                if evals < budget:
                    # Sample from a ball-like distribution using gaussian then normalize
                    z = rng.randn(dim)
                    z = z / (np.linalg.norm(z) + 1e-12)
                    # Mix with uniform noise to avoid direction collapse
                    u = rng.rand(dim) - 0.5
                    u = u / (np.linalg.norm(u) + 1e-12)
                    dir_vec = 0.7 * z + 0.3 * u
                    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)

                    # Perturbation magnitude
                    radius = global_radius * float(np.mean(step_vec))
                    x = clip(best_x + dir_vec * radius)
                    y = eval_obj(x)
                    if y < best_y:
                        best_y = y
                        best_x = x
                        improved = True
                        mem_x.append(best_x.copy())
                        mem_y.append(best_y)
                        if len(mem_x) > 5:
                            mem_x.pop(0)
                            mem_y.pop(0)

            if evals >= budget:
                break

            # Adapt step size based on success
            if improved:
                # Mildly increase step for continued search; cap for safety
                step_scale *= 1.08
                step_scale = min(step_scale, 0.75)
            else:
                # Decrease step for more local refinement
                step_scale *= 0.72
                step_scale = max(step_scale, 1e-6)

            step_vec = step_scale * span

            # If step is extremely small relative to domain, force a restart-like move
            # to avoid getting stuck due to numerical scale.
            if step_scale < 1e-5 and evals < budget:
                x = lb + rng.rand(dim) * span
                y = eval_obj(clip(x))
                if y < best_y:
                    best_x, best_y = x, y
                    mem_x = [best_x.copy()]
                    mem_y = [best_y]
                # Reset step scale
                step_scale = 0.25
                step_vec = step_scale * span

        return best_x, best_y
