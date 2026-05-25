# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimization
# algorithm (a simple evolution strategy with adaptive step size and occasional
# coordinate-wise refinement). It is designed to be robust across dimensions,
# works with arbitrary bounded continuous search spaces, and uses only numpy.
#
# Search state: The algorithm maintains a current best point x_best, its objective
# value y_best, and an isotropic mutation step size sigma. It also tracks the
# remaining evaluation budget and uses a deterministic budget accounting strategy
# to never exceed it.
#
# Candidate generation: Each generation samples several offspring by adding
# Gaussian noise to the current best point: x = x_best + sigma * N(0, I).
# A small fraction of evaluations are reserved for occasional refinement:
# it perturbs one coordinate at a time (with step proportional to sigma) to
# help escape situations where isotropic search is slow to align with the
# objective's sensitivity.
#
# Selection and replacement: After evaluating offspring, the best candidate found
# in the generation replaces x_best if it improves y_best. The best point is
# always kept (elitism). The algorithm never discards improvements.
#
# Adaptation: Sigma is adapted based on the number of successful offspring:
# if improvements are frequent, sigma increases mildly; if improvements are rare,
# sigma decreases, encouraging exploitation near optima while still exploring.
#
# Exploration mechanisms: Gaussian isotropic sampling provides global exploration
# early. The coordinate-wise refinement introduces structured local exploration
# that can be helpful for ill-conditioned problems.
#
# Exploitation mechanisms: The algorithm always centers mutations on the current
# best point and gradually shrinks sigma after unsuccessful generations, focusing
# sampling locally.
#
# Boundary handling: Candidates are clipped to the provided bounds before
# evaluation, ensuring feasibility for bounded objectives.
#
# Budget strategy: All objective calls are counted and strictly capped. The number
# of evaluations per "round" is computed from the remaining budget. The algorithm
# terminates immediately when the budget reaches zero.
#
# Closest known influences: Inspired by (1+lambda)-ES and CMA-less step-size control,
# combined with occasional coordinate perturbation as a lightweight local search.
# Novelty: The coordinate-wise refinement is budget-aware and interleaves with
# isotropic ES updates to improve performance on axis-aligned or separable
# problems.
#
# Failure modes: If the objective is extremely noisy, step-size adaptation may
# oscillate. If the optimum lies in a very narrow region with extremely small
# scale, sigma adaptation might shrink too slowly; clipping could also reduce
# effective exploration near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget

        # Read bounds from func (supports either func.lower/func.upper or func.bounds.lb/ub)
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective must provide bounds via (lower, upper) or bounds.lb/bounds.ub.")

        if lb.shape == () and ub.shape == ():
            lb = np.full(d, float(lb))
            ub = np.full(d, float(ub))
        else:
            lb = np.broadcast_to(lb, (d,)).astype(float)
            ub = np.broadcast_to(ub, (d,)).astype(float)

        # Ensure valid bounds order (robustness).
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        width = hi - lo

        def clamp(x):
            return np.minimum(hi, np.maximum(lo, x))

        # If budget is very small, just sample one point within bounds.
        evals = 0

        def eval_obj(x):
            nonlocal evals
            if evals >= budget:
                # Hard guard (should not happen if budget accounting is correct)
                return np.inf
            y = float(func(x))
            evals += 1
            return y

        # Initialize x_best by a single random point; also set sigma based on bounds scale.
        x_best = clamp(lo + (hi - lo) * np.random.rand(d))
        y_best = eval_obj(x_best)

        # Step-size: fraction of domain width, with safe minimum.
        # If width is zero in a dimension, that dimension is constant anyway.
        domain_scale = np.max(width) if d > 0 else 1.0
        sigma = 0.25 * domain_scale
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        # Strategy parameters (kept small/compact; tuned to be reasonable).
        base_lambda = 4 + int(3 * np.log(d + 1.0))  # offspring per round (approx.)
        min_sigma = 1e-12 * (domain_scale if domain_scale > 0 else 1.0)
        success_target = max(1, int(0.2 * base_lambda))  # desired number of improvements

        while evals < budget:
            remaining = budget - evals
            lam = min(base_lambda, remaining)

            # Split budget: isotropic offspring mostly, but reserve a few for coordinate refinement.
            # If dim is 1, coordinate refinement is the same as isotropic but with axis-aligned steps.
            frac_coord = 0.15
            lam_coord = int(max(0, min(lam, np.ceil(frac_coord * lam))))
            lam_iso = lam - lam_coord

            # Generate isotropic offspring around current best.
            improved_any = False
            best_round_x = x_best.copy()
            best_round_y = y_best

            successes = 0

            if lam_iso > 0:
                # Create lam_iso samples in one shot to reduce Python overhead.
                noise = np.random.randn(lam_iso, d)
                # Use a scaling that accounts for average bound widths, lightly.
                # (Still isotropic: this is robust across dimensions.)
                step = sigma
                candidates = clamp(x_best + step * noise)

                # Evaluate all; select best.
                for i in range(lam_iso):
                    y = eval_obj(candidates[i])
                    if y < best_round_y:
                        best_round_y = y
                        best_round_x = candidates[i].copy()
                        successes += 1
                        improved_any = True

            # Occasional coordinate-wise refinement (budget-aware).
            if lam_coord > 0 and evals < budget:
                # Choose coordinates biased toward larger widths (heuristic).
                widths = width.copy()
                # If all widths are zero, refinement is irrelevant but harmless.
                if np.all(widths <= 0):
                    coords = np.zeros(lam_coord, dtype=int)
                else:
                    probs = widths / (np.sum(widths) + 1e-300)
                    coords = np.random.choice(d, size=lam_coord, replace=True, p=probs)

                for j in range(lam_coord):
                    if evals >= budget:
                        break
                    c = int(coords[j])
                    # Try positive and negative step with a random sign.
                    sign = 1.0 if np.random.rand() < 0.5 else -1.0
                    x = x_best.copy()
                    # Smaller coordinate step than isotropic to refine locally.
                    coord_step = sigma * (0.5 + 0.5 * np.random.rand())
                    x[c] = x[c] + sign * coord_step
                    x = clamp(x)
                    y = eval_obj(x)
                    if y < best_round_y:
                        best_round_y = y
                        best_round_x = x.copy()
                        successes += 1
                        improved_any = True

            # Selection / elitism
            if best_round_y < y_best:
                x_best = best_round_x
                y_best = best_round_y

            # Adapt sigma (simple 1/5th-like rule but smoothed).
            # If many successes, increase; else decrease.
            if lam > 0:
                # Compare successes against a target.
                # Clamp success ratio to avoid extreme updates.
                success_ratio = successes / float(lam)
                if success_ratio >= 0.25:
                    sigma *= 1.18
                elif success_ratio <= 0.10:
                    sigma *= 0.82
                else:
                    sigma *= 0.95

            sigma = float(max(min_sigma, sigma))

            # Optional early stop: if sigma is effectively zero or no domain variation.
            if sigma <= min_sigma or (domain_scale <= 0 and np.all(width <= 0)):
                break

        return x_best, y_best
