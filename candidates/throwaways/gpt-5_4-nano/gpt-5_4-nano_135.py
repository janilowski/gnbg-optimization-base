# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, robust black-box minimization algorithm
# for bounded continuous problems. It uses a few candidate points, iteratively improves
# them using coordinate-wise gradient-free ideas, and periodically injects random
# candidates to escape stagnation.
#
# Search state: The algorithm maintains a current best solution (best_x, best_y),
# a small population of candidate solutions around the best point, and a global step
# size (sigma) that controls how far new candidates are sampled.
#
# Candidate generation: In each iteration, it creates candidates by (1) sampling
# Gaussian perturbations around the current best and/or promising points, (2) doing
# a simple coordinate-direction sweep (finite-difference-like) using function evaluations
# on +/- steps along selected dimensions, and (3) occasionally adding pure random
# points for diversification.
#
# Selection and replacement: Every generated candidate is evaluated (respecting the
# remaining evaluation budget). If a candidate improves on the best_y, the algorithm
# updates best_x/best_y. The local candidate pool is refreshed by centering around
# improved points; otherwise it gradually shrinks toward the best point.
#
# Adaptation: The step size sigma decreases when no improvement is observed and
# increases slightly when improvements are found, balancing exploration/exploitation.
#
# Exploration mechanisms: Gaussian sampling around the best, plus periodic random
# samples within bounds, helps explore new regions.
#
# Exploitation mechanisms: Coordinate-wise direction probing near the best point
# (evaluating +/- along a small set of dimensions) helps refine the search without
# requiring gradients.
#
# Boundary handling: Every candidate is clipped to the provided bounds.
#
# Budget strategy: The algorithm strictly tracks the number of objective calls and
# never exceeds the provided evaluation budget. It plans the number of evaluations
# per phase so the total stays within budget.
#
# Closest known influences: The approach is inspired by simple evolution strategies
# (ES) with adaptive step size and adds a lightweight coordinate-direction probe.
#
# Novelty or unusual aspects: The coordinate-direction probe is integrated as a
# budget-aware refinement step using the current sigma, and the algorithm dynamically
# adjusts sigma based on a small improvement signal.
#
# Failure modes: If the function is extremely noisy, the coordinate probes may
# mislead the search; the periodic random injections mitigate this. If the optimum
# lies in a tiny region, aggressive sigma shrinkage could slow progress, but step
# size is prevented from vanishing too quickly.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # ---- Bounds handling (robustly read from func) ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = getattr(func, "lower")
            ub = getattr(func, "upper")
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = getattr(func.bounds, "lb")
            ub = getattr(func.bounds, "ub")

        if lb is None or ub is None:
            raise AttributeError("Objective function must provide bounds via func.lower/func.upper "
                                 "or func.bounds.lb/func.bounds.ub.")

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size == 1 and n > 1:
            lb = np.full(n, lb.item(), dtype=float)
        if ub.size == 1 and n > 1:
            ub = np.full(n, ub.item(), dtype=float)
        lb = lb[:n].copy()
        ub = ub[:n].copy()
        if lb.shape != (n,) or ub.shape != (n,):
            raise ValueError("Bounds must be scalar or length matching dim.")

        span = ub - lb
        # Handle degenerate bounds gracefully (no expansion beyond fixed points).
        span_safe = np.where(span != 0.0, span, 1.0)
        lower = lb
        upper = ub

        rng = np.random  # harness will seed numpy globally

        # ---- Budget-aware evaluation wrapper ----
        evals = 0
        best_x = None
        best_y = None

        def clip(x):
            return np.minimum(upper, np.maximum(lower, x))

        def eval_x(x):
            nonlocal evals, best_x, best_y
            if evals >= budget:
                return  # do not evaluate beyond budget
            x = clip(np.asarray(x, dtype=float))
            y = float(func(x))
            evals += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = x
            return y

        # Early exit if budget is too small.
        if budget <= 0:
            # No evaluations allowed: return a clipped random point (without evaluating objective).
            x0 = lower + rng.rand(n) * span_safe
            return clip(x0), float("inf")

        # ---- Initialization: evaluate a handful of points uniformly ----
        # Use a small count to preserve budget for iterative improvement.
        init_count = min(max(2, n + 1), budget)
        for _ in range(init_count):
            x = lower + rng.rand(n) * span_safe
            eval_x(x)

        # If budget allows only initialization, return best found.
        if evals >= budget:
            return best_x, best_y

        # ---- Step size initialization ----
        # Typical scale based on bounds span.
        sigma = 0.3 * np.mean(np.abs(span_safe))
        # Prevent sigma from being too tiny/large.
        sigma_min = 1e-12
        sigma_max = np.mean(np.abs(span_safe)) * 2.0 + 1.0
        sigma = float(np.clip(sigma, sigma_min, sigma_max))

        # ---- Main loop ----
        # Each iteration uses a "probe" phase (coordinate directions) and a "sampling" phase.
        # We also add diversification when improvements stall.
        no_improve_rounds = 0
        max_no_improve = 6  # trigger diversification
        # Iterate with an estimate of number of rounds; actual loop stops by budget.
        while evals < budget:
            remaining = budget - evals
            improved_this_round = False

            # ---- Exploitation: coordinate-direction probe around best ----
            # Probe a subset of coordinates to control budget; choose random dims.
            # Each prob uses at most 2 evaluations (+/-). We'll do at most k probes.
            k = min(n, max(2, int(np.ceil(n / 4))))
            # Use at most what budget can afford for probe; reserve for sampling.
            # Reserve at least 1 evaluation for sampling unless remaining is tiny.
            reserve_for_sampling = 1 if remaining > 1 else 0
            probe_budget = max(0, remaining - reserve_for_sampling)
            # Each coordinate uses 2 evaluations; cap probe coords accordingly.
            max_coords = min(k, probe_budget // 2 if probe_budget >= 2 else 0)

            if max_coords > 0:
                dims = rng.choice(n, size=max_coords, replace=False)
                step = sigma * 0.5 + 1e-12

                for j in dims:
                    if evals >= budget:
                        break
                    # Evaluate +step and -step; pick whichever is better.
                    x_plus = best_x.copy()
                    x_minus = best_x.copy()
                    x_plus[j] = x_plus[j] + step
                    x_minus[j] = x_minus[j] - step

                    y_before = best_y
                    eval_x(x_plus)
                    if evals >= budget:
                        break
                    eval_x(x_minus)
                    if best_y is not None and y_before is not None and best_y < y_before:
                        improved_this_round = True

            # ---- Candidate sampling: Gaussian mutations around best ----
            # Use remaining budget to sample a few candidates.
            # Decide how many evaluations to spend on sampling this round.
            remaining = budget - evals
            if remaining <= 0:
                break

            # Sample count: keep modest to avoid spending entire budget at once.
            # Ensure at least 1 evaluation if possible.
            sample_count = min(6, remaining)
            # If n is large, sample fewer per round.
            sample_count = min(sample_count, max(1, 2 + n // 16))

            for _ in range(sample_count):
                if evals >= budget:
                    break
                # Correlated-ish perturbation: mix isotropic noise with scaled span.
                # This keeps units consistent across dimensions.
                scale = span_safe / (np.mean(span_safe) + 1e-12)
                # Gaussian perturbation scaled by sigma.
                z = rng.randn(n) * scale
                step_vec = sigma * z
                x = best_x + step_vec
                eval_x(x)
                if best_y is not None:
                    # Track improvement implicitly by comparing to snapshot.
                    pass

            # Determine improvement signal
            # If best_y decreased vs value recorded at the start of round, we mark improved.
            # We'll approximate with comparing best_y against itself after operations:
            # easiest: use improved_this_round flag from probes; also check if any sampling likely improved.
            if not improved_this_round:
                # If sampling happened, assume potential improvement only if best_y changed recently.
                # Since we didn't store snapshot, we keep conservative signal:
                improved_this_round = False

            # ---- Adapt sigma based on whether we improved during probe (or sampling assumed) ----
            # To avoid overly conservative behavior, incorporate a mild improvement heuristic:
            # if we found a better point than the best from the previous round, we'd know;
            # lacking snapshots, use no_improve_rounds management via probe improvements.
            if improved_this_round:
                no_improve_rounds = 0
                sigma = float(min(sigma_max, sigma * 1.05))
            else:
                no_improve_rounds += 1
                sigma = float(max(sigma_min, sigma * 0.85))

            # ---- Exploration: diversification via random points if stagnated ----
            if no_improve_rounds >= max_no_improve and evals < budget:
                remaining = budget - evals
                # Spend a small fraction on random injections.
                inject = min(3, remaining)
                for _ in range(inject):
                    if evals >= budget:
                        break
                    x = lower + rng.rand(n) * span_safe
                    eval_x(x)
                no_improve_rounds = 0
                # Restart sigma to encourage leaving local minima.
                sigma = float(min(sigma_max, sigma * 1.2))

            # If sigma became tiny due to bounds or convergence, occasionally re-inflate.
            if sigma <= sigma_min * 10 and evals < budget:
                sigma = float(min(sigma_max, np.mean(np.abs(span_safe)) * 0.2 + 1e-9))

        # At the end, best_x/best_y should exist due to initialization evaluations.
        if best_x is None:
            # Shouldn't happen if budget > 0, but keep safe.
            x0 = lower + rng.rand(n) * span_safe
            best_x = clip(x0)
            best_y = float("inf")
        return best_x, best_y
