# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm using a
# derivative-free strategy that combines global random sampling with local
# coordinate-wise refinement. It maintains a small population of candidate
# points and repeatedly perturbs the current best to explore nearby regions.
#
# Search state: Keeps the best solution found so far (best_x, best_y) and a
# small set of recent candidates. Tracks remaining function evaluations to
# ensure the budget is never exceeded.
#
# Candidate generation: At each iteration, generates multiple trial points by
# adding Gaussian noise to the current best in random coordinate subsets.
# It also performs a cheap coordinate-wise step attempt along randomly chosen
# directions (sign flips) when beneficial.
#
# Selection and replacement: Evaluates all generated candidates (while staying
# within budget). Any candidate that improves the incumbent best is accepted as
# the new best. A simple pool of top candidates is updated for continued
# guidance of sampling scale.
#
# Adaptation: The mutation scale (step size) adapts based on observed
# improvements: it shrinks when progress stalls and grows slightly when
# improvements are found, bounded to safe ranges relative to the domain.
#
# Exploration mechanisms: Uses an initial wide random design and continuous
# stochastic perturbations with a decaying scale, plus occasional re-seeding
# from random points to escape local minima.
#
# Exploitation mechanisms: Focuses perturbations around the incumbent best and
# includes coordinate-wise signed steps to refine along promising axes.
#
# Boundary handling: Applies clipping to keep all candidate points within
# provided bounds. When a point hits a boundary, effective step is reduced via
# shrinking the scale to avoid repeated invalid moves.
#
# Budget strategy: Converts the provided budget into an exact evaluation budget
# counter. Each objective call decrements the remaining counter; no evaluations
# are made after the budget is exhausted.
#
# Closest known influences: Inspired by basic evolutionary strategies and
# coordinate search hybrids (ES-style mutation + coordinate refinement), adapted
# to a strict evaluation budget.
#
# Novelty or unusual aspects: Combines a tiny candidate pool with scale control
# driven by improvement rate, and uses random coordinate subset perturbations
# to remain effective across dimensions with limited evaluations.
#
# Failure modes: If the objective is extremely noisy or non-smooth, the scale
# adaptation and coordinate refinements may lead to slow progress. With very
# small budgets, performance degrades toward random search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        if d <= 0:
            raise ValueError("dim must be positive")
        if self.budget <= 0:
            raise ValueError("budget must be positive")

        # Read bounds from func.
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("func must provide bounds via lower/upper or bounds.lb/bounds.ub")

        if lb.shape != (d,) or ub.shape != (d,):
            lb = np.reshape(lb, (d,))
            ub = np.reshape(ub, (d,))
        if np.any(ub <= lb):
            raise ValueError("Invalid bounds: each upper bound must be greater than lower bound")

        # Ensure numeric stability and compute scale references.
        span = ub - lb
        # Avoid extremely tiny spans which can cause step sizes to underflow.
        span_safe = np.maximum(span, 1e-12)
        center = (lb + ub) * 0.5

        evals = 0
        remaining = self.budget

        def clamp(x):
            # Keep candidates inside bounds.
            return np.minimum(ub, np.maximum(lb, x))

        def eval_obj(x):
            nonlocal evals, remaining
            if remaining <= 0:
                # Should never happen due to careful budgeting, but guard anyway.
                return None
            x = np.asarray(x, dtype=float)
            y = func(x)
            evals += 1
            remaining -= 1
            return float(y)

        # Initialization: evaluate incumbent(s).
        # Always evaluate center; optionally evaluate a few random points.
        best_x = clamp(center.copy())
        best_y = eval_obj(best_x)
        if best_y is None:
            return best_x, float("inf")

        # Small candidate pool for guidance (store best few).
        # We'll keep at most pool_size points including best.
        pool_size = 6 if d >= 6 else 4
        pool = [(best_x.copy(), best_y)]
        pool = pool[:pool_size]

        # Choose initial sampling count based on budget and dimension.
        # For large dim, keep initial random sampling modest.
        # Ensure at least 1 additional evaluation if budget allows.
        init_random = min(8, max(0, self.budget - 1))
        init_random = min(init_random, 2 + d // 3)
        for _ in range(init_random):
            if remaining <= 0:
                break
            x = lb + np.random.rand(d) * span_safe
            x = clamp(x)
            y = eval_obj(x)
            if y is None:
                break
            if y < best_y:
                best_x, best_y = x, y
            pool.append((x, y))
            pool.sort(key=lambda t: t[1])
            pool = pool[:pool_size]

        # Determine initial step size:
        # Start around a fraction of the domain.
        step = 0.35 * float(np.mean(span_safe))
        step = max(step, 1e-6)

        # Iteration loop: each loop proposes several candidates.
        # Keep proposals bounded so we never exceed budget.
        # Use a decaying schedule with adaptive adjustment.
        improvement_streak = 0
        # coordinate subset size for perturbations
        subset_k_base = max(1, min(d, 1 + d // 4))

        # Small helper to update pool and best.
        def consider(x, y):
            nonlocal best_x, best_y, pool
            if y < best_y:
                best_x, best_y = x, y
                return True
            pool.append((x, y))
            pool.sort(key=lambda t: t[1])
            pool = pool[:pool_size]
            return False

        # Main search.
        # We run while we still have budget.
        # Each iteration uses up to proposal_count evaluations.
        while remaining > 0:
            # Decide how many proposals we can afford this iteration.
            # Keep batch size small for responsiveness.
            # Typical batch: 4..min(12, remaining)
            proposal_count = min(10 if d <= 30 else 7, remaining)

            # Choose a reference center: best, or one from pool occasionally.
            if np.random.rand() < 0.85:
                ref = best_x
            else:
                ref = pool[np.random.randint(len(pool))][0]

            # Coordinate subset size: grows slowly as budget allows.
            subset_k = subset_k_base
            # Randomly pick subset size sometimes larger for exploration.
            if d > 4 and np.random.rand() < 0.2:
                subset_k = min(d, subset_k_base + max(1, d // 6))

            improved = False
            local_best_this_iter = best_y

            # Generate candidates by coordinate-subset Gaussian perturbations.
            for _ in range(proposal_count):
                if remaining <= 0:
                    break

                # Exploration schedule: occasionally re-sample uniformly.
                if np.random.rand() < 0.08 and remaining > 1:
                    x = lb + np.random.rand(d) * span_safe
                    x = clamp(x)
                else:
                    # Build perturbation in a random coordinate subset.
                    x = ref.copy()
                    idx = np.random.choice(d, size=subset_k, replace=False)
                    # Use heavier-tailed perturbations sometimes for escapes.
                    if np.random.rand() < 0.15:
                        # Student-ish via scaled normal with random scaling.
                        scale = step * (0.5 + 1.5 * np.random.rand())
                        noise = np.random.normal(0.0, 1.0, size=subset_k) * scale
                    else:
                        noise = np.random.normal(0.0, 1.0, size=subset_k) * step

                    x[idx] = x[idx] + noise
                    x = clamp(x)

                y = eval_obj(x)
                if y is None:
                    break
                if y < best_y:
                    improved = True
                    local_best_this_iter = min(local_best_this_iter, y)
                    best_x = x
                    best_y = y
                    # Update pool including new best and a few other points.
                    pool.append((x.copy(), y))
                    pool.sort(key=lambda t: t[1])
                    pool = pool[:pool_size]
                else:
                    pool.append((x.copy(), y))
                    pool.sort(key=lambda t: t[1])
                    pool = pool[:pool_size]

            # Lightweight coordinate-wise refinement (exploitation).
            # Use only if we still have budget and if last improvements happened recently.
            if remaining > 0 and (improved or np.random.rand() < 0.35):
                # Try a few coordinate steps around current best.
                tries = min(3, remaining)
                for _ in range(tries):
                    if remaining <= 0:
                        break
                    # Choose a coordinate; sometimes try from pool center to escape.
                    base = best_x if np.random.rand() < 0.8 else pool[np.random.randint(len(pool))][0]
                    j = np.random.randint(d)
                    # Signed step attempt.
                    sign = 1.0 if np.random.rand() < 0.5 else -1.0
                    # Step relative to that coordinate span.
                    coord_span = span_safe[j]
                    # Make step smaller for refinement.
                    step_j = min(step, 0.25 * coord_span)
                    x = base.copy()
                    x[j] = x[j] + sign * step_j
                    x = clamp(x)
                    y = eval_obj(x)
                    if y is None:
                        break
                    if y < best_y:
                        best_x, best_y = x, y
                        improved = True
                        pool.append((x.copy(), y))
                        pool.sort(key=lambda t: t[1])
                        pool = pool[:pool_size]

            # Adapt step size based on improvement.
            if improved:
                improvement_streak += 1
                # Increase slightly to capitalize, but bounded.
                step *= 1.08 if improvement_streak <= 3 else 1.02
            else:
                improvement_streak = 0
                # Shrink when no progress.
                step *= 0.82

            # Bound step relative to domain.
            max_step = 0.9 * float(np.mean(span_safe))
            min_step = 1e-6 * float(np.mean(span_safe))
            step = float(np.clip(step, min_step, max_step))

            # If stuck and budget remains, do a small restart around best center.
            if remaining > 0 and (step <= min_step * 2) and np.random.rand() < 0.2:
                # Replace best with a fresh perturbation around best to re-gain mobility.
                # This is still evaluation-budget controlled.
                if remaining > 0:
                    x = best_x + np.random.normal(0.0, step * 5.0, size=d)
                    x = clamp(x)
                    y = eval_obj(x)
                    if y is not None and y < best_y:
                        best_x, best_y = x, y
                        pool.append((x.copy(), y))
                        pool.sort(key=lambda t: t[1])
                        pool = pool[:pool_size]

        return best_x, best_y
