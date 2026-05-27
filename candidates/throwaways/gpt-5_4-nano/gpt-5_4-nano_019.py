import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm that
# combines population-based random search with adaptive Gaussian sampling around
# the current best point. It is designed to work across dimensions without
# requiring gradient information.
# Search state: The algorithm tracks the current best solution (best_x, best_y),
# remaining evaluation budget, and a sampling scale (sigma) that controls the size
# of perturbations. It also maintains a small "population" of recent candidate
# points to improve robustness early on.
# Candidate generation: Each iteration draws candidates from either uniform
# sampling over the box or from a Gaussian distribution centered at the current
# best. The Gaussian step size (sigma) adapts based on recent improvements.
# Selection and replacement: Every evaluated candidate is compared to best_y; if it
# improves (strictly lower for minimization), best_x and best_y are updated.
# Additionally, a small set of recent candidates helps decide whether to reduce
# or increase exploration.
# Adaptation: If the search improves frequently, sigma shrinks to exploit; if it
# stalls, sigma grows to explore. This adaptation is driven by counters of
# consecutive non-improving iterations.
# Exploration mechanisms: Uniform sampling at the start and occasional mixture
# sampling thereafter ensures the method can escape local basins.
# Exploitation mechanisms: The Gaussian sampling around the best point focuses
# evaluations on promising regions.
# Boundary handling: Candidate points are clipped to the feasible bounds after
# mutation/sampling, ensuring all evaluated points respect constraints.
# Budget strategy: The algorithm never exceeds the evaluation budget by computing
# exactly how many candidates can be evaluated in each phase and stopping early
# when the budget is exhausted.
# Closest known influences: The structure resembles an Evolution Strategies-like
# approach (best-centered Gaussian sampling) with simple adaptive step-size and
# early uniform exploration, tailored for a minimal, dependency-free benchmark.
# Novelty or unusual aspects: The implementation uses a budget-aware, dimension-
# scaled scheduling of population size and step scale; it also adapts sigma based
# on both improvements and a diversity check from recent samples.
# Failure modes: If the objective is extremely irregular/noisy, step-size adaptation
# may oscillate or overfit to a misleading incumbent; with very small budgets,
# the method can degrade to near-random search.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return a deterministic point inside bounds if possible
            lb, ub = self._get_bounds(func, dim)
            mid = (lb + ub) / 2.0
            return mid, float("inf")

        lb, ub = self._get_bounds(func, dim)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        # Ensure shapes
        if lb.shape != (dim,) or ub.shape != (dim,):
            lb = np.broadcast_to(lb, (dim,)).copy()
            ub = np.broadcast_to(ub, (dim,)).copy()

        # Global initial sigma based on box size (dimension-scaled)
        box = np.maximum(ub - lb, 1e-12)
        # Typical scale: fraction of box length; smaller for higher dims
        base_sigma = 0.25 * np.sqrt(box.mean()) / max(1.0, np.sqrt(dim / 5.0))
        base_sigma = float(max(base_sigma, 1e-6))

        evals = 0
        best_x = None
        best_y = float("inf")

        # Helper: evaluate one point with budget guard
        def eval_point(x):
            nonlocal evals, best_x, best_y
            if evals >= budget:
                return
            x = self._clip(x, lb, ub)
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()

        # Small helper for batch evaluation count
        def remaining():
            return budget - evals

        # ---- Initialization: evaluate a modest number of uniformly sampled points ----
        # Budget-aware population size
        # For very small budget, keep it tiny.
        init_n = int(min(max(5, 2 * dim), max(1, budget)))
        init_n = min(init_n, remaining())

        # Evaluate initial points uniformly
        for _ in range(init_n):
            x = lb + np.random.random(dim) * (ub - lb)
            eval_point(x)

        # If still None (should not happen unless budget==0 handled earlier)
        if best_x is None:
            best_x = lb + np.random.random(dim) * (ub - lb)

        # Recent candidates tracking for adaptation/diversity check
        recent = [best_x.copy()]
        consecutive_no_improve = 0
        sigma = base_sigma

        # Iteration schedule: use a mix of exploitation and exploration
        # We adapt step size based on improvement and "diversity" of recent candidates.
        # Number of candidate draws per iteration depends on remaining budget.
        # We will loop until budget exhausts.
        while remaining() > 0:
            rem = remaining()

            # Decide whether to explore with uniform sampling.
            # Higher when sigma is large or search seems stalled.
            stall_factor = min(1.0, consecutive_no_improve / max(1.0, 3.0 + 0.5 * np.sqrt(dim)))
            explore_prob = 0.15 + 0.5 * stall_factor  # between 0.15 and ~0.65

            # Candidates per iteration: keep overhead low; batch sizes moderate.
            # Scale with dimension but remain budget-aware.
            k = int(min(max(4, dim), rem))
            if k < 1:
                break

            improved_this_iter = False

            # Generate k candidates
            for _ in range(k):
                if remaining() <= 0:
                    break

                if np.random.random() < explore_prob:
                    # Uniform exploration
                    x = lb + np.random.random(dim) * (ub - lb)
                else:
                    # Best-centered Gaussian exploitation
                    # Use per-dimension sigma scaled by box, but keep simple and robust.
                    # Generate correlated step via diagonal Gaussian.
                    # Occasionally use a larger jump to escape local minima.
                    jump_scale = 1.0
                    if consecutive_no_improve >= 2 and np.random.random() < 0.25:
                        jump_scale = 2.0 + 2.0 * np.random.random()

                    # Normalize sigma relative to box size
                    # so that moves remain meaningful across different bound widths.
                    rel_sigma = sigma / max(1e-12, box.mean() ** 0.5)
                    # Convert to absolute scale per dimension
                    step = (np.random.normal(size=dim) * rel_sigma) * (box ** 0.5) * jump_scale
                    x = best_x + step

                    # Occasional "coordinate" perturbation for robustness in high dimensions
                    if dim >= 3 and np.random.random() < 0.08:
                        idx = np.random.randint(0, dim)
                        x[idx] = best_x[idx] + np.random.normal() * (box[idx] ** 0.5) * (sigma / (box.mean() ** 0.5 + 1e-12))

                y_before = best_y
                eval_point(x)
                if best_y < y_before:
                    improved_this_iter = True
                    consecutive_no_improve = 0
                    recent.append(best_x.copy())
                else:
                    # Still keep recent samples for diversity measure
                    # (re-evaluate best after clipping via eval_point)
                    # We'll store the point that was attempted only if space allows.
                    # To keep it simple, store best_x only when updated; otherwise leave as-is.
                    consecutive_no_improve += 1 if improved_this_iter is False else 0

            # Adapt sigma after iteration
            # If we improved, exploit: shrink sigma; otherwise explore: grow sigma.
            # Also incorporate a diversity measure over recent best_x snapshots.
            if improved_this_iter:
                # Shrink with a factor related to dimension; prevents collapse in large dim.
                sigma *= 0.82 / (1.0 + 0.1 * np.sqrt(dim))
            else:
                sigma *= 1.15 + 0.25 * stall_factor

            sigma = float(np.clip(sigma, 1e-9, 2.5 * base_sigma * (1.0 + 0.5 * np.sqrt(dim))))

            # Diversity check: if recent points are too clustered, enlarge sigma
            # (helps escape if adaptation gets overconfident).
            if len(recent) >= min(8, dim + 3):
                recent_arr = np.vstack(recent[-min(12, len(recent)):])
                # average pairwise distance to best
                diffs = recent_arr - best_x[None, :]
                avg_dist = float(np.mean(np.linalg.norm(diffs, axis=1)))
                # expected scale for avg_dist to avoid over-clustering
                # use box mean as reference
                target = 0.15 * np.sqrt(box.mean()) * (1.0 + 0.1 * np.sqrt(dim))
                if avg_dist < target * 0.35:
                    sigma *= 1.1

            # Limit recent list growth
            if len(recent) > 32:
                recent = recent[-32:]

            # If budget nearly exhausted, stop sooner to avoid extra loop overhead.
            if remaining() <= 0:
                break

        # Ensure best_x is defined
        if best_x is None:
            best_x = lb + np.random.random(dim) * (ub - lb)
            best_y = float("inf")
        return best_x, best_y

    @staticmethod
    def _get_bounds(func, dim):
        # Read bounds from either func.lower / func.upper or func.bounds.lb / func.bounds.ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
            return lower, upper
        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lower = np.asarray(b.lb, dtype=float)
                upper = np.asarray(b.ub, dtype=float)
                return lower, upper
        raise AttributeError(
            "Objective function must provide bounds via func.lower/func.upper "
            "or func.bounds.lb/func.bounds.ub."
        )

    @staticmethod
    def _clip(x, lb, ub):
        return np.minimum(np.maximum(x, lb), ub)
