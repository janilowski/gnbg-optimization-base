import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm using a
# randomized population and coordinate-wise local refinements. It maintains
# a small set of candidate points, evaluates them within the given budget,
# and repeatedly refines the current best using adaptive step sizes.
#
# Search state: Keeps a population of points (X) with their objective values (f),
# along with an evaluation counter and the current best solution (best_x, best_y).
# Also maintains a per-dimension step size vector that shrinks/grows based on
# recent improvements.
#
# Candidate generation: Each iteration generates new candidates around
# existing population members using:
#   - isotropic Gaussian perturbations around the current best,
#   - directional (coordinate) perturbations based on sampled signs.
# Candidate proposals are clipped to bounds.
#
# Selection and replacement: Newly evaluated candidates are merged with the
# current population; the best 'pop_size' points by objective value survive
# (elitist selection). The global best is updated whenever a better value is found.
#
# Adaptation: Step sizes (sigma) adapt using whether improvements occur:
#   - On success (global best improves), sigma is gently increased to explore.
#   - On failure, sigma is decreased to focus exploitation.
#
# Exploration mechanisms: Uses random Gaussian sampling and occasional
# coordinate perturbations to explore broadly early and around the best.
#
# Exploitation mechanisms: When improvements are found, step sizes shrink less
# aggressively and coordinate refinements focus on dimensions with high leverage
# (improved points).
#
# Boundary handling: All candidates are clipped to the valid box bounds.
# If clipping causes too many repeated points, noise is increased slightly.
#
# Budget strategy: Never exceeds func evaluations budget. The algorithm
# precomputes how many iterations and candidates it can afford, and each
# evaluation checks the remaining budget before calling the objective.
#
# Closest known influences: A blend of (μ+λ) evolutionary selection with
# CMA-like step adaptation simplified to a diagonal sigma vector and
# coordinate-wise local search.
#
# Novelty or unusual aspects: Combines elitist population merging with a
# lightweight coordinate-sign probing mechanism driven by the current best,
# while keeping the implementation short and robust across dimensions.
#
# Failure modes: For extremely noisy objectives, the step adaptation may
# overreact to stochastic variations; for very tight bounds or very small
# budgets, the method may rely on a near-random initial sampling.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._get_bounds(func)
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng()  # harness controls global seed externally

        # Handle degenerate cases
        if budget <= 0 or dim <= 0:
            x0 = np.clip(np.zeros(dim, dtype=float), lb, ub)
            return x0, float(self._safe_eval(func, x0, budget=budget, evals=0))

        # Ensure bounds are finite arrays
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != dim or ub.size != dim:
            # Fallback: broadcast scalar or truncate/extend
            if lb.size == 1:
                lb = np.full(dim, float(lb[0]))
            else:
                lb = np.resize(lb, dim)
            if ub.size == 1:
                ub = np.full(dim, float(ub[0]))
            else:
                ub = np.resize(ub, dim)

        # If bounds are equal, objective is evaluated at that point only
        if np.allclose(lb, ub):
            x = np.array(lb, copy=True, dtype=float)
            y = self._call_objective(func, x)
            return x, y

        evals = 0

        # Initialization: population size scales with dimension but stays small/compact
        pop_size = int(min(12, max(4, 2 + dim // 2)))
        pop_size = max(4, min(pop_size, budget))  # cannot exceed budget

        def rand_point():
            return lb + rng.random(dim) * (ub - lb)

        # Initial population
        X = np.zeros((pop_size, dim), dtype=float)
        f = np.empty(pop_size, dtype=float)

        for i in range(pop_size):
            if evals >= budget:
                break
            x = rand_point()
            y = self._call_objective(func, x)
            evals += 1
            X[i] = x
            f[i] = y

        # If budget was too small (shouldn't happen due to pop_size clamp), trim
        n = min(pop_size, evals)
        X = X[:n]
        f = f[:n]

        best_idx = int(np.argmin(f))
        best_x = np.array(X[best_idx], copy=True)
        best_y = float(f[best_idx])

        # Step size: start proportional to box size
        box = ub - lb
        # Avoid zero range dimensions
        base_sigma = 0.25 * box
        base_sigma = np.where(base_sigma > 0, base_sigma, 1.0)
        sigma = base_sigma

        # Determine iteration budget; each iteration evaluates n_children candidates
        # We'll use a mixture: around best + coordinate probes.
        remaining = budget - evals
        if remaining <= 0:
            return best_x, best_y

        # Iteration planning
        # Keep children per iteration modest to allow adaptation and merging.
        children_per_iter = int(min(10, max(4, budget // 10 if budget >= 10 else remaining)))
        children_per_iter = max(4, min(children_per_iter, remaining))
        max_iters = int(max(1, remaining // children_per_iter))

        # To reduce repeated clipping into bounds for narrow boxes
        min_sigma_floor = np.maximum(1e-12, 1e-6 * box + 1e-12)

        # Main loop
        for _ in range(max_iters):
            if evals >= budget:
                break

            # Number of children we can afford this iteration
            affordable = budget - evals
            n_children = min(children_per_iter, affordable)

            # Generate children
            children = np.zeros((n_children, dim), dtype=float)
            child_f = np.empty(n_children, dtype=float)

            # Probabilistic mix: Gaussian around best vs coordinate-sign probes
            # Higher exploration early.
            explore_prob = 0.65
            if budget > 0:
                frac_used = evals / float(budget)
                explore_prob = float(np.clip(explore_prob - 0.45 * frac_used, 0.2, 0.7))

            # Ensure population has at least one point
            if n <= 0:
                break

            for k in range(n_children):
                # Pick parent from elites
                parent = X[rng.integers(0, n)]
                mode = rng.random() < explore_prob

                if mode:
                    # Isotropic-ish perturbation around best, scaled by sigma
                    z = rng.normal(0.0, 1.0, dim)
                    x = best_x + z * sigma
                    # Slight pull toward sampled parent to diversify
                    if rng.random() < 0.3:
                        x = 0.75 * x + 0.25 * parent
                else:
                    # Coordinate-sign probing: choose a random coordinate subset
                    x = np.array(best_x, copy=True)
                    # Probe 1..ceil(dim/4) coords
                    m = int(rng.integers(1, max(2, (dim + 3) // 4) + 1))
                    idxs = rng.choice(dim, size=m, replace=False)
                    signs = rng.choice((-1.0, 1.0), size=m)
                    # Use a smaller step for coordinate refinement
                    local_sigma = np.maximum(min_sigma_floor, sigma * 0.5)
                    x[idxs] = x[idxs] + signs * local_sigma[idxs] * rng.random(m)

                    # With some probability, add a small Gaussian perturbation
                    if rng.random() < 0.4:
                        x = x + rng.normal(0.0, 0.5, dim) * min_sigma_floor * 0.5

                # Boundary handling via clipping
                x = np.clip(x, lb, ub)
                children[k] = x

                y = self._call_objective(func, x)
                evals += 1
                child_f[k] = y
                if evals >= budget:
                    # Fill rest with dummy values (not used)
                    if k + 1 < n_children:
                        child_f[k + 1 :] = np.inf
                    break

            # Merge and select elites
            # Trim if budget cut early
            valid = min(n_children, int(n_children - np.sum(np.isinf(child_f))))
            if valid <= 0:
                break

            X_new = np.vstack((X, children[:valid]))
            f_new = np.concatenate((f, child_f[:valid]))

            elite_count = min(pop_size, X_new.shape[0])
            elite_idx = np.argsort(f_new)[:elite_count]
            X = X_new[elite_idx]
            f = f_new[elite_idx]
            n = elite_count

            # Update global best and adapt sigma
            prev_best_y = best_y
            best_idx = int(np.argmin(f))
            if f[best_idx] < best_y:
                best_y = float(f[best_idx])
                best_x = np.array(X[best_idx], copy=True)

            improved = best_y < prev_best_y

            # Adapt sigma using simple rules; maintain diagonal step sizes
            if improved:
                # Mild increase to explore around new optimum
                sigma = np.minimum(box * 0.5 + 1e-12, sigma * 1.12)
            else:
                # Decrease to exploit
                sigma = np.maximum(min_sigma_floor, sigma * 0.82)

            # If sigma becomes too small or no progress, slightly re-inject randomness
            if not improved and np.all(sigma <= min_sigma_floor * 2.0):
                sigma = np.maximum(min_sigma_floor, sigma * 1.5)

        return best_x, best_y

    def _get_bounds(self, func):
        # Read bounds from func.lower/upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
            return lb, ub

        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return b.lb, b.ub

        # As a last resort, try attributes named 'lower_bound' etc.
        for lo_name, hi_name in [
            ("lower_bound", "upper_bound"),
            ("lb", "ub"),
            ("min", "max"),
        ]:
            if hasattr(func, lo_name) and hasattr(func, hi_name):
                return getattr(func, lo_name), getattr(func, hi_name)

        raise AttributeError("Objective function must provide bounds via lower/upper or bounds.lb/bounds.ub.")

    def _call_objective(self, func, x):
        # The harness expects func(x) -> scalar.
        y = func(x)
        # Ensure scalar float
        return float(np.asarray(y).item())

    def _safe_eval(self, func, x, budget, evals):
        if evals >= budget:
            return float("inf")
        return self._call_objective(func, x)
