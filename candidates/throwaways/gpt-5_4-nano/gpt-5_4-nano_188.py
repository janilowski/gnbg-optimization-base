# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm using a
# derivative-free evolutionary strategy (ES) with diagonal step-size
# control. It maintains a small population of candidate solutions and
# iteratively improves the best-so-far point by mutating around it.
# Search state: Tracks best solution (best_x, best_y) and an adaptive
# mutation scale (sigma). Also tracks evaluation count to never exceed the
# provided budget.
# Candidate generation: Each iteration samples offspring by adding
# Gaussian noise to parents, with step size sigma. Parents are selected
# from the current population with elitist pressure.
# Selection and replacement: Uses fitness ranking (lower is better). Keeps
# the elite individuals (and replaces the rest with offspring) in a simple,
# steady-state manner to reduce objective evaluations.
# Adaptation: Updates sigma using 1/5-th rule (success rate based on whether
# new samples improve relative to the current best). This provides
# robustness across dimensions without requiring gradients.
# Exploration mechanisms: Population diversity via Gaussian perturbations
# and occasional larger jumps when success is low.
# Exploitation mechanisms: Sampling is biased toward the current elite /
# best point so the search focuses as it improves.
# Boundary handling: Samples are clamped to provided bounds; if bounds are
# infinite/unknown, the algorithm proceeds without clamping.
# Budget strategy: Computes an iteration count based on the remaining
# evaluations; each candidate evaluation increments the counter. Stops
# exactly when the budget would be exceeded.
# Closest known influences: Similar in spirit to CMA-ES-lite / evolution
# strategies and the 1/5 success rule for step-size adaptation, but kept
# deliberately simpler for a compact implementation.
# Novelty or unusual aspects: Uses a small steady-state elite pool and
# blends exploration around both the current best and top-k points to remain
# robust even with tight bounds.
# Failure modes: If the objective is extremely noisy or bounds are extremely
# tight with a very small feasible region, sigma may shrink too quickly;
# the adaptation includes safeguards to avoid collapse.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._read_bounds(func)

        dim = self.dim
        budget = self.budget
        if budget <= 0:
            raise ValueError("budget must be positive")

        rng = np.random

        # Helper: clamp point to finite bounds
        def clamp(x):
            if lb is not None and ub is not None:
                x = np.where(np.isfinite(lb), np.maximum(lb, x), x)
                x = np.where(np.isfinite(ub), np.minimum(ub, x), x)
            return x

        # Evaluate with budget enforcement
        evals = 0
        best_x = None
        best_y = None

        def evaluate(x):
            nonlocal evals, best_x, best_y
            if evals >= budget:
                return best_x, best_y
            x = np.asarray(x, dtype=float)
            x = clamp(x)
            y = float(func(x))
            evals += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = x.copy()
            return x, y

        # Determine bounds range to set initial sigma
        if lb is not None and ub is not None:
            lb = np.asarray(lb, dtype=float).reshape(-1)
            ub = np.asarray(ub, dtype=float).reshape(-1)
            # If any infinite bounds, use a generic scale
            finite = np.isfinite(lb) & np.isfinite(ub)
            if np.any(finite):
                range_ = ub[finite] - lb[finite]
                # Use geometric-ish center scale
                span = float(np.median(np.abs(range_)))
                if span <= 0:
                    span = float(np.max(np.abs(ub[finite] + lb[finite])) + 1.0)
            else:
                span = 1.0
        else:
            span = 1.0

        # Initialize population size relative to dimension and budget
        # Keep it small for budget efficiency.
        pop = int(min(12, max(4, (budget // 5) // max(1, dim))))
        pop = max(4, min(pop, 12))
        pop = min(pop, budget)  # cannot evaluate more than budget

        # Initial sigma: a fraction of typical bound range or a default
        sigma = 0.2 * span
        sigma = max(sigma, 1e-6)

        # Create initial population (include a uniform sample and best candidate)
        population = []
        fitness = []
        # If bounds are finite, sample within them; else use standard normal around 0
        if lb is not None and ub is not None and np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
            for _ in range(pop):
                x0 = lb + rng.rand(dim) * (ub - lb)
                x0 = clamp(x0)
                y = float(func(x0))
                evals += 1
                population.append(x0)
                fitness.append(y)
                if best_y is None or y < best_y:
                    best_y = y
                    best_x = x0.copy()
        else:
            # Mix of near-zero and random Gaussian
            for i in range(pop):
                if lb is not None and ub is not None:
                    # For mixed infinite bounds, clamp after sampling
                    x0 = rng.randn(dim) * sigma
                    x0 = clamp(x0)
                else:
                    x0 = rng.randn(dim) * sigma
                y = float(func(x0))
                evals += 1
                population.append(x0)
                fitness.append(y)
                if best_y is None or y < best_y:
                    best_y = y
                    best_x = x0.copy()

        population = np.array(population, dtype=float)
        fitness = np.array(fitness, dtype=float)

        # If budget used all evaluations, return best so far
        if evals >= budget:
            return best_x, best_y

        # Set iteration count by remaining budget and offspring evaluations
        # Each iteration generates up to offspring_per_iter, but never exceeds budget.
        # Steady-state: produce one offspring at a time to keep control tight.
        # We'll still pick parents from a top-k elite.
        top_k = max(2, min(5, pop // 2))

        # Step-size adaptation: 1/5th success rule on improvement over best_y
        success_count = 0
        window = 20  # update sigma every ~window offspring evaluations
        window_evals = 0

        # Maximum scale to avoid runaway; minimum to avoid collapse
        sigma_min = 1e-12
        sigma_max = 10.0 * (span if span > 0 else 1.0)

        # Main loop
        while evals < budget:
            # Rank and pick elites
            order = np.argsort(fitness)  # minimization
            population = population[order]
            fitness = fitness[order]

            # Occasionally sample around multiple top points for stability.
            # Blend factor biases toward best.
            elite = population[:top_k]
            # Parent choice: mostly best, sometimes others
            if rng.rand() < 0.75 or top_k == 1:
                parent = elite[0]
            else:
                parent = elite[rng.randint(0, top_k)]

            # Additional perturbation direction uses second elite (if available)
            if top_k > 1 and rng.rand() < 0.35:
                parent2 = elite[rng.randint(1, top_k)]
                direction = parent - parent2
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                else:
                    direction = rng.randn(dim)
                    direction /= np.linalg.norm(direction) + 1e-12
            else:
                direction = rng.randn(dim)
                direction /= np.linalg.norm(direction) + 1e-12

            # Mutation: diagonal Gaussian scaled by sigma
            # Add a small component in direction of elite difference.
            z = rng.randn(dim)
            step = sigma * z + 0.1 * sigma * direction
            x_new = parent + step
            x_new = clamp(x_new)

            # Evaluate new candidate
            y_new = float(func(x_new))
            evals += 1
            if best_y is None or y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()
                success = True
            else:
                success = False

            # Replace: steady-state replacement into worst slot
            # (population already sorted; keep size constant)
            # Remove one worst individual and insert new candidate.
            # We can keep population unsorted until next iteration.
            population[-1] = x_new
            fitness[-1] = y_new

            # Adapt sigma based on success
            window_evals += 1
            if success:
                success_count += 1

            if window_evals >= window or evals >= budget:
                # 1/5 success rule:
                # if success_rate > 0.2 -> increase sigma
                # else decrease sigma
                success_rate = success_count / max(1, window_evals)
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.85

                sigma = float(np.clip(sigma, sigma_min, sigma_max))
                success_count = 0
                window_evals = 0

        return best_x, best_y

    def _read_bounds(self, func):
        # Bounds may be defined as:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        # These could be scalars or sequences. We return np arrays or None.
        lb = None
        ub = None

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = getattr(func, "lower")
            ub = getattr(func, "upper")
        elif hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = getattr(b, "lb")
                ub = getattr(b, "ub")

        if lb is None or ub is None:
            return None, None

        # Convert to arrays if possible
        try:
            lb_arr = np.asarray(lb, dtype=float).reshape(-1)
            ub_arr = np.asarray(ub, dtype=float).reshape(-1)
            if lb_arr.size == 1:
                lb_arr = np.full(self.dim, float(lb_arr), dtype=float)
            if ub_arr.size == 1:
                ub_arr = np.full(self.dim, float(ub_arr), dtype=float)
            # If sizes mismatch, attempt broadcast by padding/truncation cautiously.
            if lb_arr.size != self.dim or ub_arr.size != self.dim:
                # If one side matches, broadcast the other if scalar-ish or trim/pad
                if lb_arr.size == 1:
                    lb_arr = np.full(self.dim, float(lb_arr[0]), dtype=float)
                if ub_arr.size == 1:
                    ub_arr = np.full(self.dim, float(ub_arr[0]), dtype=float)
                lb_arr = lb_arr[: self.dim] if lb_arr.size >= self.dim else np.pad(
                    lb_arr, (0, self.dim - lb_arr.size), mode="edge"
                )
                ub_arr = ub_arr[: self.dim] if ub_arr.size >= self.dim else np.pad(
                    ub_arr, (0, self.dim - ub_arr.size), mode="edge"
                )
            return lb_arr, ub_arr
        except Exception:
            # If bounds cannot be interpreted, ignore bounds
            return None, None
