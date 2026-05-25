# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm using a global
# exploration phase followed by iterative local exploitation. It maintains a small
# population of candidate points, uses random sampling with variance scaling, and
# performs coordinate-wise local refinement around the current best.
# Search state: Tracks an evaluation budget (fixed cap), number of evaluations used,
# current best point/value, and a small set of population members (points and values).
# Candidate generation: Generates new candidates by sampling Gaussian perturbations
# around the current best and around each population member, with step sizes that
# shrink when improvements are found. Also uses a simple coordinate-wise “neighbor”
# search (±step along each dimension) around the best.
# Selection and replacement: Evaluates candidates and keeps the best K points in the
# population (elitist replacement). The global best is updated whenever a new lower
# value is found.
# Adaptation: Uses success-based step-size control: if new samples improve upon the
# best, the main step size is reduced (or kept smaller) while local step is refreshed;
# if not, step sizes contract slowly to focus the search.
# Exploration mechanisms: Early-stage broad Gaussian sampling with larger step sizes,
# plus population-based resampling around multiple elites to avoid premature convergence.
# Exploitation mechanisms: Coordinate-wise local probes around the current best, followed
# by shrinking local step sizes and continuing elite-based refinement.
# Boundary handling: Candidates are clipped to the provided bounds (box constraints).
# Budget strategy: Strictly caps the number of objective evaluations by the provided budget.
# Each evaluation checks remaining budget and stops generating candidates when exhausted.
# Closest known influences: Inspired by population-based random search (evolutionary-
# like selection) combined with deterministic local coordinate probes and success-based
# step-size control.
# Novelty or unusual aspects: The algorithm is intentionally lightweight and dimension-agnostic,
# blending population resampling and coordinate-wise neighborhood search without any
# external dependencies beyond numpy.
# Failure modes: In very rugged or highly deceptive functions, random sampling may not
# find good basins before budget is exhausted; coordinate probing can stall if the best
# region is narrow and step sizes shrink too quickly. Clipping can cause stagnation near
# boundaries for some objectives.
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
            # No evaluations possible; return zeros within bounds if available.
            lb, ub = self._get_bounds(func, dim)
            x0 = np.zeros(dim, dtype=float)
            if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
                x0 = np.clip(x0, lb, ub)
            return x0, float("inf")

        lb, ub = self._get_bounds(func, dim)
        span = ub - lb
        # If span has zeros, make them nonzero to avoid division by 0 and NaNs.
        span_safe = np.where(span > 0, span, 1.0)

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        eval_count = 0

        best_x = None
        best_y = float("inf")

        # Small elitist population
        pop_size = max(4, min(12, 2 * dim))
        pop_x = []
        pop_y = []

        # Allocate evaluations for global vs local phases
        # (roughly 60% global sampling, 40% exploitation).
        global_budget = int(budget * 0.6)
        local_budget = budget - global_budget

        main_step = 0.25  # relative to span
        local_step = 0.05  # relative to span

        # Coordinate probe count per local iteration (stagger to save budget in high dims)
        coord_stride = 1 if dim <= 12 else max(1, dim // 12)

        def evaluate(x):
            nonlocal eval_count, best_x, best_y
            if eval_count >= budget:
                return
            y = float(func(np.asarray(x, dtype=float)))
            eval_count += 1
            if y < best_y:
                best_y = y
                best_x = np.asarray(x, dtype=float).copy()
            return y

        # Create initial population: uniform sampling plus one center point
        # (helps in bounded problems where the optimum might be near the center).
        center = lb + 0.5 * span_safe
        if eval_count < budget:
            y = evaluate(center)
            if eval_count <= budget:
                pop_x.append(np.asarray(best_x, dtype=float))
                pop_y.append(best_y)

        # Remaining initial evaluations
        while eval_count < min(budget, pop_size) and (eval_count < global_budget):
            x = lb + np.random.rand(dim) * span_safe
            evaluate(x)
            if best_x is not None:
                pop_x.append(np.asarray(best_x, dtype=float))
                pop_y.append(best_y)

        # Ensure population is properly sized; fill from uniform if needed.
        while len(pop_x) < pop_size and eval_count < budget:
            x = lb + np.random.rand(dim) * span_safe
            y = evaluate(x)
            if y is not None:
                # Keep x, y as candidate in population
                pop_x.append(np.asarray(x, dtype=float))
                pop_y.append(float(y))

        # Helper: keep best K in population
        def elitist_prune(k=pop_size):
            nonlocal pop_x, pop_y
            if not pop_x:
                return
            idx = np.argsort(np.asarray(pop_y, dtype=float))
            k = min(k, len(idx))
            pop_x = [pop_x[i] for i in idx[:k]]
            pop_y = [pop_y[i] for i in idx[:k]]

        elitist_prune(pop_size)

        # If still no best_x (e.g., weird func behavior), set something valid.
        if best_x is None:
            best_x = clip(lb + 0.5 * span_safe)
            best_y = float("inf")

        def sample_around(x0, step_rel, n_samples=1):
            # Gaussian sampling with coordinate-wise scaling; then clip.
            step = step_rel * span_safe
            for _ in range(n_samples):
                eps = np.random.randn(dim) * step
                yield clip(np.asarray(x0, dtype=float) + eps)

        # --- Global exploration phase ---
        # Generate candidates from best and population elites with decreasing spread.
        # Each outer loop uses a handful of candidate evaluations.
        while eval_count < global_budget:
            if not pop_x:
                break
            # Shrink step gradually throughout global phase.
            frac = (eval_count / max(1, global_budget))
            cur_step_rel = main_step * (1.0 - 0.6 * frac)
            cur_step_rel = max(cur_step_rel, 1e-3)

            # Number of candidates per iteration depends on remaining budget.
            remaining = global_budget - eval_count
            batch = min(max(4, pop_size), remaining)

            improved_before = best_y

            # Sample around a few top elites (including best)
            elites = pop_x[: min(len(pop_x), max(2, dim // 3 + 2))]
            candidates = []
            for _ in range(batch):
                x0 = elites[np.random.randint(len(elites))]
                # Mix: with some probability sample directly around best more aggressively.
                if np.random.rand() < 0.35:
                    x0 = best_x
                    step_rel = cur_step_rel * 0.8
                else:
                    step_rel = cur_step_rel
                eps = np.random.randn(dim) * (step_rel * span_safe)
                candidates.append(clip(np.asarray(x0) + eps))

            # Evaluate candidates with strict budget enforcement
            for x in candidates:
                if eval_count >= budget or eval_count >= global_budget:
                    break
                y = evaluate(x)
                if y is None:
                    break
                # Add to population; prune later
                pop_x.append(np.asarray(x, dtype=float))
                pop_y.append(float(y))

            elitist_prune(pop_size)

            # Simple adaptation: if improved, nudge towards exploitation.
            if best_y < improved_before:
                main_step *= 0.92
                local_step = max(local_step, cur_step_rel * 0.5)
            else:
                main_step *= 0.98

        # --- Local exploitation phase ---
        # Coordinate-wise neighbor search around best, interleaved with small
        # Gaussian refinements.
        while eval_count < budget:
            remaining = budget - eval_count
            if remaining <= 0:
                break

            improved_before = best_y

            # Small Gaussian refinement batch around best
            batch = min(8 + dim, remaining)
            step_rel = local_step * (0.9 ** max(0, (eval_count - global_budget) // max(1, dim // 2 + 1)))
            step_rel = max(step_rel, 1e-6)

            candidates = list(sample_around(best_x, step_rel, n_samples=batch))
            # Evaluate a subset if near budget
            for x in candidates:
                if eval_count >= budget:
                    break
                y = evaluate(x)
                if y is None:
                    break
                pop_x.append(np.asarray(x, dtype=float))
                pop_y.append(float(y))
            elitist_prune(pop_size)

            # Coordinate-wise probing: try +/- local step along selected coordinates.
            # This can be expensive in high dimensions, so use a stride.
            if eval_count < budget:
                coords = np.arange(0, dim, coord_stride, dtype=int)
                # Randomize order for robustness
                if coords.size > 0:
                    np.random.shuffle(coords)

                step_abs = step_rel * span_safe
                # Ensure nonzero for probing dimensions with zero span
                step_abs = np.where(step_abs != 0, step_abs, 1e-9)

                for i in coords:
                    if eval_count >= budget:
                        break
                    # Probe +step
                    x_plus = best_x.copy()
                    x_plus[i] = x_plus[i] + step_abs[i]
                    x_plus = clip(x_plus)
                    y = evaluate(x_plus)
                    if y is not None:
                        pop_x.append(np.asarray(x_plus, dtype=float))
                        pop_y.append(float(y))
                    if eval_count >= budget:
                        break
                    # Probe -step
                    x_minus = best_x.copy()
                    x_minus[i] = x_minus[i] - step_abs[i]
                    x_minus = clip(x_minus)
                    y = evaluate(x_minus)
                    if y is not None:
                        pop_x.append(np.asarray(x_minus, dtype=float))
                        pop_y.append(float(y))

                    # Early break if we already improved significantly
                    if best_y < improved_before:
                        break

                elitist_prune(pop_size)

            # Adapt local step-size based on improvement
            if best_y < improved_before:
                local_step *= 0.85
                # Keep main_step aligned with local progress
                main_step = max(main_step * 0.9, local_step * 1.5)
            else:
                local_step *= 0.95
                main_step *= 0.97

            # If step sizes become tiny and no improvement, allow to finish quickly.
            if local_step < 1e-9 and main_step < 1e-9:
                break

        # If population has entries but best_x somehow missing, infer from pop
        if best_x is None and pop_x:
            idx = int(np.argmin(np.asarray(pop_y, dtype=float)))
            best_x = pop_x[idx]
            best_y = float(pop_y[idx])

        return np.asarray(best_x, dtype=float), float(best_y)

    @staticmethod
    def _get_bounds(func, dim):
        # Read bounds from func.lower/func.upper or func.bounds.lb/ub.
        lb = None
        ub = None

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        # Fallback: try attributes directly
        if lb is None or ub is None:
            # If bounds cannot be determined, assume [0,1]^dim.
            lb = np.zeros(dim, dtype=float)
            ub = np.ones(dim, dtype=float)

        # Ensure correct shape
        lb = np.reshape(lb, (dim,))
        ub = np.reshape(ub, (dim,))

        # Replace non-finite with safe defaults
        lb = np.where(np.isfinite(lb), lb, 0.0)
        ub = np.where(np.isfinite(ub), ub, 1.0)

        # If any ub < lb due to bad input, swap them.
        swap = ub < lb
        if np.any(swap):
            tmp = lb.copy()
            lb[swap] = ub[swap]
            ub[swap] = tmp[swap]

        return lb, ub
