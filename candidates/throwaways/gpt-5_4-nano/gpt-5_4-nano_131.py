import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm combining
# random direction sampling with adaptive local search. It maintains a
# current best solution and periodically performs short “coordinate-like”
# refinements around it, using an annealed step size.
# Search state: Tracks best_x, best_y (current incumbent), current evaluation
# count, and an adaptive step_scale that shrinks over time. Uses a small
# population of recent candidates to stabilize selection.
# Candidate generation: Each iteration proposes a batch of candidates by
# sampling random directions and also a few structured candidates by
# perturbing the best along coordinate directions with randomized signs.
# Selection and replacement: Every proposed candidate updates the incumbent
# if it improves the objective. A small ring buffer of recent candidates
# guides occasional restarts and step-size scaling based on observed
# improvements.
# Adaptation: The step size decreases with progress (budget fraction) and
# increases slightly if improvements stall to encourage exploration.
# Exploration mechanisms: Early-stage broad sampling from the full domain and
# random direction perturbations around the incumbent.
# Exploitation mechanisms: Later-stage local refinements around the incumbent
# with smaller structured perturbations, plus “resampled” candidates when
# improvements are detected.
# Boundary handling: Any candidate is clipped to the feasible bounds derived
# from func.lower/upper or func.bounds.lb/ub.
# Budget strategy: Never exceeds the provided evaluation budget. The number
# of evaluations per outer cycle is computed so the total stays within budget.
# Closest known influences: Related to CMA-ES/ES ideas (sampling + incumbent
# learning), but simplified to be dimension-robust and implementation-light.
# Novelty or unusual aspects: Uses a two-phase schedule (batch global-ish
# sampling then local coordinate-like refinements) with an annealed step
# size and minimal state, without external dependencies.
# Failure modes: If the objective is highly discontinuous or extremely
# noisy, small step sizes may cause stagnation; the algorithm mitigates
# with occasional exploratory increases when no improvement is observed.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        if budget <= 0:
            raise ValueError("budget must be positive")

        lb, ub = self._get_bounds(func, dim)
        # Ensure numeric and shapes
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        if lb.shape != (dim,) or ub.shape != (dim,):
            lb = np.broadcast_to(lb, (dim,)).copy()
            ub = np.broadcast_to(ub, (dim,)).copy()

        # Handle degenerate bounds: if lb==ub, that coordinate is fixed.
        span = ub - lb
        span_safe = np.where(span == 0.0, 1.0, span)

        def clip(x):
            # Clip to feasible bounds; preserves shape (dim,)
            return np.minimum(ub, np.maximum(lb, x))

        def normed_unit_vector(x):
            n = np.linalg.norm(x)
            if n == 0:
                return x
            return x / n

        evals = 0

        # Initialize incumbent: sample a starting point uniformly in the box.
        x_best = lb + np.random.rand(dim) * (ub - lb)
        x_best = clip(x_best)
        y_best = float(func(x_best))
        evals += 1

        # Step scale: start with a meaningful fraction of the domain.
        # Use median span to avoid domination by outliers.
        domain_scale = float(np.median(np.abs(span_safe)))
        # If domain_scale is extremely small, keep a tiny nonzero scale.
        domain_scale = max(domain_scale, 1e-12)
        step_scale = 0.35 * domain_scale

        # Recent improvement tracking for adaptation
        no_improve_streak = 0
        improvement_history = []

        # Small candidate buffer (keep last few best candidates)
        # to stabilize selection and occasionally use as anchors.
        k_buffer = 5
        xs_buf = [x_best.copy()]
        ys_buf = [y_best]

        # Budget management: we will iterate in outer cycles; within each cycle,
        # evaluate a batch whose size is chosen to not exceed budget.
        # We use 1 incumbent evaluation already, so remaining budget:
        remaining = budget - evals
        if remaining <= 0:
            return x_best, y_best

        # Define a schedule for batch sizes: larger early, smaller later.
        # Total candidates evaluated across all cycles must stay <= remaining.
        # We also cap total cycles to keep overhead low.
        max_cycles = 30
        cycles = min(max_cycles, max(1, remaining))  # at most one eval per cycle

        for c in range(cycles):
            # Stop if budget exhausted
            if evals >= budget:
                break

            # Progress in [0,1]
            t = evals / budget
            # Anneal step size: shrink over time (but not too aggressively).
            # Add a floor to reduce collapse due to numerical issues.
            anneal = (1.0 - t)
            step = step_scale * (0.05 + 0.95 * anneal)

            # Batch size for this cycle:
            # Early: 2-6 candidates, late: 1-3 candidates.
            if t < 0.5:
                batch_max = min(6, budget - evals)
                batch_min = 2
            else:
                batch_max = min(4, budget - evals)
                batch_min = 1
            batch = batch_min if batch_max == batch_min else np.random.randint(batch_min, batch_max + 1)

            # Decide anchors: incumbent and a couple from buffer.
            anchors = [x_best]
            if len(xs_buf) > 1 and np.random.rand() < 0.7:
                # Add up to 2 random buffered anchors
                idxs = np.random.choice(len(xs_buf), size=min(2, len(xs_buf)), replace=False)
                anchors.extend([xs_buf[i] for i in idxs])

            candidates = []

            # Exploration via random directions (spherical-ish sampling)
            for _ in range(batch):
                a = anchors[np.random.randint(len(anchors))]
                # Random direction
                d = np.random.normal(size=dim)
                d = normed_unit_vector(d)
                # Random step magnitude using a heavy-ish tail early
                # to allow occasional larger jumps.
                if t < 0.5:
                    mag = step * (0.5 + 1.5 * (np.random.rand() ** 0.5))
                else:
                    mag = step * (0.4 + 0.9 * np.random.rand())
                x = a + d * mag

                # Mild coordinate-structured perturbations mixed in
                if np.random.rand() < (0.35 if t < 0.5 else 0.55):
                    # Pick a few coordinates and perturb them
                    m = 1 if dim == 1 else np.random.randint(1, min(dim, 4) + 1)
                    coords = np.random.choice(dim, size=m, replace=False)
                    signs = np.random.choice([-1.0, 1.0], size=m)
                    # Scale coordinate steps smaller than directional ones
                    coord_step = 0.6 * step * (0.3 + 0.7 * np.random.rand())
                    x[coords] += signs * coord_step
                candidates.append(clip(x))

            # Structured local refinement around best: coordinate-like probes
            # (only if we can afford extra evaluations)
            extra_probes = 0
            if t > 0.3 and np.random.rand() < 0.6:
                # Evaluate one or two additional probes around the incumbent
                extra_probes = 1 if (budget - evals) == 1 else np.random.randint(1, 3)
                extra_probes = min(extra_probes, budget - evals - batch + 1)

            probe_candidates = []
            if extra_probes > 0:
                # Use random coordinate directions; probe both signs for a subset
                if dim == 1:
                    for _ in range(extra_probes):
                        sgn = np.random.choice([-1.0, 1.0])
                        probe_candidates.append(clip(x_best + sgn * step * (0.5 + 0.8 * np.random.rand())))
                else:
                    # Choose coordinates
                    m = min(dim, 2 * extra_probes)
                    coords = np.random.choice(dim, size=m, replace=False)
                    # For each probe, pick a coord and sign
                    for _ in range(extra_probes):
                        j = coords[np.random.randint(len(coords))]
                        sgn = np.random.choice([-1.0, 1.0])
                        probe_candidates.append(clip(x_best.copy().astype(float)))
                        probe_candidates[-1][j] += sgn * step * (0.25 + 0.9 * np.random.rand())

            # Evaluate candidates, never exceeding budget
            improved = False
            best_y_cycle = y_best
            local_count = 0

            for x in candidates:
                if evals >= budget:
                    break
                y = float(func(x))
                evals += 1
                local_count += 1
                if y < best_y_cycle:
                    best_y_cycle = y
                    improved = True
                    x_best = x
                    y_best = y

                # update buffers
                if len(xs_buf) < k_buffer:
                    xs_buf.append(x.copy())
                    ys_buf.append(y)
                else:
                    # Keep the best k_buffer values
                    # (insert if better than worst)
                    worst_i = int(np.argmax(ys_buf))
                    if y < ys_buf[worst_i]:
                        xs_buf[worst_i] = x.copy()
                        ys_buf[worst_i] = y

            # Evaluate probes if budget allows
            for x in probe_candidates:
                if evals >= budget:
                    break
                y = float(func(x))
                evals += 1
                local_count += 1
                if y < best_y_cycle:
                    best_y_cycle = y
                    improved = True
                    x_best = x
                    y_best = y

                if len(xs_buf) < k_buffer:
                    xs_buf.append(x.copy())
                    ys_buf.append(y)
                else:
                    worst_i = int(np.argmax(ys_buf))
                    if y < ys_buf[worst_i]:
                        xs_buf[worst_i] = x.copy()
                        ys_buf[worst_i] = y

            # Adaptation of step_scale based on improvement
            if improved:
                no_improve_streak = 0
                improvement_history.append(best_y_cycle)
                # If improvement is significant, allow slightly larger steps later within reason
                # by not shrinking too quickly.
                step_scale = max(step_scale * (0.85 + 0.15 * np.random.rand()), 1e-12)
            else:
                no_improve_streak += 1
                # Shrink modestly on stagnation, but occasionally re-expand to escape.
                step_scale = max(step_scale * 0.90, 1e-12)
                if no_improve_streak >= 4 and t < 0.9:
                    # Small restart: jump incumbent anchor closer to random points
                    # to avoid complete stagnation.
                    x_rand = lb + np.random.rand(dim) * (ub - lb)
                    x_rand = clip(x_rand)
                    if evals < budget:
                        y_rand = float(func(x_rand))
                        evals += 1
                        if y_rand < y_best:
                            x_best, y_best = x_rand, y_rand
                            no_improve_streak = 0
                    # Re-expand step_scale slightly
                    step_scale = min(step_scale * 1.35, domain_scale)

        return x_best, y_best

    @staticmethod
    def _get_bounds(func, dim):
        # Support func.lower/upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = getattr(func, "lower")
            ub = getattr(func, "upper")
            return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)

        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return np.asarray(b.lb, dtype=float), np.asarray(b.ub, dtype=float)

        raise AttributeError("Function must provide bounds via func.lower/upper or func.bounds.lb/ub")
