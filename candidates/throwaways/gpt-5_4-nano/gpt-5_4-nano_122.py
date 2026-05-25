# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, derivative-free black-box minimization algorithm
# based on an adaptive coordinate-wise Evolution Strategy (ES) with heavy-tailed
# (Cauchy-like) mutations and local quadratic interpolation of the best coordinate.
# Search state: Maintains a current best point x_best, its value y_best, and
# a per-dimension step-size vector sigma. Tracks remaining evaluations to never
# exceed the provided budget. Uses an internal evaluation counter.
# Candidate generation: Each iteration samples a small population by mutating
# x_best with scaled random directions drawn from a heavy-tailed distribution,
# then optionally performs 1D exploratory moves along the best improving coordinate
# using a bounded bracketing attempt.
# Selection and replacement: Selects the best candidate among the sampled points
# (minimization objective) and replaces x_best if an improvement is found.
# Adaptation: Step sizes are adapted using simple success-rate logic (increase on
# improvement, decrease otherwise), with an additional shrink when progress stalls.
# Exploration mechanisms: Heavy-tailed mutations help jump to new basins early;
# occasional 1D probing helps refine promising coordinates.
# Exploitation mechanisms: Uses x_best as the mutation center and gradually shrinks
# sigma to focus search around the best solution.
# Boundary handling: Uses reflection to keep points within bounds, avoiding bias
# at edges while preserving continuity of the search trajectory.
# Budget strategy: Iteratively generates candidates until the evaluation counter reaches
# budget. The last iteration may generate fewer candidates if needed.
# Closest known influences: Inspired by classic (μ+λ)/(1+λ) ES and coordinate-wise
# local refinement, but with heavy-tailed mutations for robustness.
# Novelty or unusual aspects: Adds lightweight coordinate bracketing with a small
# local quadratic guess to guide 1D probing without extra objective structure.
# Failure modes: In very noisy or extremely flat landscapes, step-size adaptation may
# oscillate; budget exhaustion can occur before sufficient refinement. In ill-conditioned
# bounds, reflection may cause repeated boundary interactions, slowing progress.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._get_bounds(func)
        dim = self.dim
        budget = self.budget

        # Degenerate bounds handling
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != dim or ub.size != dim:
            # If bounds are scalar-like, broadcast.
            if lb.size == 1:
                lb = np.full(dim, float(lb[0]))
            if ub.size == 1:
                ub = np.full(dim, float(ub[0]))
        # Ensure lb <= ub
        swap = lb > ub
        if np.any(swap):
            lb2 = lb.copy()
            lb[swap], ub[swap] = ub[swap], lb2[swap]

        # Helper: reflect into bounds
        def reflect(x):
            # Reflect across [lb, ub] per coordinate
            # Works even if x goes far beyond due to modulo + reflection.
            w = ub - lb
            # If any width is zero, clamp.
            zero = w == 0
            if np.any(zero):
                x = x.copy()
                x[zero] = lb[zero]
                # Only reflect non-zero dimensions
                idx = ~zero
                if not np.any(idx):
                    return x
                w = w[idx]
                y = x[idx] - lb[idx]
                # Map to [0, 2w) then fold
                # Use remainder to handle large excursions.
                y = np.mod(y, 2.0 * w)
                over = y > w
                y[over] = 2.0 * w[over] - y[over]
                x[idx] = lb[idx] + y
                return x

            y = x - lb
            w2 = 2.0 * (ub - lb)
            # modulo in [0, 2w)
            y = np.mod(y, w2)
            over = y > (ub - lb)
            y[over] = 2.0 * (ub - lb)[over] - y[over]
            return lb + y

        # Helper: evaluate with budget accounting
        n_eval = 0

        def eval_x(x):
            nonlocal n_eval
            if n_eval >= budget:
                return np.inf
            x = np.asarray(x, dtype=float)
            val = float(func(x))
            n_eval += 1
            return val

        # Robust initialization: random point(s), pick best
        # Start from mid with jitter; avoids dependence on arbitrary initial x.
        rng = np.random
        center = (lb + ub) / 2.0
        width = ub - lb
        # If width is 0 in some dims, sigma should be 0 there
        sigma0 = 0.3 * np.where(width > 0, width, 0.0)
        # Avoid all-zero sigma
        sigma0 = np.where(sigma0 == 0, 1e-12, sigma0)

        # Small initial sampling budget (up to 2*dim, but leave room)
        init_tries = min(max(1, 2 * dim), max(1, budget // 3))
        init_tries = min(init_tries, budget)

        x_best = reflect(center + rng.normal(0.0, 1.0, size=dim) * sigma0)
        y_best = eval_x(x_best)

        for _ in range(init_tries - 1):
            if n_eval >= budget:
                break
            x = reflect(center + rng.normal(0.0, 1.0, size=dim) * sigma0)
            y = eval_x(x)
            if y < y_best:
                x_best, y_best = x, y

        # ES-like step-size vector (adaptive)
        # Use per-dimension sigma to fit varying scales.
        sigma = sigma0.copy()
        # Success-rate tracker
        successes = 0
        trials = 0

        # Mutation distribution: Cauchy-like heavy tail
        # generate eps ~ t/|N(0,1)| to approximate Student-t-ish behavior.
        def heavy_t_noise(size):
            # Avoid division by zero
            n = rng.standard_normal(size)
            d = np.abs(rng.standard_normal(size)) + 1e-12
            return n / d

        # Coordinate probing: lightweight 1D bracketing around current best
        # using a small set of samples; uses quadratic guess from 3 points.
        def probe_coordinate(x0, coord, y0, step):
            # Try x0 +/- step, bracket and select the best.
            # Returns (x_new, y_new) if improvement else (x0, y0)
            nonlocal n_eval
            if n_eval >= budget:
                return x0, y0
            step = float(step)
            if step == 0.0:
                return x0, y0
            # Candidate 1D points
            candidates = []
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[coord] = x_plus[coord] + step
            x_minus[coord] = x_minus[coord] - step
            x_plus = reflect(x_plus)
            x_minus = reflect(x_minus)
            y_plus = eval_x(x_plus) if n_eval < budget else np.inf
            y_minus = eval_x(x_minus) if n_eval < budget else np.inf
            # Choose best among {x_minus, x0, x_plus}
            best_local_x = x0
            best_local_y = y0
            if y_minus < best_local_y:
                best_local_x, best_local_y = x_minus, y_minus
            if y_plus < best_local_y:
                best_local_x, best_local_y = x_plus, y_plus
            # Optional quadratic refinement if both sides sampled within budget.
            # Use three points at (-h, 0, +h) if we didn't collapse due to reflection.
            if (n_eval < budget) and best_local_x is not x0 and best_local_y < y0:
                # Determine effective offsets after reflection by reconstructing along coord
                # If x0, x_plus, x_minus all differ due to reflection, still do a cautious step.
                h = step
                # Quadratic minimizer for f(-h)=y_minus, f(0)=y0, f(h)=y_plus:
                # x* = (h*(y_minus - y_plus)) / (2*(y_minus - 2*y0 + y_plus))
                denom = (y_minus - 2.0 * y0 + y_plus)
                if abs(denom) > 1e-18:
                    t = (h * (y_minus - y_plus)) / (2.0 * denom)
                    t = float(np.clip(t, -2.0 * h, 2.0 * h))
                    x_ref = x0.copy()
                    x_ref[coord] = x_ref[coord] + t
                    x_ref = reflect(x_ref)
                    y_ref = eval_x(x_ref) if n_eval < budget else np.inf
                    if y_ref < best_local_y:
                        return x_ref, y_ref
            return best_local_x, best_local_y

        # Main loop: adaptive (1+λ) with occasional 1D probing
        # Keep population small for speed and budget compliance.
        while n_eval < budget:
            # Choose λ based on dimension and remaining budget
            remaining = budget - n_eval
            # Target about 2-4*dim evaluations over some span; keep small per iteration.
            lam = int(np.clip(2 + dim // 2, 3, 15))
            lam = min(lam, remaining)

            # Generate candidates
            candidates = []
            for _ in range(lam):
                if n_eval >= budget:
                    break
                # Heavy-tailed isotropic-ish mutation around x_best with per-dim sigma
                z = heavy_t_noise(dim)
                # Scale noise; use normal as well to stabilize
                z2 = rng.standard_normal(dim)
                # Combine to create correlated but robust steps
                direction = 0.65 * z + 0.35 * z2
                step = direction * sigma
                x = reflect(x_best + step)
                y = eval_x(x)
                candidates.append((y, x))

            if not candidates:
                break

            # Select best
            candidates.sort(key=lambda t: t[0])
            y_c, x_c = candidates[0]

            trials += len(candidates)
            if y_c < y_best:
                x_best, y_best = x_c, y_c
                successes += 1

            # Adapt sigma with simple success logic
            # Shrink more aggressively when no improvement to focus.
            if successes > 0 and (successes / max(1, trials)) > 0.2:
                # Increase slightly to encourage exploration
                sigma *= 1.12
            else:
                # Decrease, but not too fast
                sigma *= 0.82

            # Ensure sigma is within reasonable range relative to bounds
            w = ub - lb
            max_sigma = 0.5 * np.where(w > 0, w, 0.0)
            max_sigma = np.where(max_sigma == 0, 1e-12, max_sigma)
            # If very small widths, keep sigma tiny
            sigma = np.minimum(sigma, max_sigma)
            sigma = np.maximum(sigma, 1e-12)

            # Occasional coordinate probing for local refinement
            if n_eval < budget and dim > 0:
                # Probe at most once per a few iterations by using remaining fraction heuristic
                # Prefer probing dimension with the largest sigma (most uncertainty).
                if rng.rand() < 0.35:
                    coord = int(np.argmax(sigma))
                    # Probe step-size magnitude
                    probe_step = sigma[coord] * (0.8 + 0.4 * rng.rand())
                    x_new, y_new = probe_coordinate(x_best, coord, y_best, probe_step)
                    trials += 1
                    if y_new < y_best:
                        x_best, y_best = x_new, y_new
                        successes += 1
                        # If success, expand a touch in that coordinate
                        sigma[coord] *= 1.15

            # Additional stall damping: if close to optimum in value, shrink
            if np.all(sigma < 1e-9):
                break

        return x_best, y_best

    @staticmethod
    def _get_bounds(func):
        # Bounds can be stored as:
        # - func.lower / func.upper (array-like)
        # - func.bounds.lb / func.bounds.ub (array-like)
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return np.asarray(func.lower, dtype=float), np.asarray(func.upper, dtype=float)
        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return np.asarray(b.lb, dtype=float), np.asarray(b.ub, dtype=float)
        if hasattr(func, "lb") and hasattr(func, "ub"):
            return np.asarray(func.lb, dtype=float), np.asarray(func.ub, dtype=float)
        # If bounds are missing, fall back to [-1, 1]
        # (Still respects budget; best-effort robustness.)
        # Dim is not available here, so return scalars for broadcasting.
        return np.array([-1.0]), np.array([1.0])
