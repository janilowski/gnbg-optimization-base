import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimizer.
# It blends a population-based global search with local refinement around the best
# solutions found so far. The algorithm is designed to be robust across dimensions
# while respecting a strict evaluation budget.
# Search state: The algorithm maintains a small population of candidate points,
# tracks the best point/value seen, and keeps a notion of the current step size
# (mutation scale) that shrinks over time. No external state is stored.
# Candidate generation: Each generation samples candidates using Gaussian mutations
# around population members. Mutation scale starts relatively large and then
# decreases. Additionally, a simple coordinate-wise local probe is used around
# the current best to refine.
# Selection and replacement: After evaluating offspring, selection keeps the best
# candidates (elitism) and replaces the population using an “(mu + lambda)”-like
# strategy based on objective values.
# Adaptation: The mutation scale is adapted based on progress: if improvements
# happen, the scale contracts slower; if not, it contracts faster. A small amount
# of diversity is preserved by injecting occasional random points.
# Exploration mechanisms: Global exploration comes from population-wide Gaussian
# sampling and occasional random re-initialization within bounds when stagnation
# is detected.
# Exploitation mechanisms: Local refinement probes several directions (coordinates)
# around the current best using diminishing step sizes, and also uses Gaussian
# mutation tightly around the best.
# Boundary handling: All candidate points are clipped to the provided bounds.
# If a function does not provide per-dimension bounds but only a scalar, the code
# still supports numpy broadcasting.
# Budget strategy: The algorithm converts the input budget into a maximum number
# of objective evaluations and never calls the objective more than that. It
# determines how many candidates can be evaluated per generation and adjusts
# the final generation to fit the remaining budget.
# Closest known influences: The overall structure resembles (mu+lambda) evolution
# strategies with a decreasing mutation step and simple local search around the
# current elite, but it is implemented from scratch and kept intentionally minimal.
# Novelty or unusual aspects: The local coordinate probing around the best is an
# inexpensive addition that can quickly improve accuracy without gradient information.
# Failure modes: If the objective is highly irregular or extremely noisy, the step
# size adaptation may become unstable; budget exhaustion can leave only coarse
# solutions in very hard problems. Bounds handling assumes finite limits if
# provided.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")

    def __call__(self, func):
        d = self.dim
        lb, ub = self._get_bounds(func)

        # Ensure finite bounds if provided (clipping still works for inf but
        # sampling scales become problematic). We'll handle inf by falling back
        # to wide defaults.
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        if lb.shape == () and d != 1:
            lb = np.full(d, lb.item())
        if ub.shape == () and d != 1:
            ub = np.full(d, ub.item())
        lb = np.broadcast_to(lb, (d,))
        ub = np.broadcast_to(ub, (d,))

        # If bounds are infinite, define a reasonable finite box for sampling.
        finite_lb = np.isfinite(lb)
        finite_ub = np.isfinite(ub)
        has_finite = finite_lb & finite_ub
        if not np.any(has_finite):
            # No usable finite bounds: choose a default region [-5, 5]
            lb = np.full(d, -5.0)
            ub = np.full(d, 5.0)
        else:
            # Fill missing finite side(s)
            span_default = 10.0
            lb = np.where(finite_lb, lb, ub - span_default)
            ub = np.where(finite_ub, ub, lb + span_default)

        # Helpers
        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        def rand_in_bounds(n=1):
            u = np.random.random((n, d))
            return lb + u * (ub - lb)

        def eval_batch(X, remaining):
            ys = []
            for x in X:
                if remaining[0] <= 0:
                    break
                y = float(func(x))
                ys.append(y)
                remaining[0] -= 1
            return np.asarray(ys, dtype=float)

        remaining = [self.budget]

        # Population sizes. Kept small for compactness and to reduce overhead.
        # We will adapt if the remaining budget is tight.
        mu = max(2, min(8, d + 1))
        # Offspring per generation. Keep modest.
        lam = max(4, min(14, 2 * d + 2))

        # Initialize population:
        # - include some random points
        # - include a best-guess center point
        center = (lb + ub) / 2.0
        center = clip(center)

        population = []
        population.append(center)
        # Add a few random points
        extra = max(0, mu - 1)
        if extra > 0:
            population.extend(rand_in_bounds(extra).tolist())
        population = np.asarray(population[:mu], dtype=float)

        # Evaluate initial population
        y_pop = np.empty((len(population),), dtype=float)
        # Evaluate until budget exhaustion
        init_n = min(len(population), remaining[0])
        if init_n <= 0:
            # Degenerate: no evaluation allowed; return a clipped center.
            return center, float("inf")

        y_pop[:init_n] = eval_batch(population[:init_n], remaining)
        if init_n < len(population):
            # Fill with +inf for unevaluated candidates
            y_pop[init_n:] = np.inf
        # Consider only evaluated portion for best
        best_idx = int(np.argmin(y_pop))
        best_x = population[best_idx].copy()
        best_y = float(y_pop[best_idx])

        # Step size schedule
        # Start with a fraction of the box diagonal; shrink over time.
        box_span = (ub - lb)
        box_span = np.where(box_span > 0, box_span, 1.0)
        diag = float(np.linalg.norm(box_span))
        sigma0 = 0.2 * diag / np.sqrt(d) if diag > 0 else 0.1
        sigma = sigma0 if sigma0 > 0 else 0.1

        # Progress tracking
        best_improve = 0
        last_best = best_y

        # Estimate number of generations from budget
        # Each generation evaluates up to lam candidates.
        # We also do occasional local probes.
        gen = 0
        while remaining[0] > 0:
            gen += 1
            # Stagnation detection
            if best_y < last_best - 1e-12:
                best_improve += 1
                last_best = best_y
            else:
                best_improve = max(0, best_improve - 1)

            # Mutation scales: shrink with progress; avoid sigma collapsing too quickly.
            # If we improved recently, shrink slower; else faster.
            shrink = 0.98 if best_improve > 0 else 0.93
            sigma = max(sigma * shrink, 1e-12)

            # Selection indices: use best half of population for sampling centers.
            order = np.argsort(y_pop)
            mu_eff = min(mu, len(order))
            parents = population[order[: max(2, mu_eff // 2)]]
            n_par = len(parents)

            # Determine how many offspring we can afford
            max_offspring = min(lam, remaining[0])

            if max_offspring <= 0:
                break

            # Candidate generation (Gaussian mutations around selected parents)
            # To reduce correlation, mix parent selection and random rotation via per-dim noise.
            idx_par = np.random.randint(0, n_par, size=max_offspring)
            Z = np.random.randn(max_offspring, d)
            # Slightly bias towards direction of negative objective improvement:
            # use the vector from parent to current best as a drift component.
            drift = (best_x - parents[idx_par])
            X_off = parents[idx_par] + sigma * Z + 0.15 * (sigma / (diag + 1e-12)) * drift
            X_off = clip(X_off)

            # Evaluate offspring
            y_off = eval_batch(X_off, remaining)
            if len(y_off) == 0:
                break

            # Combine and select new population (elitism)
            # If offspring fewer than max_offspring due to budget, handle length.
            n_off = len(y_off)
            combined_X = np.vstack([population, X_off[:n_off]])
            combined_y = np.concatenate([y_pop, y_off])

            sel = np.argsort(combined_y)[:mu]
            population = combined_X[sel].copy()
            y_pop = combined_y[sel].copy()

            # Update global best
            cur_best_idx = int(np.argmin(y_pop))
            cur_best_y = float(y_pop[cur_best_idx])
            if cur_best_y < best_y:
                best_y = cur_best_y
                best_x = population[cur_best_idx].copy()

            # Occasional local probing around best (coordinate-wise)
            # Only if we have spare budget and sigma is not tiny.
            if remaining[0] > 0 and sigma > 1e-8 and (gen % 2 == 0):
                # Probe a small subset of coordinates for efficiency.
                # Prefer larger coordinates' impact by cycling through indices.
                k = min(d, 5) if d > 1 else 1
                # Deterministic-ish selection: stride with gen
                stride = (gen % d) + 1
                coords = [(i * stride) % d for i in range(k)]
                coords = list(dict.fromkeys(coords))  # unique
                # Diminishing probe radius
                probe_steps = [sigma, sigma * 0.5, sigma * 0.25]
                # Create probe points
                probe_points = []
                for c in coords:
                    for s in probe_steps:
                        x = best_x.copy()
                        x[c] = x[c] - s
                        probe_points.append(x)
                        x = best_x.copy()
                        x[c] = x[c] + s
                        probe_points.append(x)

                # Add a tight random perturbation occasionally
                if (gen % 3) == 0:
                    for _ in range(2):
                        rr = np.random.randn(d) * (0.25 * sigma)
                        probe_points.append(clip(best_x + rr))

                # Evaluate probes with remaining budget
                if remaining[0] > 0 and len(probe_points) > 0:
                    # Clip and sample only what fits
                    # (eval_batch already checks budget; keep list short anyway)
                    probe_points = np.asarray([clip(x) for x in probe_points], dtype=float)
                    # Evaluate up to remaining
                    # To avoid dtype conversions repeatedly, slice to remaining if possible.
                    n_probe_fit = min(len(probe_points), remaining[0])
                    if n_probe_fit > 0:
                        y_probe = eval_batch(probe_points[:n_probe_fit], remaining)
                        if len(y_probe) > 0:
                            j = int(np.argmin(y_probe))
                            if float(y_probe[j]) < best_y:
                                best_y = float(y_probe[j])
                                best_x = probe_points[j].copy()

            # Exploration injection on stagnation
            if remaining[0] > 0 and best_improve == 0 and (gen % 3 == 0):
                # Replace worst individuals with random points
                worst = np.argsort(y_pop)[-max(1, mu // 3):]
                n_rep = len(worst)
                # Fit evaluation budget: random replacement would require evaluation.
                # We'll sample random points and evaluate them; if budget is small,
                # evaluate only the number that fits.
                if n_rep > 0:
                    n_fit = min(n_rep, remaining[0])
                    if n_fit > 0:
                        X_new = rand_in_bounds(n_fit)
                        y_new = eval_batch(X_new, remaining)
                        # Place evaluated replacements
                        # Map into worst indices subset
                        rep_idx = worst[-n_fit:]
                        population[rep_idx] = X_new[:n_fit]
                        y_pop[rep_idx] = y_new

                        cur_best_idx = int(np.argmin(y_pop))
                        cur_best_y = float(y_pop[cur_best_idx])
                        if cur_best_y < best_y:
                            best_y = cur_best_y
                            best_x = population[cur_best_idx].copy()

        return best_x, best_y

    def _get_bounds(self, func):
        # Priority:
        # 1) func.lower / func.upper
        # 2) func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return getattr(func, "lower"), getattr(func, "upper")
        if hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return getattr(b, "lb"), getattr(b, "ub")
        raise AttributeError(
            "Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub"
        )
