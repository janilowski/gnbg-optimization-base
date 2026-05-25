import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a robust black-box minimization strategy using a population-based
# evolutionary approach with intermittent local refinement. It maintains a small pool of
# candidate solutions, evaluates them within the given budget, and uses simple adaptive
# step sizes to balance exploration and exploitation.
# Search state: Tracks a population of points in the decision space, their objective values,
# the current best solution, and an evaluation counter to ensure the budget is never exceeded.
# Candidate generation: Creates new candidates by perturbing elite solutions with Gaussian
# noise whose scale adapts over time. Also includes occasional mutation from the current best
# and a lightweight coordinate-wise local improvement step.
# Selection and replacement: Uses elitist selection (keeps the best individuals). New candidates
# replace the worst members of the population if they improve upon them; otherwise the
# population is partially refreshed to avoid stagnation.
# Adaptation: Decreases the global mutation scale as the budget is consumed, and also adapts
# it based on whether recent improvements were observed.
# Exploration mechanisms: Population-level random perturbations, plus re-injection (random
# re-start of some individuals) when the search stagnates.
# Exploitation mechanisms: Biases sampling toward elites and applies local refinement around
# the best point by trying coordinate-wise +/- moves with decreasing step sizes.
# Boundary handling: All generated points are clipped to the provided bounds (lower/upper).
# Budget strategy: All function evaluations are counted explicitly; the algorithm stops
# generating new points once the evaluation budget is reached.
# Closest known influences: Inspired by CMA-ES-like ideas (elite-driven search and adaptive step),
# but implemented compactly without covariance estimation.
# Novelty or unusual aspects: Combines a compact elitist evolutionary loop with a simple
# coordinate-wise local search trigger, while strictly enforcing budget and using only numpy.
# Failure modes: If the objective is highly noisy or extremely flat, the method may rely on
# re-injection and still make limited progress; if bounds are very tight, clipping can lead
# to premature stagnation.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # ---- Read bounds (minimization) ----
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective must provide bounds via (lower, upper) or bounds.lb/bounds.ub")

        if lb.shape == () or ub.shape == ():
            lb = np.full(self.dim, float(lb))
            ub = np.full(self.dim, float(ub))

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError("Bounds dimensionality does not match dim")

        # Safety: ensure lb < ub; if equal, use constant coordinate.
        span = ub - lb
        span = np.where(span > 0, span, 0.0)
        # avoid divide-by-zero / NaNs later
        span_safe = np.where(span > 0, span, 1.0)

        # ---- Budget handling ----
        max_evals = max(1, self.budget)
        evals = 0

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_point(x):
            nonlocal evals
            if evals >= max_evals:
                # Do not exceed budget; return +inf so it will not be chosen.
                return np.inf
            y = func(x)
            evals += 1
            return float(y)

        # ---- Population setup ----
        # Choose a small population to keep evaluations reasonable across dimensions.
        # If budget is very small, fall back to minimal sampling.
        # pop_size is limited by remaining evaluations.
        pop_size = int(np.clip(max_evals // 6, 4, 20))
        pop_size = min(pop_size, max_evals)  # can't evaluate more than budget

        # Initialize: uniform random in bounds plus optionally an extra best guess at center.
        center = lb + 0.5 * span
        X = np.empty((pop_size, self.dim), dtype=float)
        for i in range(pop_size):
            if i == 0 and max_evals >= 2:
                # A deterministic-ish start at center helps on some functions.
                X[i] = center
            else:
                r = np.random.random(self.dim)
                X[i] = lb + r * span_safe
                # If span is zero for some dims, ensure exact clipping.
                X[i] = clip(X[i])

        vals = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            vals[i] = eval_point(X[i])

        best_idx = int(np.argmin(vals))
        best_x = X[best_idx].copy()
        best_y = float(vals[best_idx])

        # Initial mutation scale: fraction of the search box.
        # Use span_safe so zero-span dims don't cause NaNs.
        init_scale = 0.25
        global_sigma = init_scale * np.mean(span_safe)
        global_sigma = max(global_sigma, 1e-12)

        # Stagnation tracking / adaptation.
        no_improve_steps = 0
        improve_history = 0

        # Coordinate-wise local refinement parameters
        # (triggered occasionally, consumes a small number of evaluations).
        local_trigger_every = max(5, pop_size // 2)
        local_budget_fraction = 0.05  # local attempts are bounded

        # ---- Main loop ----
        # We'll generate candidates in batches until the budget is exhausted.
        while evals < max_evals:
            remaining = max_evals - evals
            # Scale decreases with progress to shift from exploration to exploitation.
            progress = evals / max_evals
            sigma = global_sigma * (1.0 - 0.85 * progress + 0.05)

            # Determine how many new points we can evaluate this round.
            # Keep batch size small for responsiveness.
            batch = int(np.clip(pop_size // 2, 2, 10))
            batch = min(batch, remaining)

            # ---- Candidate generation ----
            # Elites: sample parents from the best part of the population.
            elite_k = max(2, pop_size // 3)
            elite_idx = np.argsort(vals)[:elite_k]

            # Generate new population candidates.
            new_X = np.empty((batch, self.dim), dtype=float)
            for j in range(batch):
                # Choose parent with probability biased toward better individuals.
                if elite_k == 1:
                    parent = X[elite_idx[0]]
                else:
                    # Inverse-rank sampling: better ranks more likely.
                    ranks = np.arange(elite_k)
                    w = (elite_k - ranks).astype(float)
                    w /= w.sum()
                    p = elite_idx[np.random.choice(elite_k, p=w)]
                    parent = X[p]

                # Perturbation: Gaussian noise with a dimension-wise scaling to respect bounds span.
                noise = np.random.randn(self.dim)
                # dimension-wise scale: larger where bounds are wider
                dim_scale = 0.15 + 0.85 * (span_safe / np.max(span_safe))
                step = sigma * dim_scale * noise

                # Add a slight occasional direction from best to exploit.
                if np.random.random() < 0.3:
                    dir_to_best = best_x - parent
                    # normalized-ish
                    denom = np.linalg.norm(dir_to_best) + 1e-12
                    step += (sigma * 0.25) * (dir_to_best / denom) * np.random.randn(self.dim)

                x = clip(parent + step)

                # If the span is zero in all dims, clip collapses to a single point.
                new_X[j] = x

            # ---- Evaluation ----
            new_vals = np.empty(batch, dtype=float)
            for j in range(batch):
                new_vals[j] = eval_point(new_X[j])

            # ---- Selection and replacement (elitist) ----
            # Merge current pop with new points and keep best pop_size.
            X_all = np.vstack((X, new_X))
            vals_all = np.concatenate((vals, new_vals))

            # Select best pop_size indices
            keep_idx = np.argsort(vals_all)[:pop_size]
            X = X_all[keep_idx]
            vals = vals_all[keep_idx]

            # Update best
            cur_best_idx = int(np.argmin(vals))
            cur_best_y = float(vals[cur_best_idx])
            if cur_best_y + 1e-18 < best_y:
                best_y = cur_best_y
                best_x = X[cur_best_idx].copy()
                no_improve_steps = 0
                improve_history += 1
            else:
                no_improve_steps += 1

            # ---- Adaptation of global sigma ----
            # If improvements happen, slightly reduce sigma; else increase/refresh.
            if cur_best_y + 1e-18 < best_y:
                # (Handled above) but keep this stable
                pass
            if no_improve_steps >= 3:
                # Increase exploration a bit (and re-inject some random points).
                global_sigma = min(global_sigma * 1.15, 0.7 * np.mean(span_safe) + 1.0)
                no_improve_steps = 0

                # Re-inject a fraction of individuals at random positions
                reinj = max(1, pop_size // 6)
                idxs = np.argsort(vals)[-reinj:]  # worst
                for k, i in enumerate(idxs):
                    if evals >= max_evals:
                        break
                    r = np.random.random(self.dim)
                    x = lb + r * span_safe
                    x = clip(x)
                    X[i] = x
                    vals[i] = eval_point(x)

                if evals < max_evals:
                    best_idx = int(np.argmin(vals))
                    if float(vals[best_idx]) < best_y:
                        best_y = float(vals[best_idx])
                        best_x = X[best_idx].copy()

            # ---- Occasional local refinement around the best ----
            # Trigger based on iteration count implied by evals/pop_size.
            iter_count = evals // max(1, pop_size)
            if (iter_count > 0) and (iter_count % local_trigger_every == 0):
                # Limit local attempts to a small fraction of remaining budget.
                local_budget = int(min(max_evals - evals, max(1, local_budget_fraction * max_evals)))
                if local_budget > 0:
                    # Use diminishing step based on remaining progress.
                    local_progress = evals / max_evals
                    local_sigma = (0.15 + 0.35 * (1.0 - local_progress)) * np.mean(span_safe)
                    local_sigma = max(local_sigma, 1e-12)

                    # Coordinate-wise tries: pick a few random coordinates each time.
                    # Budget-aware: we try +/- on a coordinate until local_budget is used.
                    coords = np.random.permutation(self.dim)
                    attempts = 0
                    improved = False

                    for c in coords:
                        if attempts + 2 > local_budget:
                            break

                        for sign in (-1.0, 1.0):
                            if evals >= max_evals:
                                break
                            x = best_x.copy()
                            # Step magnitude proportional to coordinate span.
                            step = sign * local_sigma * (0.2 + 0.8 * (span_safe[c] / np.max(span_safe)))
                            if span_safe[c] == 0.0:
                                # If coordinate is fixed, skip movement.
                                continue
                            x[c] = x[c] + step
                            x = clip(x)
                            y = eval_point(x)
                            attempts += 1

                            if y + 1e-18 < best_y:
                                best_y = y
                                best_x = x.copy()
                                improved = True

                        if evals >= max_evals or attempts >= local_budget:
                            break

                    if improved:
                        # After local improvement, pull population slightly toward best
                        # (cheap exploitation by replacing one worst point).
                        worst_idx = int(np.argmax(vals))
                        if evals < max_evals:
                            # Move worst toward best by a random factor.
                            alpha = np.random.uniform(0.25, 0.75)
                            x_new = clip((1 - alpha) * X[worst_idx] + alpha * best_x)
                            y_new = eval_point(x_new)
                            X[worst_idx] = x_new
                            vals[worst_idx] = y_new
                        cur_best_idx = int(np.argmin(vals))
                        if float(vals[cur_best_idx]) < best_y:
                            best_y = float(vals[cur_best_idx])
                            best_x = X[cur_best_idx].copy()

            # Stop condition if budget exhausted
            if evals >= max_evals:
                break

        return best_x, best_y
