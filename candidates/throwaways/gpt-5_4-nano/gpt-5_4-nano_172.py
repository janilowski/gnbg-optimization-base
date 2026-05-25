import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, robust black-box minimization algorithm using
# a population-based, gradient-free evolutionary strategy (ES) with adaptive
# step sizes. It maintains multiple candidate solutions, samples offspring by
# adding Gaussian noise, and selects the best solutions to drive search toward
# lower objective values.
# Search state: Stores a population of candidate vectors, their objective
# values, a per-individual mutation scale, and bookkeeping of how many
# function evaluations have been used.
# Candidate generation: Offspring are created by perturbing selected parents
# with Gaussian noise. Each offspring step size is derived from its parent's
# mutation scale and may be further randomized to maintain diversity.
# Selection and replacement: After evaluating offspring, the algorithm merges
# parents and offspring, then keeps the best individuals (elitist truncation)
# to form the next generation.
# Adaptation: Mutation scales adapt using a simple success-based rule: if
# offspring improve upon their parents, step sizes are slightly increased;
# otherwise they are slightly decreased.
# Exploration mechanisms: High mutation scales and multiple offspring per
# generation provide global exploration early in the run.
# Exploitation mechanisms: Selection pressure and decreasing mutation scales
# encourage local refinement around the best found solutions.
# Boundary handling: Candidate positions are clipped to the provided bounds
# (read from func.lower/func.upper or func.bounds.lb/ub). This guarantees
# feasibility while allowing exploration near boundaries.
# Budget strategy: The algorithm strictly respects the evaluation budget by
# limiting the number of evaluations for initialization and each generation.
# It never calls the objective after the remaining budget reaches zero.
# Closest known influences: Inspired by (1+λ)-style ES and (μ+λ) selection
# principles, with lightweight adaptive step size akin to evolution strategies
# using success-based scaling.
# Novelty or unusual aspects: Uses a per-individual adaptive sigma with a
# minimal success rule and a careful budget-aware generation loop so it
# works reliably for small and large budgets across different dimensions.
# Failure modes: If the budget is extremely small, the algorithm may only
# evaluate the initial population; clipping can also concentrate samples on
# bounds if the optimum lies outside/at the boundary.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # ---- Bound extraction (supports multiple harness conventions) ----
        lower = None
        upper = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            # Expected: b.lb, b.ub
            lower = np.asarray(b.lb, dtype=float)
            upper = np.asarray(b.ub, dtype=float)

        if lower is None or upper is None:
            raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/ub.")

        lower = np.broadcast_to(lower, (self.dim,)).astype(float, copy=False)
        upper = np.broadcast_to(upper, (self.dim,)).astype(float, copy=False)
        span = upper - lower

        # If bounds are degenerate, avoid NaNs.
        span_safe = np.where(span != 0.0, span, 1.0)

        def clip(x):
            return np.minimum(upper, np.maximum(lower, x))

        # ---- Evaluation helper with strict budget tracking ----
        max_evals = max(1, self.budget)
        evals_used = 0

        def eval_one(x):
            nonlocal evals_used
            if evals_used >= max_evals:
                # In case of rounding errors, ensure we never exceed.
                return np.inf
            y = float(func(x))
            evals_used += 1
            return y

        # ---- Hyperparameters chosen to be robust across dims ----
        # Population size depends mildly on dimension but is capped for budget.
        # Using at least 2 keeps selection meaningful.
        # For very small budget, we'll effectively evaluate only a tiny number.
        dim = self.dim
        # Budget-aware population sizing: ensure at least 2 evaluations.
        # We'll reserve evaluations for at least one generation when possible.
        pop = int(np.clip(6 + dim // 2, 2, 40))
        pop = min(pop, max_evals)  # cannot evaluate more than budget in init
        # Offspring count per generation: proportional to pop, capped.
        lam = int(np.clip(2 * pop, 4, 80))
        # Elitist truncation
        elite_keep = max(1, pop // 2)

        # Initial mutation scale: fraction of the search range.
        sigma0 = 0.25 * span_safe / max(1.0, float(dim) ** 0.5)

        # ---- Initialize population uniformly in bounds ----
        # If sigma0 is tiny, search becomes near-deterministic; still fine.
        X = lower + np.random.rand(pop, dim) * span_safe
        X = clip(X)

        sigmas = np.tile(sigma0, (pop, 1))
        # Ensure positive scales for each coordinate
        sigmas = np.maximum(sigmas, 1e-12)

        y = np.empty(pop, dtype=float)
        for i in range(pop):
            y[i] = eval_one(X[i])

        # Track best-so-far (minimization)
        best_idx = int(np.argmin(y))
        best_x = X[best_idx].copy()
        best_y = float(y[best_idx])

        # ---- Budget-aware main loop ----
        # Each generation will use up to min(lam, remaining budget).
        # We also account for the fact that we evaluate offspring only.
        # Parents carry over implicitly via elitist selection.
        # A simple success rule updates each parent's sigma based on improvement.
        while evals_used < max_evals:
            remaining = max_evals - evals_used
            if remaining <= 0:
                break

            # Select parents via tournament selection (bias toward better, keep diversity).
            # Number of evaluated offspring:
            this_lam = min(lam, remaining)
            # But we also want some meaningful mixing; if remaining is very small,
            # we can still do one "generation" of only a few offspring.
            if this_lam <= 0:
                break

            # Determine tournament size based on population size
            tsize = 2 if pop < 6 else 3

            def tournament_select():
                # random indices
                cand = np.random.randint(0, pop, size=tsize)
                return cand[np.argmin(y[cand])]

            parents_idx = np.array([tournament_select() for _ in range(this_lam)], dtype=int)
            # Offspring generation: x_child = x_parent + N(0, sigma)
            # Use per-parent sigma and additionally randomize across coordinates.
            # Create offspring.
            X_off = np.empty((this_lam, dim), dtype=float)
            y_off = np.empty(this_lam, dtype=float)
            # Success flags to adapt sigmas for the corresponding parent individuals
            improved = np.zeros(this_lam, dtype=bool)

            for k in range(this_lam):
                p = parents_idx[k]
                # Add Gaussian noise scaled by sigma; allow mild global rescaling
                # to maintain robustness across dimensions.
                # Using a log-normal multiplier tends to keep sigma positive.
                global_mult = np.exp(np.random.normal(0.0, 0.25))
                noise = np.random.normal(0.0, 1.0, size=dim)
                step = sigmas[p] * global_mult * noise
                child = X[p] + step
                child = clip(child)

                X_off[k] = child
                yk = eval_one(child)
                y_off[k] = yk
                if yk < y[p]:
                    improved[k] = True

                if yk < best_y:
                    best_y = yk
                    best_x = child.copy()

            # ---- Adapt mutation scales (per-parent success-based) ----
            # For each parent that generated one or more offspring, update its sigma.
            # Simple multiplicative rule: successful -> increase, else decrease.
            # Blend over multiple offspring by using mean success for each parent.
            success_by_parent = np.zeros(pop, dtype=float)
            count_by_parent = np.zeros(pop, dtype=float)
            for k in range(this_lam):
                p = parents_idx[k]
                count_by_parent[p] += 1.0
                success_by_parent[p] += 1.0 if improved[k] else 0.0

            # Rates
            inc = 1.15
            dec = 0.85
            for p in range(pop):
                if count_by_parent[p] > 0:
                    rate = success_by_parent[p] / count_by_parent[p]  # in [0,1]
                    # If rate high, increase; if low, decrease; smooth via threshold.
                    if rate >= 0.34:
                        sigmas[p] *= inc
                    else:
                        sigmas[p] *= dec
                    sigmas[p] = np.maximum(sigmas[p], 1e-12)

            # ---- Selection and replacement (elitist truncation over combined pool) ----
            # Merge parents and offspring, keep best 'pop' individuals but prefer elitism.
            X_comb = np.vstack((X, X_off))
            y_comb = np.concatenate((y, y_off))
            # Sort by objective (min)
            order = np.argsort(y_comb)
            # Keep best pop
            keep = min(pop, X_comb.shape[0])
            order = order[:keep]
            X = X_comb[order]
            y = y_comb[order]
            # Update sigmas for kept individuals.
            # For parents that survive, keep their corresponding sigmas.
            # For offspring, assign sigma sampled near its parent's current sigma.
            # Since we do not store parent sigma per offspring in arrays, approximate:
            # compute sigma as fraction of coordinate span scaled by parent's sigma if possible.
            # We'll use: sigma_child = sigmas[parent] * global_mult_effect is not stored;
            # so approximate using sigma0 decay based on generation progress.
            # Better: reconstruct by reusing parents_idx mapping for each offspring and
            # assigning the exact sigma of the generating parent scaled by a random factor.
            # To do that, we need mapping from kept individuals to whether they are offspring.
            # We'll rebuild using indices relative to merged arrays.
            merged_size = X_comb.shape[0]
            # Determine which kept indices correspond to offspring:
            # Parents indices [0, pop-1], offspring indices [pop, pop+this_lam-1]
            new_sigmas = np.empty_like(X, dtype=float)
            for i_new, idx in enumerate(order):
                if idx < pop:
                    new_sigmas[i_new] = sigmas[idx]
                else:
                    off_pos = idx - pop  # 0..this_lam-1
                    p = parents_idx[off_pos]
                    # Approximate sigma for the child by using the parent's sigma (small random jitter)
                    jitter = np.exp(np.random.normal(0.0, 0.15))
                    new_sigmas[i_new] = sigmas[p] * jitter
            sigmas = np.maximum(new_sigmas, 1e-12)

            # Update best index
            best_idx = int(np.argmin(y))
            if float(y[best_idx]) < best_y:
                best_y = float(y[best_idx])
                best_x = X[best_idx].copy()

            # Light exploitation boost: occasionally move the best individual slightly
            # toward the current best direction by re-sampling around best_x.
            # This helps with low budgets and avoids stagnation.
            # Only spend a tiny amount of evaluations.
            if evals_used < max_evals:
                # With small probability, perform 1 local refinement around best_x.
                # Use at most 1 eval to keep budget compliance.
                if np.random.rand() < 0.25:
                    child = clip(best_x + np.random.normal(0.0, 0.5, size=dim) * sigma0)
                    yk = eval_one(child)
                    if yk < best_y:
                        best_y = yk
                        best_x = child.copy()
                        # Optionally inject into population by replacing worst
                        worst = int(np.argmax(y))
                        X[worst] = child
                        y[worst] = yk
                        sigmas[worst] = np.maximum(sigma0, 1e-12)

        return best_x, best_y
