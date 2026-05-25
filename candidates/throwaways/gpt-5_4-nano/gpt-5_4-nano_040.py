# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm using a
# population-based evolutionary strategy (ES) with self-adaptive step sizes
# per individual. It supports arbitrary dimensions and only uses function
# evaluations from the provided budget.
# Search state: A population of candidate solutions plus per-individual
# mutation step sizes (sigma). Tracks the global best solution/value found.
# Candidate generation: Each iteration samples Gaussian mutations around
# current individuals using their own sigmas, producing offspring.
# Selection and replacement: Uses (μ, λ) selection: offspring compete by
# fitness; the best μ among combined parents+offspring become the next
# generation. The global best is updated from the evaluated candidates.
# Adaptation: Step sizes self-adapt using a log-normal rule
# (sigma *= exp(tau * N(0,1) + tau2 * N_i(0,1))) which increases adaptively
# according to the randomness of offspring.
# Exploration mechanisms: Random Gaussian mutations with sigmas provide
# exploration, and a small occasional "restart-like" increase of sigma helps
# escape stagnation.
# Exploitation mechanisms: As selection prefers better individuals, the
# distribution concentrates around low objective regions; sigmas shrink
# gradually due to selection pressure and log-normal adaptation.
# Boundary handling: Mutations are reflected at the bounds to keep candidates
# feasible while preserving diversity better than naive clipping.
# Budget strategy: Computes an evaluation budget cap and ensures every
# objective call is counted; stops exactly when the budget is exhausted or
# no evaluations remain. No evaluation beyond budget.
# Closest known influences: Inspired by (μ, λ)-ES with self-adaptive step sizes
# and simple restarts; designed to be lightweight and robust for black-box
# benchmarking.
# Novelty or unusual aspects: Uses reflection boundary handling and per-
# individual sigma adaptation with a small stagnation-driven sigma boost.
# Failure modes: In very noisy or highly non-smooth problems, selection
# may mislead and sigmas can collapse; stagnation detection attempts to
# mitigate this by temporarily increasing sigmas.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # --- Bounds acquisition (required) ---
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("func must provide bounds via (lower, upper) or func.bounds.lb/ub")

        lb = np.broadcast_to(lb, (self.dim,)).astype(float)
        ub = np.broadcast_to(ub, (self.dim,)).astype(float)
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            raise ValueError("Bounds must be finite.")
        if np.any(ub <= lb):
            raise ValueError("Each upper bound must be greater than lower bound.")

        # --- Evaluation budget management ---
        max_evals = max(1, self.budget)
        evals_used = 0

        def eval_one(x):
            nonlocal evals_used
            if evals_used >= max_evals:
                # Strictly never exceed the provided budget.
                # Raise to catch accidental overuse.
                raise RuntimeError("Evaluation budget exceeded.")
            evals_used += 1
            return float(func(x))

        # --- Helper: reflection boundary handling ---
        def reflect(x):
            # Reflect each dimension into [lb, ub] using periodic reflection.
            # Works for arbitrary step sizes.
            w = ub - lb
            # Map to [0, w), reflect about endpoints.
            # For numerical stability, ensure w>0 already checked.
            y = (x - lb) % (2.0 * w)
            # y in [0,2w). If y>w reflect: y' = 2w - y
            y = np.where(y > w, 2.0 * w - y, y)
            return lb + y

        # --- Initialize population ---
        # Choose population sizes heuristically but robust to small budgets/dim.
        # Ensure at least 2 individuals if possible.
        pop_max = min(max(4, 2 + self.dim // 2), 40)
        # Use a conservative (μ, λ)-style scheme.
        lam = min(pop_max, max(2, max_evals // 3))  # offspring per iteration target
        mu = min(max(2, lam // 2), lam)              # selected parents size

        # If budget is tiny, fall back to random search.
        if max_evals <= 4:
            best_x = None
            best_y = float("inf")
            for _ in range(max_evals):
                x = lb + (ub - lb) * np.random.random(self.dim)
                y = eval_one(x)
                if y < best_y:
                    best_y = y
                    best_x = x
            return best_x, best_y

        # Initial sigma: fraction of range.
        sigma0 = 0.25 * np.mean(ub - lb)
        # Avoid sigma0 too small.
        sigma0 = max(sigma0, 1e-12)

        # Start with mu parents + one extra for best estimate if budget allows.
        parents_n = mu
        # If budget allows additional initial sampling beyond parents, do a few.
        # Total initial evals: parents_n (parents) and possibly a couple extra.
        initial_extra = 0
        # Keep a cushion for at least one iteration of offspring.
        if max_evals - parents_n >= lam:
            # allow a small extra budget for better initial coverage
            initial_extra = min(2, max(0, max_evals - parents_n - lam))
        parents_n = min(parents_n, max_evals)
        if parents_n <= 0:
            parents_n = 1

        parents = lb + (ub - lb) * np.random.random((parents_n, self.dim))
        sigmas = np.full(parents_n, sigma0, dtype=float)

        # Evaluate parents
        parent_vals = np.empty(parents_n, dtype=float)
        for i in range(parents_n):
            parent_vals[i] = eval_one(parents[i])

        # Global best
        best_idx = int(np.argmin(parent_vals))
        best_x = parents[best_idx].copy()
        best_y = float(parent_vals[best_idx])

        if initial_extra > 0 and evals_used + initial_extra <= max_evals:
            extras = lb + (ub - lb) * np.random.random((initial_extra, self.dim))
            extra_vals = np.empty(initial_extra, dtype=float)
            for i in range(initial_extra):
                extra_vals[i] = eval_one(extras[i])
                if extra_vals[i] < best_y:
                    best_y = float(extra_vals[i])
                    best_x = extras[i].copy()
            # Incorporate extras into parent pool if we still have room.
            # Keep total parent pool size bounded by mu.
            all_parents = np.vstack([parents, extras])
            all_vals = np.concatenate([parent_vals, extra_vals])
            if all_parents.shape[0] > mu:
                sel = np.argsort(all_vals)[:mu]
                parents = all_parents[sel]
                parent_vals = all_vals[sel]
                sigmas = np.full(len(sel), sigma0, dtype=float)
            else:
                parents = all_parents
                parent_vals = all_vals
                sigmas = np.full(len(all_vals), sigma0, dtype=float)

        # --- Evolution parameters ---
        # Self-adaptation (log-normal), typical values:
        tau = 1.0 / np.sqrt(2.0 * self.dim) if self.dim > 0 else 1.0
        tau2 = 1.0 / np.sqrt(2.0 * np.sqrt(self.dim)) if self.dim > 0 else 1.0

        # Stagnation handling
        no_improve_iters = 0
        best_y_prev = best_y

        # Determine max iterations by budget consumption.
        # We'll stop when insufficient remaining evals for another offspring batch.
        while evals_used < max_evals:
            remaining = max_evals - evals_used
            if remaining <= 0:
                break

            # Offspring count for this iteration: can't exceed remaining budget.
            cur_lam = min(lam, remaining)

            # Sample parent indices for offspring
            # Prefer better parents more often by using fitness-proportional via ranks.
            # Rank-based weights avoid numerical issues.
            order = np.argsort(parent_vals)  # ascending
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(order))  # 0 is best
            # Convert ranks to weights: best has highest weight.
            # Add small floor to keep exploration.
            w = (len(parent_vals) - ranks).astype(float) + 1.0
            w /= w.sum()
            parent_choices = np.random.choice(len(parent_vals), size=cur_lam, p=w)

            offspring = np.empty((cur_lam, self.dim), dtype=float)
            off_sigmas = np.empty(cur_lam, dtype=float)

            # Generate offspring
            # For each offspring: mutate one selected parent with its self-adaptive sigma.
            # Mutation: x' = reflect(x + sigma * N(0,I))
            for k in range(cur_lam):
                p = parent_choices[k]
                # Self-adapt sigma
                gi = np.random.randn()  # N(0,1)
                gj = np.random.randn()  # N(0,1)
                sig = sigmas[p] * np.exp(tau * gi + tau2 * gj)

                # Occasional diversification if stagnating: boost sigma slightly.
                if no_improve_iters >= 6 and np.random.rand() < 0.15:
                    sig *= np.exp(0.5 * np.random.rand())

                # Ensure sigma not degenerate
                sig = float(max(sig, 1e-15))
                off_sigmas[k] = sig

                step = sig * np.random.randn(self.dim)
                x_new = parents[p] + step
                offspring[k] = reflect(x_new)

            # Evaluate offspring (only cur_lam evaluations)
            off_vals = np.empty(cur_lam, dtype=float)
            for k in range(cur_lam):
                off_vals[k] = eval_one(offspring[k])
                if off_vals[k] < best_y:
                    best_y = float(off_vals[k])
                    best_x = offspring[k].copy()

            # Selection and replacement: choose best mu from parents+offspring
            combined_x = np.vstack([parents, offspring])
            combined_vals = np.concatenate([parent_vals, off_vals])

            # Combine sigmas similarly: keep parent's sigmas for parents, offspring sigmas for offspring.
            combined_sigmas = np.concatenate([sigmas, off_sigmas])

            sel = np.argsort(combined_vals)[:min(mu, combined_vals.shape[0])]
            parents = combined_x[sel]
            parent_vals = combined_vals[sel]
            sigmas = combined_sigmas[sel]

            # Track improvement / stagnation
            if best_y < best_y_prev - 1e-12 * (abs(best_y_prev) + 1.0):
                no_improve_iters = 0
                best_y_prev = best_y
            else:
                no_improve_iters += 1

            # If remaining evaluations are too low, loop will exit naturally.

        return best_x, best_y
