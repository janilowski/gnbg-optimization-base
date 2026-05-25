# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, robust black-box minimization algorithm
# for continuous domains using a population-based evolutionary strategy. It maintains
# a set of candidate solutions and repeatedly generates new candidates by sampling
# around the current best solution and a small set of elites. Each evaluation consumes
# budget units; the algorithm never exceeds the provided evaluation budget.
# Search state: A population of points is maintained along with their objective values.
# The algorithm tracks how many function evaluations have been used, the current best
# solution, and an annealed step-size that controls how far new samples deviate.
# Candidate generation: New candidates are created by combining two sources of
# variation: (1) Gaussian sampling around the best/elite points with shrinking scale,
# and (2) occasional larger jumps using random directions derived from the population.
# Selection and replacement: After each batch of offspring is evaluated, candidates are
# merged with the population and the next generation is formed by keeping the best
# (lowest objective) individuals.
# Adaptation: The mutation step-size decays linearly with remaining budget, and
# additionally it is adjusted based on recent improvement (a simple heuristic).
# Exploration mechanisms: Early iterations use larger variance and allow stronger
# random exploration. Occasionally, offspring are sampled from perturbed elite
# points using random direction vectors from the current population.
# Exploitation mechanisms: Most offspring are sampled near the current best solution
# (and elites) to refine promising regions.
# Boundary handling: Candidate points are clipped to the provided bounds after
# sampling. If bounds are infinite, clipping is skipped for those dimensions.
# Budget strategy: The number of evaluations is controlled exactly. The algorithm
# allocates an initial evaluation budget for an initial population and then evaluates
# offspring until budget is exhausted.
# Closest known influences: The implementation is loosely inspired by evolutionary
# strategies (ES) with elitism and annealed mutation step-size, adapted for a strict
# evaluation budget and bounded domains.
# Novelty or unusual aspects: Uses a small elite-guided recombination of Gaussian
# samples and direction vectors, plus a simple step-size improvement heuristic to
# improve robustness under limited budgets.
# Failure modes: If the objective is extremely noisy or has deceptive landscapes,
# the simplistic adaptation may stagnate. With very small budgets, the algorithm relies
# mostly on initial sampling and local refinement around the best found point.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from func ----
        # Preferred: func.lower/func.upper; fallback: func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
        else:
            b = getattr(func, "bounds")
            lb = np.asarray(b.lb, dtype=float).reshape(-1)
            ub = np.asarray(b.ub, dtype=float).reshape(-1)

        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Handle infinite bounds: clipping only where finite.
        finite_lb = np.isfinite(lb)
        finite_ub = np.isfinite(ub)
        finite_mask = finite_lb & finite_ub

        def clip_to_bounds(x):
            # x: (dim,) or (n, dim)
            if np.all(~finite_mask):
                return x  # no finite box to clip
            if x.ndim == 1:
                y = x.copy()
                if np.any(finite_lb):
                    y[finite_lb] = np.maximum(y[finite_lb], lb[finite_lb])
                if np.any(finite_ub):
                    y[finite_ub] = np.minimum(y[finite_ub], ub[finite_ub])
                return y
            else:
                y = x.copy()
                if np.any(finite_lb):
                    y[:, finite_lb] = np.maximum(y[:, finite_lb], lb[finite_lb])
                if np.any(finite_ub):
                    y[:, finite_ub] = np.minimum(y[:, finite_ub], ub[finite_ub])
                return y

        # Determine an initial scale from the box size; fall back to 1.0.
        box = ub - lb
        box_finite = np.where(np.isfinite(box), box, 0.0)
        box_scale = np.max(np.abs(box_finite))
        if not np.isfinite(box_scale) or box_scale <= 0:
            box_scale = 1.0

        # ---- Evaluation bookkeeping ----
        evals_used = 0

        def eval_one(x):
            nonlocal evals_used
            if evals_used >= budget:
                # Hard stop safeguard; should not happen if we manage counts properly.
                return np.inf
            evals_used += 1
            return float(func(x))

        def eval_batch(X):
            # X: (n, dim)
            ys = np.empty(X.shape[0], dtype=float)
            for i in range(X.shape[0]):
                ys[i] = eval_one(X[i])
            return ys

        # ---- Population/Evolution strategy hyperparameters ----
        # Choose population size based on budget and dim, but keep it modest.
        # Ensure at least 2 individuals if budget permits.
        pop_size = min(16, max(2, budget)) if budget >= 2 else 1
        # Offspring per generation: small so we can respect strict budget.
        offspring_per_gen = min(16, max(1, pop_size))
        elite_k = max(1, min(4, pop_size // 2))

        # If budget is extremely small, just sample uniformly and return best.
        # (We still clip to bounds.)
        if budget <= pop_size:
            n = budget
            X = np.empty((n, dim), dtype=float)
            for i in range(n):
                if np.any(finite_mask):
                    # Uniform within bounds where possible; for infinite, use standard normal.
                    y = np.zeros(dim, dtype=float)
                    # finite where both bounds finite -> uniform in [lb, ub]
                    if np.any(finite_mask):
                        y[finite_mask] = lb[finite_mask] + np.random.rand(np.sum(finite_mask)) * (ub[finite_mask] - lb[finite_mask])
                    # for only one-sided/infinite, sample from a reasonable distribution around best guess
                    # (use 0.0 as anchor; clipping will handle finite sides).
                    # For simplicity: fill non-finite with N(0,1).
                    nf = ~finite_mask
                    if np.any(nf):
                        y[nf] = np.random.randn(np.sum(nf))
                    X[i] = clip_to_bounds(y)
                else:
                    X[i] = np.random.randn(dim)
            y = eval_batch(X)
            best_idx = int(np.argmin(y))
            return X[best_idx].copy(), float(y[best_idx])

        # ---- Initial population ----
        # Initialize around the box center with random offsets.
        center = np.where(finite_mask, 0.5 * (lb + ub), 0.0)
        init_sigma = 0.25 * box_scale

        X = np.empty((pop_size, dim), dtype=float)
        for i in range(pop_size):
            noise = np.random.randn(dim)
            # If bounds are finite, keep within a plausible range by using sigma from box scale.
            x = center + init_sigma * noise
            # Additionally, mix in uniform samples in the finite box early to improve coverage.
            if np.any(finite_mask):
                if np.random.rand() < 0.5:
                    u = np.random.rand(dim)
                    xx = center.copy()
                    xx[finite_mask] = lb[finite_mask] + u[finite_mask] * (ub[finite_mask] - lb[finite_mask])
                    # For infinite coords, keep gaussian
                    nf = ~finite_mask
                    xx[nf] = center[nf] + init_sigma * np.random.randn(np.sum(nf)) if np.any(nf) else xx[nf]
                    x = 0.5 * x + 0.5 * xx
            X[i] = clip_to_bounds(x)

        Y = eval_batch(X)
        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])

        # Step-size annealing and improvement tracking
        sigma = init_sigma
        prev_best_y = best_y
        no_improve_counter = 0

        # Remaining budget
        while evals_used < budget:
            remaining = budget - evals_used
            n_off = min(offspring_per_gen, remaining)

            # Sort population; keep elites
            order = np.argsort(Y)
            elites = X[order[:elite_k]]
            elites_y = Y[order[:elite_k]]
            # Update best known if needed (in case of numerical oddities)
            if float(elites_y[0]) < best_y:
                best_y = float(elites_y[0])
                best_x = elites[0].copy()

            # Linear annealing factor: shrink sigma with remaining budget.
            # Map remaining to progress [0,1]
            progress = 1.0 - (remaining / max(1, budget))
            # sigma decreases as progress increases; keep a floor to avoid collapse.
            sigma_target = init_sigma * max(0.08, 1.0 - progress)
            sigma = 0.7 * sigma + 0.3 * sigma_target

            # Improvement heuristic: if recent best didn't improve, slightly decrease sigma less (encourage exploration).
            # We'll use no_improve_counter to adjust mutation.
            if best_y < prev_best_y - 1e-12:
                prev_best_y = best_y
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            if no_improve_counter >= 3:
                # encourage exploration a bit
                sigma *= 1.15
                no_improve_counter = 0

            # Direction vectors from population differences (for occasional larger moves)
            if pop_size >= 2:
                i1 = np.random.randint(0, pop_size)
                i2 = np.random.randint(0, pop_size)
                direction = X[i1] - X[i2]
                d_norm = np.linalg.norm(direction)
                if d_norm > 0:
                    direction = direction / d_norm
                else:
                    direction = np.random.randn(dim)
                    dn = np.linalg.norm(direction)
                    direction = direction / dn if dn > 0 else direction
            else:
                direction = np.random.randn(dim)
                dn = np.linalg.norm(direction)
                direction = direction / dn if dn > 0 else direction

            # ---- Generate offspring ----
            # Most offspring: Gaussian around best + elites.
            # Some offspring: larger steps along random direction plus Gaussian.
            X_off = np.empty((n_off, dim), dtype=float)
            for i in range(n_off):
                if np.random.rand() < 0.75:
                    # Exploit: sample around a randomly chosen elite, centered with drift toward current best.
                    e = elites[np.random.randint(0, elite_k)]
                    # Blend elite and best to bias toward the best region.
                    anchor = 0.6 * e + 0.4 * best_x
                    step = sigma * np.random.randn(dim)
                    # Add a small correlated component using direction to promote escapes.
                    if dim > 1 and np.random.rand() < 0.2:
                        step += 0.25 * sigma * direction * np.random.randn()
                    x = anchor + step
                else:
                    # Explore: take a bigger jump along a direction, plus Gaussian noise.
                    scale = sigma * (1.5 + 2.5 * np.random.rand())
                    x = best_x + scale * direction + sigma * 0.5 * np.random.randn(dim)
                X_off[i] = clip_to_bounds(x)

            Y_off = eval_batch(X_off)

            # ---- Selection and replacement (elitist) ----
            # Merge and keep best pop_size.
            X = np.vstack([X, X_off])
            Y = np.concatenate([Y, Y_off])

            if X.shape[0] > pop_size + 2 * offspring_per_gen:
                # Rarely reduce early to control memory; selection will happen anyway.
                order = np.argsort(Y)[:pop_size]
                X = X[order]
                Y = Y[order]

            order = np.argsort(Y)
            X = X[order[:pop_size]]
            Y = Y[order[:pop_size]]

            if Y[0] < best_y:
                best_y = float(Y[0])
                best_x = X[0].copy()

        return best_x, best_y
