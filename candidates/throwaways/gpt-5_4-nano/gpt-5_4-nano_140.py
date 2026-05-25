import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, budget-aware black-box minimization algorithm
# (derivative-free) using a global-to-local strategy: it samples a batch of points uniformly
# within the provided bounds, then repeatedly refines a Gaussian "trust region" around the
# best-so-far point while shrinking step sizes and occasionally restarting exploration to
# avoid stagnation.
# Search state: The algorithm keeps track of the current best point and value (best_x,
# best_y), the number of evaluations used, and a step-size scale for generating candidate
# points.
# Candidate generation: Candidates are generated either (1) by uniform random sampling
# across the whole domain (initial and restart phases), or (2) by sampling from a Gaussian
# around best_x with a tunable per-dimension step size. The Gaussian points are clipped
# to the bounds.
# Selection and replacement: After evaluating each candidate, the best-so-far solution is
# updated whenever a candidate has a lower objective value. The "local center" stays at
# best_x.
# Adaptation: The step size shrinks when improvements are found and grows slightly when
# they are not (using a simple multiplicative schedule with caps). This balances exploration
# and exploitation.
# Exploration mechanisms: Random restarts are triggered when there is no improvement for a
# while. During restarts, new points are sampled uniformly in the feasible box.
# Exploitation mechanisms: Between restarts, candidates are drawn from a shrinking Gaussian
# distribution around best_x to concentrate search near promising regions.
# Boundary handling: Candidates are always clipped to [lb, ub] to ensure feasibility.
# Budget strategy: The algorithm strictly respects the evaluation budget by computing an
# allocation for the initial sampling batch and then looping while remaining budget permits.
# Closest known influences: The design resembles common simple evolution strategies /
# (1+λ)-style trust-region heuristics adapted to a pure black-box setting.
# Novelty or unusual aspects: The algorithm combines a small initial global Latin-ish random
# batch (via shuffled stratified bins) with a lightweight adaptive trust region and a
# stagnation-based restart policy, all within a single self-contained module.
# Failure modes: If the budget is extremely small (e.g., budget <= 1), it may return only a
# single evaluated point. On highly multimodal or deceptive functions, the algorithm can
# stagnate; restarts and step adaptation mitigate but cannot guarantee success.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if dim <= 0:
            raise ValueError("dim must be positive")

        # Read bounds from func.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("func must provide bounds via (lower, upper) or func.bounds.lb/ub")

        if lb.shape != (dim,) or ub.shape != (dim,):
            lb = np.reshape(lb, (dim,))
            ub = np.reshape(ub, (dim,))
        if np.any(ub <= lb):
            raise ValueError("Invalid bounds: require ub > lb for all dimensions.")

        lb = lb.astype(float)
        ub = ub.astype(float)
        span = ub - lb

        evals = 0
        best_x = None
        best_y = None

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_point(x):
            nonlocal evals, best_x, best_y
            y = float(func(x))
            evals += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Handle tiny budgets: evaluate one point and return.
        if budget <= 0:
            # No evaluations allowed; still need a return signature.
            # We choose the midpoint without evaluating (best_y unknown).
            mid = clip((lb + ub) / 2.0)
            return mid, float("inf")
        if budget == 1:
            x0 = clip((lb + ub) / 2.0)
            y0 = eval_point(x0)
            return best_x, y0

        rng = np.random.default_rng()

        # --- Initial global exploration ---
        # Use a small stratified/reshuffled initial batch to cover the space a bit better
        # than pure uniform sampling, while staying simple.
        # Batch size chosen to leave room for local refinement.
        init_frac = 0.25
        init_batch = max(2, min(budget // 4, int(np.ceil(init_frac * budget))))
        init_batch = min(init_batch, budget)

        # Create stratified samples in [0,1]^d (each dimension independently binned and permuted).
        # If budget is small, fewer points are used.
        t = np.linspace(0.0, 1.0, init_batch, endpoint=False) + 0.5 / init_batch
        # For each dimension, randomly permute bins and add a small jitter.
        U = np.empty((init_batch, dim), dtype=float)
        for j in range(dim):
            perm = rng.permutation(init_batch)
            # jitter within bin width
            bin_w = 1.0 / init_batch
            jitter = (rng.random(init_batch) - 0.5) * bin_w * 0.9
            U[:, j] = t[perm] + jitter
        X = lb + U * span
        X = clip(X)

        # Evaluate initial batch
        for i in range(init_batch):
            if evals >= budget:
                break
            eval_point(X[i])

        # --- Local adaptive trust-region search with restarts ---
        # Initial step: a fraction of domain size.
        # Using per-dimension scaling makes it robust to different axis scales.
        step = 0.25 * span
        min_step = 1e-12 * span
        max_step = span

        # Stagnation/restart policy
        no_improve = 0
        patience = max(10, min(50, 2 * dim + 10))

        # Choose a candidate batch size to reduce overhead while respecting budget.
        # The harness is deterministic via numpy seeding; avoid additional nondeterminism.
        # We'll use rng from default_rng which is independent of np.random.seed; that would break
        # reproducibility. Instead, use np.random (legacy) which respects np.random.seed.
        # To keep compact, we will switch: use np.random for sampling.
        # (Re-seed effect is handled by the harness before each run.)
        rng = None  # signals to use np.random below

        def rand_unit(shape):
            return np.random.random(shape)

        # Main loop
        while evals < budget:
            remaining = budget - evals
            # λ candidates per iteration, bounded to remaining budget
            lam = min(20, remaining)
            # scale of Gaussian noise relative to step
            # Small additional factor to promote variation.
            gauss = step * (0.8 + 0.4 * rand_unit((lam, dim)))

            # Sample from N(best_x, gauss^2) and clip
            Z = rand_unit((lam, dim)) * 2.0 - 1.0  # in [-1,1]
            # Convert to roughly Gaussian-like by summing uniforms (CLT)
            # This avoids np.random.normal and stays simple/deterministic.
            g = (Z + rand_unit((lam, dim)) * 2.0 - 1.0 + rand_unit((lam, dim)) * 2.0 - 1.0) / 3.0
            Xc = best_x + g * gauss
            Xc = clip(Xc)

            improved = False
            # Evaluate candidates
            for i in range(lam):
                if evals >= budget:
                    break
                prev_best = best_y
                eval_point(Xc[i])
                if best_y < prev_best - 0.0:
                    improved = True

            if improved:
                no_improve = 0
                # Shrink step to focus exploitation after improvement.
                step = np.maximum(min_step, step * (0.6 + 0.15 * rand_unit(dim)))
            else:
                no_improve += lam
                # Slightly expand or keep step to escape stagnation; with restart it will reset.
                step = np.minimum(max_step, step * (1.05 + 0.10 * rand_unit(dim)))

                if no_improve >= patience and evals < budget:
                    # Restart: sample new points uniformly across the domain.
                    # Use at least a couple points but not exceed remaining budget.
                    r_batch = min(max(4, 2 * dim), budget - evals)
                    U2 = rand_unit((r_batch, dim))
                    Xr = lb + U2 * span
                    Xr = clip(Xr)
                    for i in range(r_batch):
                        if evals >= budget:
                            break
                        eval_point(Xr[i])
                    # Reset step around the (possibly updated) best.
                    step = 0.3 * span
                    no_improve = 0

        return best_x, best_y
