# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer using a
# restartable randomized search with an adaptive Gaussian sampling strategy.
# The algorithm maintains a current best solution and repeatedly generates
# candidate points around it, using an expanding/contracting step size based
# on whether improvements are found, while also preserving a small fraction of
# purely exploratory samples.
# Search state: Keeps track of (1) best_x/best_y, (2) current step_size,
# (3) recent_improvement counter, and (4) remaining evaluation budget.
# Candidate generation: For each iteration, draws several candidate points:
# most are sampled as best_x + step_size * N(0, I), with one additional
# "directional" candidate based on a random unit vector and a small multi-step
# offset; a few are uniform random samples across the domain for exploration.
# Selection and replacement: Evaluates each candidate; any improvement replaces
# best_x/best_y. The number of evaluations per iteration is accounted for so the
# total objective calls never exceed the provided budget.
# Adaptation: Uses a step_size update rule: if enough improvements are observed
# in an iteration, step_size shrinks (focus/exploit); otherwise it expands
# (broaden/explore). A restart is triggered if the algorithm fails to
# improve for many consecutive iterations by resetting best to the best among
# a fresh batch and reinitializing step_size.
# Exploration mechanisms: Uniform random samples across bounds and occasional
# directional proposals.
# Exploitation mechanisms: Gaussian sampling centered at best_x and optional
# local refinement via directional offsets.
# Boundary handling: All candidates are clipped to [lb, ub] (box constraints).
# Budget strategy: Uses an internal evaluation counter; each objective call
# increments it and stops generating new candidates when the budget is exhausted.
# Closest known influences: A lightweight mixture of random search,
# (1+λ)-style selection, and step-size control reminiscent of evolution strategies,
# but implemented in a compact, standard-library-only way.
# Novelty or unusual aspects: Includes both Gaussian and directional candidates
# plus a simple restart trigger based on consecutive lack of improvements.
# Failure modes: In very high dimensions or highly irregular objectives, the
# algorithm may require many evaluations; clipping can reduce effective diversity
# near bounds; restarts may not help if improvements are extremely rare.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._read_bounds(func)
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)

        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Ensure lb <= ub; if equal, the dimension is fixed.
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        lb, ub = lb2, ub2

        # Robust "center" for initialization: mid-point, clipped to bounds.
        x_center = 0.5 * (lb + ub)

        evals_used = 0
        best_x = None
        best_y = None

        def eval_point(x):
            nonlocal evals_used, best_x, best_y
            if evals_used >= self.budget:
                return None, True
            y = float(func(x))
            evals_used += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = x.copy()
            return y, False

        # Initial step size based on scale of the domain.
        domain = ub - lb
        # Avoid zero scales.
        scale = np.where(domain > 0, domain, 1.0)
        base_step = 0.25 * np.mean(scale)
        if not np.isfinite(base_step) or base_step <= 0:
            base_step = 1.0

        # Budget allocation per iteration: more candidates early, fewer later.
        # Keep small to remain efficient.
        def batch_size_for_progress(progress):
            # progress in [0,1], bigger progress -> smaller batch
            # Ensure at least 1 candidate per iteration.
            size = int(max(1, round(6 - 4 * progress)))
            return size

        # Exploration ratio decreases as we progress.
        def exploration_ratio(progress):
            # starts ~0.35, ends ~0.1
            return 0.35 - 0.25 * progress

        # Restart policy
        max_iters = max(1, self.budget)  # upper bound for loop iterations
        consecutive_no_improve = 0
        restart_patience = 6  # in iterations

        # Start with a small random sampling around center to get a reasonable best.
        initial_batch = min(self.budget, 1 + min(10, self.dim))
        # Include center as a candidate.
        y, stopped = eval_point(x_center)
        if stopped:
            return best_x, best_y

        for _ in range(initial_batch - 1):
            if evals_used >= self.budget:
                break
            x = x_center + base_step * np.random.randn(self.dim)
            x = np.clip(x, lb, ub)
            eval_point(x)

        step_size = base_step
        it = 0
        prev_best_y = best_y

        while evals_used < self.budget:
            it += 1
            progress = evals_used / max(1, self.budget)

            # Determine how many new points to sample this iteration.
            remaining = self.budget - evals_used
            batch = batch_size_for_progress(progress)
            batch = min(batch, remaining)

            # Decide exploration vs exploitation.
            exp_ratio = exploration_ratio(progress)
            n_explore = int(max(0, round(exp_ratio * batch)))
            n_exploit = batch - n_explore

            improved_this_iter = False
            best_y_before = best_y

            # Exploitation: Gaussian around current best.
            # Use isotropic Gaussian; step size adapts.
            for _ in range(n_exploit):
                if evals_used >= self.budget:
                    break
                x = best_x + step_size * np.random.randn(self.dim)
                x = np.clip(x, lb, ub)
                eval_point(x)

            # Add a directional candidate (local refinement) for diversity.
            if evals_used < self.budget:
                if n_exploit < batch and (batch - n_explore - 1) >= 0:
                    # already consumed most exploitation slots; still add direction only sometimes
                    if np.random.rand() < 0.7:
                        u = np.random.randn(self.dim)
                        nu = np.linalg.norm(u)
                        if nu == 0:
                            u = np.ones(self.dim) / np.sqrt(self.dim)
                        else:
                            u = u / nu
                        # Try a couple of offsets.
                        # Use a small step and a larger step fraction.
                        for alpha in (0.8, -0.6):
                            if evals_used >= self.budget:
                                break
                            x = best_x + (alpha * step_size) * u
                            x = np.clip(x, lb, ub)
                            eval_point(x)
                else:
                    # Not enough remaining exploitation slots; add direction with probability.
                    if np.random.rand() < 0.5:
                        u = np.random.randn(self.dim)
                        nu = np.linalg.norm(u)
                        if nu == 0:
                            u = np.ones(self.dim) / np.sqrt(self.dim)
                        else:
                            u = u / nu
                        alpha = (0.5 + 0.5 * np.random.rand()) * step_size
                        x = best_x + alpha * u * (1 if np.random.rand() < 0.5 else -1)
                        x = np.clip(x, lb, ub)
                        eval_point(x)

            # Exploration: uniform random samples across the bounds.
            for _ in range(n_explore):
                if evals_used >= self.budget:
                    break
                x = lb + (ub - lb) * np.random.rand(self.dim)
                eval_point(x)

            # Adapt step size based on whether we improved.
            if best_y is not None and best_y < best_y_before:
                improved_this_iter = True

            if improved_this_iter:
                consecutive_no_improve = 0
                # Contract a bit to exploit.
                step_size *= 0.82
            else:
                consecutive_no_improve += 1
                # Expand to search wider.
                step_size *= 1.12

            # Keep step_size within reasonable limits.
            mean_span = float(np.mean(scale))
            min_step = 1e-12 * mean_span if mean_span > 0 else 1e-12
            max_step = 0.75 * mean_span if mean_span > 0 else 1.0
            step_size = float(np.clip(step_size, min_step, max_step))

            # Restart if stuck.
            if consecutive_no_improve >= restart_patience and evals_used < self.budget:
                consecutive_no_improve = 0
                # Sample a new "best" from a fresh batch to escape local traps.
                # The restart uses a portion of remaining budget but stays within it.
                restart_batch = min(10 + self.dim // 2, self.budget - evals_used)
                candidate_best_x = best_x
                candidate_best_y = best_y
                for _ in range(restart_batch):
                    x = lb + (ub - lb) * np.random.rand(self.dim)
                    y = float(func(x))
                    evals_used += 1
                    if candidate_best_y is None or y < candidate_best_y:
                        candidate_best_y = y
                        candidate_best_x = x.copy()
                    if evals_used >= self.budget:
                        break
                if candidate_best_y is not None and (best_y is None or candidate_best_y < best_y):
                    best_x = candidate_best_x
                    best_y = candidate_best_y
                # Reinitialize step around domain scale to promote exploration.
                step_size = base_step

            # If no progress but budget is nearly exhausted, break soon.
            if evals_used >= self.budget:
                break

        # Ensure best_x is set even if budget was 0 (shouldn't happen per harness, but safe).
        if best_x is None:
            best_x = np.clip(x_center, lb, ub)
            best_y = float(func(best_x)) if self.budget > 0 else float("inf")

        return best_x, best_y

    @staticmethod
    def _read_bounds(func):
        # Prefer func.lower/func.upper; otherwise func.bounds.lb/func.bounds.ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return func.lower, func.upper
        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return b.lb, b.ub
        raise AttributeError(
            "Objective function must expose bounds via func.lower/func.upper "
            "or func.bounds.lb/func.bounds.ub."
        )
