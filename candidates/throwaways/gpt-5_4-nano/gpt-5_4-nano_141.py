# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm that
# works in any dimension with only function evaluations. It maintains a small
# population of candidate points and repeatedly improves them by sampling around
# promising regions. It is designed to be robust for minimization and respects
# the evaluation budget exactly.
# Search state: The algorithm tracks a population of candidate vectors, their
# objective values, the current global best (best_x, best_y), a per-run step
# size (sigma), and an evaluation counter to ensure we never exceed the budget.
# Candidate generation: Each iteration generates new candidates by adding
# Gaussian perturbations to either the current best point or randomly chosen
# elite points from the population. Perturbation scale is controlled by sigma.
# Selection and replacement: New candidates are evaluated, then accepted if they
# improve the population member they replace (elitist/greedy replacement). The
# global best is updated whenever a better value is found.
# Adaptation: sigma is adapted based on progress: if improvements occur, sigma
# gradually shrinks (focus); if no progress is observed, sigma grows (escape).
# Exploration mechanisms: Random sampling around elites plus occasional larger
# jumps controlled by a decaying probability provide exploration early on.
# Exploitation mechanisms: Most samples are centered at the current best and
# small perturbations are used when improvements are frequent.
# Boundary handling: Candidates are clipped to the provided bounds (read from
# func.lower/func.upper or func.bounds.lb/ub) to stay feasible.
# Budget strategy: Evaluations are counted explicitly; the loop terminates when
# remaining budget is insufficient. Initial population evaluation plus iterative
# updates never exceed the provided evaluation budget.
# Closest known influences: The approach is a simplified, budget-safe variant of
# evolution strategies / CMA-like “ask-tell” behavior using isotropic Gaussian
# mutations and elitist selection, with a lightweight sigma adaptation.
# Novelty or unusual aspects: Uses a small, greedy replacement population and a
# progress-based sigma schedule with both best-centered exploitation and elite-
# centered exploration; designed to be readable and compact.
# Failure modes: If the objective is very noisy or highly deceptive, greedy
# replacement may stall; sigma growth helps, but extremely adversarial noise
# may still limit performance. If bounds are extremely tight, clipping can
# reduce effective search movement.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _read_bounds(self, func):
        # Try func.lower/func.upper first
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        # Fallback to func.bounds.lb / func.bounds.ub
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError(
                "Objective function must provide bounds via func.lower/func.upper "
                "or func.bounds.lb/func.bounds.ub."
            )

        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            # Allow scalar-like bounds if dimension matches by broadcasting
            if lb.size == 1:
                lb = np.full((self.dim,), float(lb))
            if ub.size == 1:
                ub = np.full((self.dim,), float(ub))

        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            raise ValueError(f"Bounds must be vectors of shape ({self.dim},). Got {lb.shape}, {ub.shape}.")

        if np.any(ub < lb):
            raise ValueError("Upper bounds must be >= lower bounds elementwise.")
        return lb, ub

    def __call__(self, func):
        lb, ub = self._read_bounds(func)
        dim = self.dim

        # Evaluate with strict budget accounting.
        evals = 0

        def f(x):
            nonlocal evals
            if evals >= self.budget:
                # Should never happen if loops are budget-safe.
                return np.inf
            y = func(np.asarray(x, dtype=float))
            evals += 1
            return float(y)

        # Budget can be very small; handle gracefully.
        if self.budget <= 0:
            # No evaluations allowed; return a feasible point with +inf objective.
            mid = (lb + ub) / 2.0
            return mid, float("inf")

        # Population size: keep small for compactness, but ensure at least 1.
        # Also ensure we don't evaluate more than budget initially.
        pop_size = min(8 + dim // 2, self.budget)
        pop_size = max(1, pop_size)

        rng = np.random

        # Initialize population uniformly in bounds.
        # If bounds are degenerate, all points collapse to same location.
        width = ub - lb
        # Avoid sigma being exactly 0 in degenerate dimensions.
        base_sigma = np.linalg.norm(width) / np.sqrt(dim) if dim > 0 else 1.0
        base_sigma = max(1e-12, base_sigma)

        pop = lb + width * rng.rand(pop_size, dim)
        vals = np.array([f(pop[i]) for i in range(pop_size)], dtype=float)

        best_idx = int(np.argmin(vals))
        best_x = pop[best_idx].copy()
        best_y = float(vals[best_idx])

        # Step size adaptation.
        sigma = base_sigma * 0.25  # start fairly exploratory
        no_improve_steps = 0

        # Number of "generations" is not fixed; we use budget as the stopping rule.
        # Each generation tries to replace a subset of population members.
        # Use a small batch size to remain budget-safe.
        while evals < self.budget:
            remaining = self.budget - evals
            # Determine how many new candidates to evaluate this round.
            # Replace up to pop_size members but do not exceed remaining.
            batch = min(pop_size, remaining)

            improved_this_round = False

            # Elite set: the best few points influence sampling.
            elite_k = min(3, pop_size)
            elite_idx = np.argsort(vals)[:elite_k]
            elites = pop[elite_idx]

            # Exploration probability decays with time.
            # Early on: more diversity; later: more exploitation.
            t = evals / max(1, self.budget)
            explore_p = 0.35 * (1.0 - t)  # from ~0.35 down to 0

            # Generate and evaluate replacements.
            for _ in range(batch):
                # Choose a target index to attempt replacement:
                # prefer non-best individuals to reduce wasted evaluations.
                if pop_size == 1:
                    target_i = 0
                else:
                    # Choose among indices excluding current best if possible
                    if best_idx < pop_size:
                        candidates = [i for i in range(pop_size) if i != best_idx]
                        target_i = candidates[rng.randint(len(candidates))] if candidates else rng.randint(pop_size)
                    else:
                        target_i = rng.randint(pop_size)

                # Select center: mostly best for exploitation, sometimes elite/random for exploration.
                if rng.rand() < (1.0 - explore_p):
                    center = best_x
                else:
                    # Pick an elite at random
                    center = elites[rng.randint(elite_k)]

                # Mutation scale:
                # - small steps when improving
                # - occasionally larger jumps for escape.
                if rng.rand() < explore_p and no_improve_steps > 0:
                    jump = 2.0 + 4.0 * rng.rand()
                else:
                    jump = 1.0

                # Gaussian perturbation (isotropic).
                z = rng.randn(dim)
                x_new = center + (sigma * jump) * z

                # Boundary handling via clipping.
                if np.any(width == 0):
                    # For degenerate dimensions, clip still works but ensures exact feasibility.
                    pass
                x_new = np.minimum(ub, np.maximum(lb, x_new))

                y_new = f(x_new)

                # Greedy replacement: keep improvement against target member.
                if y_new < vals[target_i]:
                    pop[target_i] = x_new
                    vals[target_i] = y_new

                # Update global best if improved.
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new.copy()
                    improved_this_round = True
                    # Update best_idx quickly
                    best_idx = int(np.argmin(vals))

            # Adapt sigma based on progress.
            if improved_this_round:
                no_improve_steps = 0
                # Shrink to exploit around the best.
                sigma *= 0.85
            else:
                no_improve_steps += 1
                # Grow to explore when stalling.
                sigma *= 1.08

            # Keep sigma within reasonable limits based on bounds.
            # Lower bound prevents collapse; upper bound prevents wild wandering.
            # Upper bound tied to the average bound width.
            avg_width = float(np.mean(width)) if dim > 0 else 1.0
            avg_width = max(avg_width, 1e-12)
            sigma = float(np.clip(sigma, 1e-12, 2.5 * avg_width))

        return best_x, best_y
