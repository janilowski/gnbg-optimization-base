# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact population-based black-box minimizer (derivative-free)
# that mixes exploration (random directions and covariance-less mutations) with
# exploitation (Gaussian-perturbed steps around the current best). It is designed
# to be robust across dimensions while staying strictly within the given evaluation
# budget.
# Search state: Maintains a small population of candidate points, their objective
# values, and the current best solution found so far. Tracks remaining evaluation
# budget and stops early if exhausted.
# Candidate generation: Generates new candidates by mutating around the best
# and around randomly chosen elite individuals. Uses multiple mutation types:
# (1) isotropic Gaussian step from the best, (2) coordinate-wise jitter,
# (3) differential-style move using two population members, and (4) occasional
# uniform random re-seeding for diversity.
# Selection and replacement: Uses elitist selection: any candidate that improves the
# current best replaces it; population is refreshed by accepting candidates that
# improve over the worst in the population, otherwise retaining existing elites.
# Adaptation: Step sizes adapt based on progress: if improvement is observed,
# the mutation scale is decreased (more exploitation); if not, the scale is
# increased (more exploration). Mutation scale is also bounded to keep steps
# meaningful relative to the search domain.
# Exploration mechanisms: Random re-seeding, differential moves with random pairs,
# and larger mutation scales when stagnation is detected.
# Exploitation mechanisms: Gaussian perturbations centered at the best (and elite)
# with shrinking scale after improvements.
# Boundary handling: Clamps all candidates to the provided variable bounds
# after every mutation.
# Budget strategy: Converts the evaluation budget into an integer number of
# function calls. Uses careful counters and stops once the budget is reached.
# Closest known influences: Conceptually resembles ES/CMA-free strategies and
# blends PSO-like differential moves with adaptive Gaussian exploitation.
# Novelty or unusual aspects: Uses a covariance-less adaptive scale plus multiple
# mutation operators selected stochastically each iteration, all with strict
# budget awareness.
# Failure modes: For extremely noisy or deceptive objectives, the adaptive
# scale may oscillate; boundary clamping can also create plateaus if the optimum
# lies near edges. In very high dimensions with tiny budgets, performance is
# limited by the number of initial samples.
# ALGORITHM_ANALYSIS_NOTE_END

from typing import Tuple, Optional
import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func) -> Tuple[np.ndarray, float]:
        # ---- Bounds handling (supports lower/upper or bounds.lb/bounds.ub) ----
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub.")

        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            lb = lb.reshape(-1)
            ub = ub.reshape(-1)
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        # Handle degenerate bounds robustly
        span = ub - lb
        span = np.where(span > 0, span, 1.0)

        # ---- Budget-aware evaluation wrapper ----
        evals = 0
        budget = max(0, int(self.budget))

        def clamp(x: np.ndarray) -> np.ndarray:
            return np.minimum(ub, np.maximum(lb, x))

        def eval_one(x: np.ndarray) -> float:
            nonlocal evals
            if evals >= budget:
                # Should never happen if used carefully.
                return float("inf")
            evals += 1
            return float(func(np.asarray(x, dtype=float)))

        # Edge cases
        if budget == 0:
            x0 = clamp(lb + 0.5 * span * np.zeros(self.dim))
            return x0, float("inf")
        if self.dim <= 0:
            x0 = np.zeros(0, dtype=float)
            y0 = eval_one(x0)
            return x0, y0

        # ---- Population size selection (small, budget-friendly) ----
        # Try to get at least a handful of candidates, but not exceed budget.
        # Ensure at least 2 individuals for differential-style moves.
        pop_size = int(min(max(4, 2 * self.dim + 2), max(5, budget)))
        pop_size = max(2, min(pop_size, budget))

        # ---- Initialize population uniformly in bounds ----
        # Use NumPy RNG seeded by harness.
        X = lb + np.random.rand(pop_size, self.dim) * span
        Y = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            Y[i] = eval_one(X[i])

        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])

        # Mutation/exploration parameters (adapt over time)
        # Start with a fraction of domain size.
        base_sigma = 0.2 * np.mean(span)
        sigma = base_sigma if base_sigma > 0 else 0.1
        sigma_min = 1e-12
        sigma_max = 0.5 * np.mean(span) + 1e-12

        stagnation = 0
        # Determine iterations by remaining budget, leaving room for candidate evaluations.
        # Each loop proposes a small batch.
        # Proposed evaluations per iteration:
        batch = int(min(max(4, self.dim // 2 + 1), max(1, budget - evals)))
        batch = min(batch, max(1, budget - evals))

        # Elite/worst tracking
        def replace_if_better(i_candidate_x, y_candidate, accept_elite=False):
            nonlocal best_x, best_y, sigma, stagnation
            if y_candidate < best_y:
                best_y = y_candidate
                best_x = i_candidate_x.copy()
                # Improvement observed: exploitation slightly stronger
                stagnation = 0
                sigma = max(sigma_min, sigma * 0.85)
            else:
                stagnation += 1 if not accept_elite else 0

        # ---- Main loop ----
        # Each iteration:
        # - generate batch of candidates using multiple operators
        # - evaluate candidates (respecting budget)
        # - elitist replacement into population
        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break
            k = min(batch, remaining)

            # Choose elites to guide steps
            elite_count = max(2, int(min(pop_size, 2 + self.dim // 3)))
            elite_idx = np.argpartition(Y, elite_count - 1)[:elite_count]
            elite = X[elite_idx]

            # Worst index for population replacement
            worst_idx = int(np.argmax(Y))

            candidates = np.empty((k, self.dim), dtype=float)
            cand_ops = np.random.randint(0, 4, size=k)  # operator selection
            for t in range(k):
                op = int(cand_ops[t])

                if op == 0:
                    # Gaussian exploitation around current best
                    step = np.random.randn(self.dim)
                    # occasional larger step
                    scale = sigma * (2.2 if np.random.rand() < 0.15 else 1.0)
                    x = best_x + scale * step
                elif op == 1:
                    # Coordinate-wise jitter (keeps some axes close, others explore)
                    x = best_x.copy()
                    # mutate a random subset of coordinates
                    m = max(1, int(np.random.randint(1, self.dim + 1)))
                    idxs = np.random.choice(self.dim, size=m, replace=False)
                    # individual step sizes
                    for j in idxs:
                        x[j] += sigma * np.random.randn() * (0.5 + np.random.rand())
                elif op == 2:
                    # Differential-style move: elite_i + F*(elite_j - elite_k)
                    # avoids needing covariance information.
                    a, b, c = np.random.choice(elite.shape[0], size=3, replace=True)
                    x = elite[a] + (0.6 + 0.7 * np.random.rand()) * (elite[b] - elite[c])
                    # add small local noise
                    x += 0.25 * sigma * np.random.randn(self.dim)
                else:
                    # Random re-seeding for diversity, occasionally
                    x = lb + np.random.rand(self.dim) * span

                candidates[t] = clamp(x)

            # Evaluate candidates, update best and population
            for t in range(k):
                y = eval_one(candidates[t])
                # Update global best
                if y < best_y:
                    replace_if_better(candidates[t], y)
                # Elitist replacement into population
                if y < Y[worst_idx]:
                    X[worst_idx] = candidates[t]
                    Y[worst_idx] = y
                    worst_idx = int(np.argmax(Y))

            # Adaptation after each batch based on stagnation
            # If no improvement for a while, increase sigma to escape.
            if stagnation > 2 + self.dim // 4:
                sigma = min(sigma_max, sigma * 1.2)
                stagnation = max(0, stagnation - 3)

            # Update worst_idx for next iteration
            worst_idx = int(np.argmax(Y))

            # Recompute elite-based parameters lightly (optional, keep compact)
            # (sigma bounds already enforce stability)

            # If budget is tight, stop once near exhaustion
            if evals >= budget:
                break

        return best_x, best_y
