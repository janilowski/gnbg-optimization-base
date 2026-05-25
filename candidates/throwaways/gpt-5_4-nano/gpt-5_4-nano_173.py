# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization routine using a
# population-based evolution strategy with restart-like behavior. It is
# designed to work for any dimension given bounds and a function handle.
#
# Search state: Maintains a small population of candidate solutions and
# their objective values. Tracks the global best-so-far point/value.
#
# Candidate generation: Samples new candidates using isotropic Gaussian
# mutations around a recombination center (chosen from the best elites).
# Step size (sigma) adapts based on improvement.
#
# Selection and replacement: Uses (μ+λ)-style selection: combine parents and
# offspring, then keep the top μ by lowest objective (minimization).
#
# Adaptation: Uses a simple success-based adaptation: when the iteration
# yields improved best, sigma slightly decreases; otherwise sigma increases,
# promoting exploration when stuck and finer search when progressing.
#
# Exploration mechanisms: Gaussian mutations with adaptive sigma and occasional
# diversity injection by sampling around the best or random elites.
#
# Exploitation mechanisms: Recombination center is biased toward elites, so
# mutations are concentrated near good regions.
#
# Boundary handling: After mutation, candidates are clipped to the provided
# bounds (lb/ub) to respect constraints.
#
# Budget strategy: Carefully counts objective evaluations and never exceeds
# the provided budget. The algorithm uses iteration blocks, and the final
# iteration is truncated so the total evaluations remain within budget.
#
# Closest known influences: Similar in spirit to CMA-ES/ES heuristics but much
# simpler—an isotropic evolution strategy with elite recombination and
# success-based step-size control.
#
# Novelty or unusual aspects: Uses a dimension-aware initial sigma and a
# robust evaluation-budget truncation to guarantee the budget constraint.
#
# Failure modes: If the objective is extremely noisy or deceptive, step-size
# adaptation may oscillate or converge slowly. In very high dimensions with
# tight bounds, isotropic mutations may be inefficient; clipping can also
# reduce effective diversity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        if dim <= 0:
            raise ValueError("dim must be positive")
        if budget <= 0:
            # No evaluations allowed; return a deterministic in-bounds point.
            lb, ub = self._get_bounds(func, dim)
            x0 = np.clip(np.zeros(dim, dtype=float), lb, ub)
            return x0, float("inf")

        lb, ub = self._get_bounds(func, dim)
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        # Population sizes chosen to be robust across dimensions while staying cheap.
        # Keep at least 2 for recombination; limit offspring to keep evaluation budget safe.
        mu = max(2, int(2 + np.sqrt(dim)))
        # Offspring per iteration; ensure at least 1.
        lam = max(1, int(4 + 2 * np.sqrt(dim)))
        # Clamp population to not exceed budget (since each candidate needs an evaluation).
        mu = min(mu, budget)
        lam = min(lam, max(1, budget - mu))

        rng = np.random
        evals = 0

        # Initial sigma: fraction of diagonal length, with floor to avoid stagnation.
        span = (ub - lb)
        # If bounds are degenerate, fall back to 1.0 in those dimensions.
        span_safe = np.where(span > 0, span, 1.0)
        sigma = 0.3 * float(np.mean(span_safe))
        sigma = max(sigma, 1e-12)

        # --- Initialize population uniformly in bounds (budget-aware) ---
        n_init = min(mu, budget)
        pop = rng.uniform(lb, ub, size=(n_init, dim))
        vals = np.empty(n_init, dtype=float)
        for i in range(n_init):
            vals[i] = func(pop[i])
        evals += n_init

        # If budget is smaller than mu, we can stop immediately.
        best_idx = int(np.argmin(vals))
        best_x = pop[best_idx].copy()
        best_y = float(vals[best_idx])

        # If we have used all budget, return.
        if evals >= budget:
            return best_x, best_y

        # --- Main optimization loop in evaluation blocks ---
        # Each iteration evaluates `k` offspring; k is truncated to fit remaining budget.
        # We keep top mu parents each iteration via (mu+lam) selection.
        while evals < budget:
            remaining = budget - evals
            # Evaluate at most remaining; also cap at lam.
            k = min(lam, remaining)
            if k <= 0:
                break

            # Select elites from current population for recombination center.
            # Recompute parent set if mu > current pop size (can happen at start).
            parent_order = np.argsort(vals)
            parent_order = parent_order[: min(mu, len(vals))]
            elites = pop[parent_order]

            # Recombination center: weighted mean favoring best elites.
            # Use weights that decay with rank.
            eranks = np.arange(len(elites), dtype=float)
            weights = np.exp(-eranks)
            weights /= np.sum(weights)
            center = np.sum(elites * weights[:, None], axis=0)

            # Diversity injection probability: if stuck, sample also around random elite.
            # This helps escape in some flat or multimodal landscapes.
            inject = 0.2

            # Generate offspring: isotropic Gaussian around center / elite.
            offspring = np.empty((k, dim), dtype=float)
            for i in range(k):
                if rng.rand() < inject and len(elites) > 0:
                    base = elites[rng.randint(0, len(elites))]
                else:
                    base = center
                cand = base + sigma * rng.randn(dim)
                # Clip to bounds (constraint handling).
                offspring[i] = np.minimum(np.maximum(cand, lb), ub)

            # Evaluate offspring within budget.
            off_vals = np.empty(k, dtype=float)
            for i in range(k):
                off_vals[i] = func(offspring[i])
            evals += k

            # Track global best
            it_best_idx = int(np.argmin(off_vals))
            it_best_y = float(off_vals[it_best_idx])
            if it_best_y < best_y:
                best_y = it_best_y
                best_x = offspring[it_best_idx].copy()
                improved = True
            else:
                improved = False

            # (μ+λ) selection: keep best mu among parents and offspring.
            combined_pop = np.vstack((pop, offspring))
            combined_vals = np.concatenate((vals, off_vals))
            order = np.argsort(combined_vals)
            keep = min(mu, len(order))
            pop = combined_pop[order[:keep]]
            vals = combined_vals[order[:keep]]

            # Step size adaptation: success-based.
            # If improved, slightly reduce sigma for exploitation; otherwise increase.
            if improved:
                sigma *= 0.9
            else:
                sigma *= 1.07

            # Avoid sigma exploding too large relative to span.
            sigma_max = float(0.5 * np.mean(span_safe)) + 1e-12
            sigma_max = max(sigma_max, 1e-6)
            sigma = min(sigma, sigma_max)

        return best_x, best_y

    @staticmethod
    def _get_bounds(func, dim):
        # Bounds can be provided as:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        if hasattr(func, "bounds") and func.bounds is not None:
            b = func.bounds
            lb = getattr(b, "lb", None)
            ub = getattr(b, "ub", None)
            if lb is not None and ub is not None:
                lb = np.asarray(lb, dtype=float)
                ub = np.asarray(ub, dtype=float)
            else:
                lb = ub = None
        else:
            lb = ub = None

        if lb is None or ub is None:
            if hasattr(func, "lower") and hasattr(func, "upper"):
                lb = np.asarray(func.lower, dtype=float)
                ub = np.asarray(func.upper, dtype=float)
            else:
                raise AttributeError(
                    "Objective must provide bounds via func.lower/func.upper "
                    "or func.bounds.lb/func.bounds.ub."
                )

        # Support scalar bounds or length-d arrays.
        if lb.size == 1:
            lb = np.full(dim, float(lb), dtype=float)
        if ub.size == 1:
            ub = np.full(dim, float(ub), dtype=float)

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.shape[0] != dim or ub.shape[0] != dim:
            raise ValueError(f"Bounds must match dimension {dim}. Got lb={lb.shape}, ub={ub.shape}.")

        # Ensure lb <= ub (swap if needed)
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        return lb2, ub2
