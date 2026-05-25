# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# that does not assume gradient access. It keeps a population of candidate
# solutions and iteratively improves them by combining (1) local exploitation
# around the current best point and (2) global exploration via random
# re-sampling and differential-style moves.
#
# Search state: The algorithm tracks the evaluation budget, the number of
# evaluations used, a best-so-far solution (best_x, best_y), and a small
# population of points with their objective values. A per-dimension scale
# controls how far new candidates are perturbed.
#
# Candidate generation: Each iteration generates candidates by:
#   - Local Gaussian steps around the current best (exploitation).
#   - Differential-style steps using two other population members,
#     scaled by the current search scale (exploration / recombination).
#   - Occasional pure random restarts uniformly in the bounds.
#
# Selection and replacement: Candidates are evaluated and, if they improve
# over the worst individual in the population, they replace it. The global
# best is updated whenever a new lower objective value is found.
#
# Adaptation: The search scale shrinks when improvements are found (to
# concentrate exploitation) and grows slightly when progress stalls
# (to encourage exploration). The mutation step size is proportional to the
# variable range.
#
# Exploration mechanisms: Random restarts and differential-style moves from
# population members allow the algorithm to traverse the domain beyond the
# neighborhood of the current best.
#
# Exploitation mechanisms: Gaussian perturbations centered at the incumbent
# best point drive local refinement.
#
# Boundary handling: After generating a candidate, it is clipped to the valid
# bounds. This ensures all evaluations remain feasible.
#
# Budget strategy: The constructor receives a total evaluation budget. The
# algorithm initializes by sampling a small number of points, then repeatedly
# generates and evaluates candidates while never exceeding the remaining
# evaluation budget.
#
# Closest known influences: The design blends elements similar in spirit to
# evolution strategies / CMA-like step adaptation (simplified) and
# differential evolution-style mutation, tailored for strict evaluation
# budgeting and bound handling.
#
# Novelty or unusual aspects: The code uses a minimal, dimension-robust
# population strategy with an adaptive per-iteration scale based on success
# events, while keeping the overall logic and state very small.
#
# Failure modes: If the objective is extremely noisy or the budget is too
# small, the population may not get sufficient diversity. Also, clipping to
# bounds can bias search near tight constraints. The algorithm mitigates this
# with occasional restarts and scale growth on stagnation.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lower, upper = self._get_bounds(func, self.dim)
        lb = lower.astype(float, copy=False)
        ub = upper.astype(float, copy=False)
        span = ub - lb
        span = np.where(span > 0, span, 1.0)  # avoid degeneracy

        # Ensure budget is at least enough to return something.
        if self.budget <= 0:
            # Still call the objective once if possible to be safe.
            x0 = np.clip(lb, lb, ub)
            return x0, float(func(x0))

        # Population size: small but scalable; must not exceed budget.
        pop_size = int(np.clip(4 + self.dim, 4, 20))
        pop_size = min(pop_size, self.budget)

        # Track evaluations strictly; this wrapper must be used for every call.
        evals = 0

        def eval_obj(x):
            nonlocal evals
            if evals >= self.budget:
                # Should never happen if logic is correct; raise to catch bugs.
                raise RuntimeError("Evaluation budget exceeded.")
            y = func(x)
            evals += 1
            return float(y)

        rng = np.random

        # Initialize population uniformly in bounds.
        pop = lb + rng.random((pop_size, self.dim)) * span
        vals = np.empty(pop_size, dtype=float)
        best_idx = 0
        best_x = None
        best_y = np.inf

        for i in range(pop_size):
            vals[i] = eval_obj(pop[i])
            if vals[i] < best_y:
                best_y = vals[i]
                best_x = pop[i].copy()
                best_idx = i

        # Adaptive scale: starts as a fraction of span.
        # Use a scalar base with per-dimension scaling for robustness.
        base_scale = 0.25  # fraction of span
        scale = base_scale * span

        # Stagnation tracking.
        no_improve_steps = 0
        max_stagnation = 8 + self.dim // 2

        # Main loop: each step tries to insert a candidate into the population.
        while evals < self.budget:
            # Identify best and worst in population.
            worst_idx = int(np.argmax(vals))
            best_idx = int(np.argmin(vals))
            incumbent = pop[best_idx]

            # Success flag and step budget: we only evaluate if budget remains.
            # We can try up to a small number of candidates per loop, but keep
            # it simple: one candidate per loop to stay budget-tight.
            if evals >= self.budget:
                break

            # Choose move type.
            # With some probability restart to maintain exploration.
            r = rng.random()
            if r < 0.10:
                # Random restart within bounds.
                cand = lb + rng.random(self.dim) * span
            else:
                # Exploitation or differential-style exploration.
                if r < 0.65:
                    # Local exploitation around incumbent (Gaussian step).
                    # Heavier tail occasionally for robustness.
                    if rng.random() < 0.15:
                        step = rng.standard_cauchy(self.dim) * (0.10 * scale)
                    else:
                        step = rng.standard_normal(self.dim) * scale
                    cand = incumbent + step
                else:
                    # Differential-style mutation: cand = a + F*(b - c) + noise
                    # Pick distinct indices.
                    idxs = np.arange(pop_size)
                    # Ensure distinct and usable indices.
                    # Avoid complicated sampling overhead: use permutation.
                    perm = rng.permutation(pop_size)
                    a_i, b_i, c_i = perm[0], perm[1], perm[2]
                    a = pop[a_i]
                    b = pop[b_i]
                    c = pop[c_i]
                    F = 0.4 + 0.4 * rng.random()  # in [0.4, 0.8]
                    noise = rng.standard_normal(self.dim) * (0.10 * scale)
                    cand = a + F * (b - c) + noise

            # Boundary handling: clip to bounds.
            cand = np.clip(cand, lb, ub)

            # Evaluate candidate (strict budget).
            y = eval_obj(cand)

            # Selection and replacement: replace worst if better.
            if y < vals[worst_idx]:
                pop[worst_idx] = cand
                vals[worst_idx] = y

            # Update global best.
            if y < best_y:
                best_y = y
                best_x = cand.copy()
                no_improve_steps = 0

                # Shrink scale slightly after success for exploitation.
                scale *= (0.90 + 0.05 * rng.random())
            else:
                no_improve_steps += 1

                # If stuck for a while, increase scale to explore more.
                if no_improve_steps >= max_stagnation:
                    scale *= 1.15
                    no_improve_steps = 0

            # Also very gently shrink over time to favor convergence.
            # Keep this mild to avoid premature convergence.
            t = evals / max(1, self.budget)
            scale *= (1.0 - 0.01 * t)

            # Prevent scale from becoming too tiny or exploding.
            scale = np.clip(scale, 1e-12 * span, 2.0 * span)

        return best_x, float(best_y)

    @staticmethod
    def _get_bounds(func, dim):
        # Bounds can be provided in multiple ways as per requirements.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = func.lower
            upper = func.upper
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lower = func.bounds.lb
            upper = func.bounds.ub
        else:
            raise AttributeError(
                "Function object must provide bounds via (lower, upper) "
                "or func.bounds.lb / func.bounds.ub."
            )

        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)

        if lower.size != dim or upper.size != dim:
            raise ValueError(f"Bounds must match dim={dim}. Got lower={lower.size}, upper={upper.size}.")

        # Allow for numeric issues; enforce correct ordering.
        if np.any(upper < lower):
            raise ValueError("Invalid bounds: some upper values are less than lower values.")

        return lower, upper
