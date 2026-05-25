import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box optimizer for minimization using a
# budgeted Evolution Strategy (ES) with self-adaptive global step size. It works for any
# dimension by maintaining a population of candidate solutions sampled around the current
# best.
# Search state: The algorithm tracks a current best point (best_x) and its objective value
# (best_y), plus a global step-size (sigma) used to control the perturbation scale. It also
# maintains iteration counters and uses the provided evaluation budget.
# Candidate generation: At each iteration, it samples lambda offspring by adding Gaussian
# noise to the current best, scaled by sigma. It then evaluates each offspring using the
# black-box function. The population size is chosen based on dimension for robust performance.
# Selection and replacement: From the evaluated offspring, it selects the best candidate (lowest
# objective) and replaces the current best if an improvement is found. Selection pressure is
# thus greedy: only improvements update the incumbent.
# Adaptation: sigma is adapted using a simple 1/5th success rule: if recent improvements are
# frequent enough, sigma grows; otherwise it shrinks. This helps handle different landscapes
# without requiring gradients.
# Exploration mechanisms: The Gaussian sampling around the best provides exploration early and
# contracts automatically as sigma shrinks when improvements become rare.
# Exploitation mechanisms: Greedy replacement of the incumbent ensures the search concentrates
# around the best-so-far once good regions are found.
# Boundary handling: Candidate solutions are clamped to the feasible box derived from the
# function bounds. Values are clipped to [lb, ub] to keep evaluations valid.
# Budget strategy: The algorithm never exceeds the provided evaluation budget. It computes how
# many evaluations it can perform for the chosen population and stops early if the budget is
# about to be exhausted. It also ensures the incumbent is evaluated at least once.
# Closest known influences: The structure resembles a (μ+λ)-ES with greedy (1-best) replacement
# and self-adaptive step size (1/5 success rule).
# Novelty or unusual aspects: The code uses a conservative, budget-aware loop with an adaptive
# sigma rule and robust bound parsing that supports multiple possible bound attribute layouts.
# Failure modes: If the objective is highly irregular/noisy, greedy updates may stall; in very
# flat landscapes, sigma may shrink prematurely. However, the success-based adaptation and
# clamping help mitigate these issues.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()

        # --- Parse bounds from func ---
        lb, ub = None, None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and func.bounds is not None:
            b = func.bounds
            # expected layouts: b.lb / b.ub
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
        if lb is None or ub is None:
            raise AttributeError(
                "Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )
        if lb.shape == ():  # scalar bounds
            lb = np.full(self.dim, float(lb))
            ub = np.full(self.dim, float(ub))
        else:
            lb = lb.reshape(-1).astype(float)
            ub = ub.reshape(-1).astype(float)
        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Ensure ordering
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo

        # Degenerate spans -> use small perturbations around bounds midpoint
        midpoint = 0.5 * (lo + hi)
        # Initial sigma scale: fraction of box size; if span is ~0 use 1.
        sigma = 0.25 * np.mean(np.where(span > 0, span, 1.0))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        # --- Evaluation budget management ---
        remaining = self.budget
        if remaining <= 0:
            # No evaluations allowed: return a feasible midpoint with inf objective
            best_x = np.clip(midpoint, lo, hi)
            return best_x, float("inf")

        eval_count = 0

        def eval_f(x):
            nonlocal eval_count, remaining
            if eval_count >= self.budget:
                # Should not happen due to checks; keep safe.
                return float("inf")
            y = func(x)
            eval_count += 1
            return float(y)

        # --- Initialize incumbent: random feasible point + evaluate ---
        # Prefer midpoint if finite; then optionally a random restart if budget allows.
        x0 = midpoint
        best_x = np.clip(x0, lo, hi)
        best_y = eval_f(best_x)
        remaining = self.budget - eval_count

        # --- Main loop parameters ---
        dim = self.dim
        # Population size: scalable and budget-aware.
        # Use lambda based on dimension, but keep reasonable.
        lam = int(np.clip(4 + 3 * np.sqrt(dim), 4, 40))
        # For small budgets, adjust lambda.
        lam = max(2, min(lam, self.budget - 1)) if self.budget > 1 else 1

        # Success tracking for 1/5 rule
        # Track successes across the last few iterations.
        window = 5
        succ = 0
        iter_hist = []

        # How many iterations can we do at most, given lambda evals each iteration
        # We'll break when remaining is insufficient to evaluate at least one offspring.
        while remaining > 0:
            # Determine how many offspring we can evaluate this round.
            k = min(lam, remaining)
            if k <= 0:
                break

            # Generate candidates: best_x + sigma * Normal(0,1) per coordinate.
            # Use isotropic Gaussian for robustness across dims.
            # Broadcasting: (k, dim)
            noise = rng.standard_normal(size=(k, dim))
            X = best_x[None, :] + sigma * noise

            # Boundary handling: clamp to feasible box.
            X = np.clip(X, lo[None, :], hi[None, :])

            # Evaluate and select best
            ys = np.empty(k, dtype=float)
            best_i = 0
            best_off_y = float("inf")
            for i in range(k):
                yi = eval_f(X[i])
                ys[i] = yi
                if yi < best_off_y:
                    best_off_y = yi
                    best_i = i

            remaining = self.budget - eval_count

            # Improvement check (greedy update).
            improved = best_off_y < best_y
            if improved:
                best_y = best_off_y
                best_x = X[best_i].copy()

            # Update success statistics and adapt sigma using 1/5 rule.
            iter_hist.append(1 if improved else 0)
            succ = sum(iter_hist)
            if len(iter_hist) > window:
                iter_hist = iter_hist[-window:]
                succ = sum(iter_hist)

            # Only adapt when we have enough history (or early on with smaller history)
            # Target success probability: 0.2 (one-fifth).
            p = succ / max(1, len(iter_hist))
            if p > 0.2:
                sigma *= 1.22
            else:
                sigma /= 1.22

            # Keep sigma within reasonable bounds to avoid numerical issues.
            # Upper bound: about the box diameter scale.
            box_diameter = np.linalg.norm(span) / np.sqrt(dim) if np.any(span > 0) else 1.0
            if not np.isfinite(box_diameter) or box_diameter <= 0:
                box_diameter = 1.0
            sigma = float(np.clip(sigma, 1e-12 * box_diameter, 2.0 * box_diameter))

            # If budget is extremely tight, loop will end naturally.
            # Optional micro-stopping: if sigma is tiny, further exploration is unlikely.
            if sigma <= 1e-12 * box_diameter and remaining <= 0:
                break

        return best_x, best_y
