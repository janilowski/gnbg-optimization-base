# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box optimizer for minimization.
# It uses a population-based search with adaptive step size, periodic recombination,
# and a simple coordinate-wise improvement operator, all while strictly respecting
# the provided evaluation budget.
# Search state: Maintains a small population of candidate points (x) and their
# objective values (y), tracks the global best (best_x, best_y), and maintains
# a per-run mutable step size (sigma).
# Candidate generation: Samples new candidates by adding Gaussian noise scaled
# by sigma to selected parents. Also uses occasional recombination (averaging of
# top individuals) to create offspring.
# Selection and replacement: For each iteration, new candidates are evaluated and the
# population is updated by keeping the best individuals (elitist selection). The
# global best is updated whenever an improved value is found.
# Adaptation: Sigma shrinks when improvements stall and grows slightly when progress
# is observed, based on a simple success counter.
# Exploration mechanisms: Gaussian mutation around different parents plus occasional
# recombination introduces diversity.
# Exploitation mechanisms: Uses a local “coordinate step” refinement attempt around
# the best point (probabilistically) to more quickly reduce the objective near current
# optima.
# Boundary handling: Candidates are clipped to the provided bounds after mutation and
# recombination to ensure feasibility.
# Budget strategy: The optimizer precomputes a maximum number of evaluations from the
# provided budget and uses an internal counter to guarantee it never exceeds it.
# Remaining evaluations are distributed across iterations as needed.
# Closest known influences: Similar in spirit to evolution strategies (ES) with elitist
# selection and step-size adaptation, plus a lightweight coordinate local search.
# Novelty or unusual aspects: The algorithm blends ES-like Gaussian sampling with a
# stochastic coordinate refinement step while keeping the code short and robust.
# Failure modes: If the objective is extremely noisy or highly discontinuous, step-size
# adaptation may oscillate or converge prematurely. With very small budgets, the method
# may behave like a random local search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        lb, ub = _read_bounds(func)

        # Handle degenerate bounds gracefully
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != dim or ub.size != dim:
            # Fallback: scalar bounds
            lb = np.full(dim, float(lb[0]) if lb.size else float(lb), dtype=float)
            ub = np.full(dim, float(ub[0]) if ub.size else float(ub), dtype=float)

        # Ensure lb <= ub
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo
        span = np.where(span == 0.0, 1.0, span)  # avoid divide-by-zero for sigma

        # Budget and bookkeeping
        max_evals = max(1, self.budget)
        evals = 0

        def eval_x(x):
            nonlocal evals
            if evals >= max_evals:
                # Must not exceed budget; return current best if called accidentally
                return best_y
            y = func(np.asarray(x, dtype=float))
            evals += 1
            return float(y)

        rng = np.random.default_rng()  # harness sets np seed; we use default_rng deterministically via global seed

        # Initialization
        n_pop = _population_size(dim, max_evals)
        # Initial sigma based on bounds span
        sigma = 0.25 * np.mean(span)
        sigma = float(sigma) if np.isfinite(sigma) else 1.0
        sigma = max(sigma, 1e-12)

        # Best known
        best_x = None
        best_y = float("inf")

        # Start with uniform random population within bounds
        X = rng.uniform(lo, hi, size=(n_pop, dim))
        Y = np.empty(n_pop, dtype=float)
        for i in range(n_pop):
            if evals >= max_evals:
                break
            y = eval_x(X[i])
            Y[i] = y
            if y < best_y:
                best_y = y
                best_x = X[i].copy()

        # If budget is too small to fill population, return what we have
        if best_x is None:
            # Shouldn't happen, but keep robust
            best_x = lo.copy()
            best_y = eval_x(best_x)

        # Determine iteration count based on remaining budget
        # Each iteration proposes up to n_offspring candidates.
        while evals < max_evals:
            remaining = max_evals - evals
            # Propose a manageable number of offspring based on remaining budget
            n_off = min(n_pop, remaining)

            # Rank parents by objective (lower is better)
            order = np.argsort(Y)
            X = X[order]
            Y = Y[order]

            # Elitist selection: retain top half (or at least 1)
            n_elite = max(1, n_pop // 2)
            parents = X[:n_elite]

            improved = False
            for k in range(n_off):
                if evals >= max_evals:
                    break

                # Parent choice: bias towards better individuals
                # Create a simple probability distribution proportional to 1/rank
                idx = _biased_choice(rng, n_elite, bias=1.7)
                parent = parents[idx]

                # Exploration/exploitation blend:
                # - majority: Gaussian mutation around parent
                # - occasional recombination: average top elites
                if rng.random() < 0.15 and n_elite >= 2:
                    j = _biased_choice(rng, n_elite, bias=1.7)
                    offspring = 0.5 * (parent + parents[j])
                    # Add a smaller perturbation to keep diversity
                    offspring = offspring + 0.35 * sigma * rng.normal(size=dim)
                else:
                    offspring = parent + sigma * rng.normal(size=dim)

                # Stochastic local coordinate refinement around current best
                # (probabilistically, and only when near the center to reduce wasted moves)
                if rng.random() < 0.06:
                    offspring = offspring.copy()
                    # Choose a few coordinates to nudge
                    m = 1 + int(rng.integers(1, 1 + min(3, dim)))
                    coords = rng.choice(dim, size=m, replace=False)
                    # Step size for coordinates: smaller than sigma
                    coord_step = (0.15 + 0.35 * rng.random()) * sigma
                    for c in coords:
                        # Try a small step; direction random
                        offspring[c] = offspring[c] + coord_step * (1.0 if rng.random() < 0.5 else -1.0)

                # Boundary handling: clip to [lo, hi]
                offspring = np.clip(offspring, lo, hi)

                y = eval_x(offspring)
                if y < best_y:
                    best_y = y
                    best_x = offspring.copy()
                    improved = True

                # Replace worst individual (keep population size constant)
                # Maintain feasibility by writing to worst index.
                worst = n_pop - 1
                # If population wasn't full due to small budget, handle carefully
                if k < X.shape[0]:
                    # Replace within sorted population by updating worst
                    # But we haven't recomputed order after each insertion; do direct replacement:
                    X[worst] = offspring
                    Y[worst] = y
                else:
                    # Should not happen; but keep robust
                    X = np.vstack([X, offspring])
                    Y = np.append(Y, y)
                    n_pop = X.shape[0]

            # Adapt sigma based on improvement
            # If no improvement, shrink more aggressively; else grow slightly or keep.
            if improved:
                sigma *= 1.05
            else:
                sigma *= 0.82

            # Keep sigma within reasonable range relative to bounds
            sigma = float(np.clip(sigma, 1e-12, 2.5 * np.mean(span)))

            # Quick coordinate refinement around best_x if we have enough remaining budget
            # This tries to reduce y without large sampling cost.
            if evals + min(dim, 5) <= max_evals and rng.random() < 0.25:
                # Try small coordinate moves in random order, accept improvements only.
                m = min(dim, 5)
                coords = rng.choice(dim, size=m, replace=False)
                step = 0.25 * sigma
                for c in coords:
                    if evals >= max_evals:
                        break
                    x1 = best_x.copy()
                    x2 = best_x.copy()
                    dir1 = 1.0 if rng.random() < 0.5 else -1.0
                    x1[c] = np.clip(x1[c] + dir1 * step, lo[c], hi[c])
                    x2[c] = np.clip(x2[c] - dir1 * step, lo[c], hi[c])

                    y1 = eval_x(x1)
                    if y1 < best_y:
                        best_y = y1
                        best_x = x1.copy()
                        improved = True
                        continue
                    y2 = eval_x(x2)
                    if y2 < best_y:
                        best_y = y2
                        best_x = x2.copy()
                        improved = True

        return best_x, best_y


def _read_bounds(func):
    # Support:
    # - func.lower/func.upper
    # - func.bounds.lb/func.bounds.ub
    if hasattr(func, "lower") and hasattr(func, "upper"):
        return func.lower, func.upper
    if hasattr(func, "bounds"):
        b = func.bounds
        if hasattr(b, "lb") and hasattr(b, "ub"):
            return b.lb, b.ub
    # Some harnesses might provide bounds directly as attributes on the callable object
    if hasattr(func, "lb") and hasattr(func, "ub"):
        return func.lb, func.ub
    if hasattr(func, "lower_bound") and hasattr(func, "upper_bound"):
        return func.lower_bound, func.upper_bound
    raise AttributeError("Could not read bounds from func.lower/upper or func.bounds.lb/ub.")


def _population_size(dim, budget):
    # Keep small for compactness; ensure at least 4 when budget allows.
    # Also cap by budget because each evaluation costs one objective call.
    # Use a heuristic: ~6 + sqrt(dim), capped.
    base = 6 + int(np.sqrt(max(1, dim)))
    # Ensure at least 4 for diversity, but don't exceed budget.
    return int(max(4, min(base, max(4, budget))))


def _biased_choice(rng, n, bias=1.7):
    # Picks an index in [0, n) with probability ~ 1/(rank^bias).
    # rank 0 (best) is most likely.
    if n <= 1:
        return 0
    ranks = np.arange(n, dtype=float)
    w = 1.0 / np.maximum(1.0, (ranks + 1.0) ** bias)
    p = w / w.sum()
    return int(rng.choice(n, p=p))
