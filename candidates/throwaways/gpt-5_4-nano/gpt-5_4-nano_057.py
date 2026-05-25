# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer using a
# resilient mixture of coordinate-wise local search and stochastic sampling.
# It maintains a single incumbent solution and refines it using progressively
# shrinking step sizes, while intermittently injecting random candidate
# solutions to escape stagnation.
# Search state: Stores current best x (incumbent), best y value, an evaluation
# counter, current step scale(s), and recent improvement history for
# stagnation detection.
# Candidate generation: Generates candidates in two modes:
# (1) Exploitation: coordinate perturbations around the incumbent using the
#     current step size (including both +/- directions), plus a small number of
#     random directions for robustness.
# (2) Exploration: uniform random samples across the domain at a rate that
#     increases when progress stalls.
# Selection and replacement: Any candidate with strictly lower objective replaces
# the incumbent. The best value is tracked; improvement history updates when
# the incumbent improves.
# Adaptation: If no improvement is observed over a window, the algorithm
# increases exploration (more random samples) and/or enlarges the perturbation
# occasionally, while successful improvements reduce step size to focus search.
# Exploration mechanisms: Random uniform samples across bounds; also uses a
# small random-direction perturbation during exploitation to avoid being purely
# axis-aligned.
# Exploitation mechanisms: Coordinate-wise bracket moves with mirrored steps,
# plus a fallback reflective step when proposals go out of bounds (via clipping).
# Boundary handling: Proposed points are clipped to [lb, ub]. Step sizes are kept
# within reasonable numeric ranges based on domain size.
# Budget strategy: Computes the number of evaluations to use by distributing the
# budget across iterative refinement phases, never exceeding the given budget.
# Closest known influences: Inspired by pattern search / coordinate descent with
# stochastic restarts and step-size adaptation; implemented in a black-box,
# compact form without external dependencies.
# Novelty or unusual aspects: Uses a unified "eval budget governor" and a
# stagnation-triggered blend between coordinate exploration and uniform random
# exploration, tuned to be robust across dimensions and bound types.
# Failure modes: If the objective landscape is extremely deceptive or bounds
# are very tight, the algorithm may spend most time exploiting and stagnate;
# the exploration schedule mitigates this but cannot guarantee global optimality.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = max(0, int(self.budget))
        if budget == 0 or dim <= 0:
            # Return a dummy point; must not call func.
            lb, ub = self._get_bounds(func, dim)
            x0 = lb.copy() if np.all(np.isfinite(lb)) else np.zeros(dim)
            return x0, float("inf")

        lb, ub = self._get_bounds(func, dim)

        # Ensure arrays are float and broadcast-safe.
        lb = np.asarray(lb, dtype=float).reshape(dim)
        ub = np.asarray(ub, dtype=float).reshape(dim)
        # Handle degenerate/invalid bounds robustly.
        span = ub - lb
        span = np.where(np.isfinite(span) & (span > 0), span, 0.0)
        span_safe = span.copy()
        span_safe[span_safe == 0] = 1.0  # avoid division/scale issues

        evals = 0

        def clip(x):
            # Clip to bounds; if some bounds are weird, clip still works.
            return np.minimum(ub, np.maximum(lb, x))

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                # Safety: never exceed budget.
                return float("inf")
            y = func(np.asarray(x, dtype=float))
            evals += 1
            # In case objective returns numpy scalar, coerce to float.
            return float(np.asarray(y).item())

        # Initialization: start from a random point and also test a second point if budget allows.
        x_best = clip(lb + np.random.rand(dim) * span_safe)
        y_best = evaluate(x_best)

        if evals < budget:
            x2 = clip(lb + np.random.rand(dim) * span_safe)
            y2 = evaluate(x2)
            if y2 < y_best:
                x_best, y_best = x2, y2

        # Step size initialization: fraction of domain span.
        # If span is 0 for some coordinates, step becomes 0 there.
        step = 0.25 * span_safe
        step = np.maximum(step, 1e-12)  # allow progress even if tiny spans

        # Stagnation tracking
        best_improve = 0.0
        hist = []
        window = max(8, min(32, budget // 6 + 1))
        # Determine phase counts to use budget smoothly.
        # Keep phases small to reduce control overhead.
        phases = max(1, min(10, budget // (dim + 2) + 1))
        evals_per_phase = max(1, budget // phases)

        # Helper: generate candidates (exploitation or exploration)
        def exploitation_candidates(x0, step_size, num_coords=None):
            # Coordinate perturbations: tries +/- along selected coords and one random direction.
            # Returns a list of candidate points (not evaluated here).
            if num_coords is None:
                # Use up to all coords if dimension small; otherwise sample subset.
                num_coords = dim if dim <= 12 else max(4, dim // 4)

            # Choose coordinates: favor those with larger step/span.
            # To stay robust, just sample without replacement.
            idx = np.random.choice(dim, size=min(num_coords, dim), replace=False)

            candidates = []
            # Try mirrored moves for selected coordinates
            for i in idx:
                d = step_size[i]
                if d == 0:
                    continue
                x = x0.copy()
                x[i] = x0[i] - d
                candidates.append(clip(x))
                x = x0.copy()
                x[i] = x0[i] + d
                candidates.append(clip(x))

            # Add a random-direction move (helps escape axis-aligned traps)
            if np.any(step_size > 0):
                for _ in range(2):
                    r = np.random.normal(size=dim)
                    # Scale direction by step_size componentwise
                    # Normalize to avoid huge effects across dimensions.
                    nr = np.linalg.norm(r)
                    if nr == 0:
                        continue
                    r = r / nr
                    x = x0 + r * step_size
                    candidates.append(clip(x))
                    x = x0 - r * step_size
                    candidates.append(clip(x))

            return candidates

        def exploration_candidates(num_samples):
            # Uniform random candidates across the domain.
            # span_safe used to avoid all-zero span_safe issues; clipping will fix.
            return [clip(lb + np.random.rand(dim) * span_safe) for _ in range(num_samples)]

        # Main loop: budget governor
        for phase in range(phases):
            if evals >= budget:
                break

            # Determine how many evaluations to spend in this phase
            remaining = budget - evals
            k = min(evals_per_phase, remaining)
            if k <= 0:
                break

            # Stagnation measure
            recent_best = y_best
            # update from history if available
            if len(hist) >= window:
                recent = hist[-window:]
                # improvement if any strict decrease occurred in recent window
                recent_improved = (min(recent) < min(hist[:-window]) if len(hist) > window else False)
                if not recent_improved:
                    stagnated = True
                else:
                    stagnated = False
            else:
                stagnated = (len(hist) > 0 and (phase > 0) and (max(hist) - min(hist)) < 1e-12)

            # Adapt exploration/exploitation blend
            # When stagnated, do more exploration and sometimes re-expand step.
            if stagnated:
                explore_rate = 0.55
                step = np.minimum(step * 1.35, 0.75 * span_safe + 1e-12)
            else:
                explore_rate = 0.18
                step = np.maximum(step * 0.82, 1e-12)

            num_explore = int(round(k * explore_rate))
            num_exploit = k - num_explore

            # Build candidate set for exploitation
            cand_list = []
            if num_exploit > 0:
                # Use a number of coordinate perturbations roughly proportional to k.
                # This is a heuristic; exact evals are governed by remaining budget.
                # Select a subset of candidates then evaluate until budget runs out.
                # Limit candidate count to keep evaluations within k.
                # Use larger num_coords early when step is larger.
                scale = float(np.mean(step) / (np.mean(span_safe) + 1e-12))
                num_coords = dim if dim <= 8 else max(4, int(dim / 3))
                if scale < 0.2:
                    num_coords = max(4, num_coords // 2)
                cand_list.extend(exploitation_candidates(x_best, step_size=step, num_coords=num_coords))

            # Add exploration candidates
            if num_explore > 0:
                cand_list.extend(exploration_candidates(num_explore))

            # Shuffle for mixed evaluation order (avoid bias)
            if len(cand_list) > 1:
                cand_list = np.random.permutation(np.array(cand_list, dtype=float)).tolist()

            # Evaluate candidates until k evaluations consumed
            for x in cand_list:
                if evals >= budget or evals >= budget - (remaining - k + 0):
                    # Not strictly needed, but keep safe.
                    pass
                if evals >= budget:
                    break
                if evals > budget - (remaining - k):
                    # enforce k more precisely
                    break

                y = evaluate(x)
                if y < y_best:
                    x_best, y_best = x, y
                    # If improvement is meaningful, slightly tighten step
                    step = np.maximum(step * 0.9, 1e-12)
                    best_improve = max(best_improve, abs(y_best - y) + 1e-16)

            hist.append(y_best)

        return x_best, y_best

    def _get_bounds(self, func, dim):
        # Try bounds from preferred attributes.
        lb = ub = None

        # Common formats:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, "bounds") and func.bounds is not None and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = func.bounds.lb
            ub = func.bounds.ub

        if lb is None or ub is None:
            # If bounds missing, assume [-5, 5] to keep algorithm functional.
            # (Harness likely provides bounds, but this prevents crashes.)
            lb = -5.0 * np.ones(dim, dtype=float)
            ub = 5.0 * np.ones(dim, dtype=float)

        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        # Broadcast to dim if possible.
        if lb.size == 1:
            lb = np.full(dim, float(lb.item()), dtype=float)
        if ub.size == 1:
            ub = np.full(dim, float(ub.item()), dtype=float)

        # If shapes mismatch but can be reshaped to dim:
        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != dim:
            # Fallback: pad/truncate deterministically
            if lb.size > dim:
                lb = lb[:dim]
            else:
                lb = np.pad(lb, (0, dim - lb.size), mode="edge")
        if ub.size != dim:
            if ub.size > dim:
                ub = ub[:dim]
            else:
                ub = np.pad(ub, (0, dim - ub.size), mode="edge")

        # Ensure lb <= ub when possible.
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        # If any non-finite, replace with safe defaults.
        finite_lb = np.isfinite(lb2)
        finite_ub = np.isfinite(ub2)
        if not np.all(finite_lb):
            lb2 = np.where(finite_lb, lb2, -5.0)
        if not np.all(finite_ub):
            ub2 = np.where(finite_ub, ub2, 5.0)

        return lb2, ub2
