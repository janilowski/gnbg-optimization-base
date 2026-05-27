# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm for
# continuous domains using a combination of coordinate-wise local search and
# global restart-style sampling. It is designed to work under a strict evaluation
# budget and be robust across dimensions.
#
# Search state: The algorithm maintains the current best solution (best_x, best_y),
# a current step-size sigma controlling exploration scale, and a history of
# evaluated points to avoid redundant work when possible.
#
# Candidate generation: Each iteration proposes a set of candidates:
#   1) A small coordinate-wise "probing" around the current best by moving along
#      randomly chosen coordinate directions by +/- sigma.
#   2) A handful of purely random points to provide diversification.
#   3) (When sigma is small) optional "recombination" by averaging best_x with
#      random points, creating intermediate candidates.
#
# Selection and replacement: All proposed candidates are evaluated (within the
# remaining budget). The candidate with the smallest objective value replaces the
# current best if it improves.
#
# Adaptation: sigma is adapted multiplicatively: it is reduced when improvements
# are found (more exploitation) and increased slightly when improvements stall
# (more exploration).
#
# Exploration mechanisms: Random points and occasional averaging with random points
# help explore new regions; coordinate probing explores local structure around the
# best solution.
#
# Exploitation mechanisms: Coordinate-wise probing around the current best provides
# directed local improvement while sigma shrinks after successful steps.
#
# Boundary handling: Any candidate is clipped to the provided bounds
# (from func.lower/func.upper or func.bounds.lb/ub). If bounds are infinite or missing,
# fallback defaults are used and clipping is skipped.
#
# Budget strategy: The algorithm estimates a per-iteration candidate batch size and
# computes the number of iterations possible from the total evaluation budget.
# It never evaluates more than `budget` times by dynamically stopping when the budget
# is nearly exhausted. The final return is the best solution seen.
#
# Closest known influences: The design is inspired by simple evolutionary strategies
# and coordinate local search hybrids, using step-size adaptation and restart-like
# diversification under a strict query limit.
#
# Novelty or unusual aspects: It is tailored to evaluation-limited settings by combining
# a deterministic batch evaluation schedule (coordinate probes) with stochastic diversification,
# and it includes careful bookkeeping to strictly honor the budget.
#
# Failure modes: If the objective is extremely noisy or deceptive, sigma adaptation may
# overfocus locally; budget scarcity can limit exploration. If bounds are poorly specified,
# the algorithm may clip to an unhelpful region.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import math
import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = max(0, int(self.budget))
        if dim <= 0 or budget <= 0:
            # No evaluations possible; return a zero vector by convention.
            x = np.zeros(dim, dtype=float)
            return x, float("inf")

        # ---- Bounds handling ----
        lb = None
        ub = None
        # Prefer func.lower / func.upper
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        if lb is not None and ub is not None:
            # Ensure shapes are compatible
            if lb.shape == () or lb.shape == (1,):
                lb = np.full(dim, float(lb), dtype=float)
            if ub.shape == () or ub.shape == (1,):
                ub = np.full(dim, float(ub), dtype=float)
            lb = lb.reshape(-1)
            ub = ub.reshape(-1)
            if lb.size != dim or ub.size != dim:
                # Fallback if malformed
                lb, ub = None, None

        # Helper: clip only if both finite bounds exist
        finite_lb = lb is not None and np.all(np.isfinite(lb))
        finite_ub = ub is not None and np.all(np.isfinite(ub))
        use_bounds = lb is not None and ub is not None and finite_lb and finite_ub

        def clip(x):
            if use_bounds:
                return np.minimum(np.maximum(x, lb), ub)
            return x

        # ---- Evaluation with strict budget ----
        eval_count = 0
        best_x = None
        best_y = float("inf")

        # History to reduce pointless exact duplicates (only helps a bit; keeps code simple)
        # Store rounded points to combat float noise.
        seen = set()

        def round_key(x):
            # Robust key for floats: round to 12 decimals
            return tuple(np.round(x, 12).tolist())

        def eval_x(x):
            nonlocal eval_count, best_x, best_y
            if eval_count >= budget:
                return None
            x = np.asarray(x, dtype=float)
            x = clip(x)
            key = round_key(x)
            # If already evaluated exactly (after rounding), don't count again
            if key in seen:
                # Still attempt to find better? We don't have cached value, so ignore.
                # The harness objective calls are expected deterministic; skipping is safe.
                return None
            seen.add(key)
            y = float(func(x))
            eval_count += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # ---- Initialize ----
        # Choose initial sigma based on bounds or a heuristic scale.
        if use_bounds:
            span = ub - lb
            # Guard against zero spans
            span = np.where(span > 0, span, 1.0)
            sigma = 0.3 * float(np.median(span))
            x0 = lb + np.random.rand(dim) * (ub - lb)
        else:
            sigma = 1.0
            x0 = np.random.randn(dim)

        # Make sure best_x is set by evaluating at least one point.
        eval_x(x0)
        if best_x is None:
            best_x = clip(x0)

        # If budget is only 1, we're done.
        if eval_count >= budget:
            return best_x, best_y

        # ---- Main loop ----
        # We use batches of candidates each iteration to amortize loop overhead.
        # Candidate count is capped to remaining budget.
        # Mix of exploitation (coordinate probes) and exploration (random samples / averaging).
        # The loop count is derived from remaining budget and batch size.
        # Batch sizes adapt to dimensions and remaining budget.
        while eval_count < budget:
            remaining = budget - eval_count

            # Coordinate probing: choose how many coordinates to try this round.
            # For high dim, probing fewer coordinates reduces wasted evaluations.
            # At least 1 coordinate.
            k_coords = min(dim, max(1, int(math.ceil(dim / 4))))
            # How many random points to include
            n_rand = min(6, max(1, remaining // 4))
            # How many averaged/intermediate points
            n_avg = min(4, max(0, remaining // 6))
            # Total candidates
            n_probe = 2 * k_coords  # +/- moves along each chosen coordinate
            total = n_probe + n_rand + n_avg
            if total > remaining:
                # Trim proportionally to fit remaining budget
                # Keep at least one probe pair if possible, otherwise random.
                # Here we prioritize probes, then random, then averaging.
                total = remaining

            # Determine actual counts that sum to <= remaining
            probe_budget = min(n_probe, max(0, total))
            rem2 = total - probe_budget
            rand_budget = min(n_rand, max(0, rem2))
            rem3 = rem2 - rand_budget
            avg_budget = min(n_avg, max(0, rem3))

            candidates = []

            # --- Exploitation: coordinate probing around best_x ---
            if probe_budget > 0:
                # Randomly pick coordinates to probe
                coords = np.random.choice(dim, size=min(dim, k_coords), replace=False)
                # For each coordinate, propose +/-sigma steps
                # Use random direction scaling slightly to avoid symmetry issues.
                # To exactly meet probe_budget, we may use only part.
                # Create list of 2*k_coords candidates then slice.
                probes = []
                # Random scaling factors near 1
                scales = np.exp(0.15 * np.random.randn(len(coords)))
                for i, c in enumerate(coords):
                    s = sigma * float(scales[i])
                    step = np.zeros(dim, dtype=float)
                    step[c] = s
                    probes.append(best_x + step)
                    probes.append(best_x - step)
                # Ensure enough if probe_budget smaller than generated
                if probe_budget < len(probes):
                    # Choose a subset deterministically random
                    idx = np.random.choice(len(probes), size=probe_budget, replace=False)
                    probes = [probes[j] for j in idx]
                candidates.extend(probes)

            # --- Exploration: random samples in bound box or gaussian ---
            if rand_budget > 0:
                if use_bounds:
                    for _ in range(rand_budget):
                        u = np.random.rand(dim)
                        x = lb + u * (ub - lb)
                        candidates.append(x)
                else:
                    # Heavy-tailed-ish exploration by mixing normal with occasional wider jumps
                    for _ in range(rand_budget):
                        if np.random.rand() < 0.2:
                            x = best_x + (sigma * 5.0) * np.random.randn(dim)
                        else:
                            x = best_x + sigma * np.random.randn(dim)
                        candidates.append(x)

            # --- Intermediate recombination: average best with random points ---
            if avg_budget > 0:
                # Create random reference points and average
                if use_bounds:
                    for _ in range(avg_budget):
                        r = lb + np.random.rand(dim) * (ub - lb)
                        t = 0.2 + 0.6 * np.random.rand()
                        candidates.append((1.0 - t) * best_x + t * r)
                else:
                    for _ in range(avg_budget):
                        r = best_x + (sigma * (0.5 + 3.0 * np.random.rand())) * np.random.randn(dim)
                        t = 0.2 + 0.6 * np.random.rand()
                        candidates.append((1.0 - t) * best_x + t * r)

            # --- Evaluate candidates ---
            y_before = best_y
            improved = False
            for x in candidates:
                if eval_count >= budget:
                    break
                prev_best = best_y
                eval_x(x)
                if best_y < prev_best:
                    improved = True

            # If we couldn't evaluate any new points (due to duplicates), just perturb sigma.
            if eval_count >= budget:
                break

            # --- Adapt sigma based on success ---
            if best_y < y_before:
                improved = True

            if improved:
                # Successful step: shrink sigma to focus
                sigma *= 0.82
            else:
                # No improvement: expand slightly to explore
                sigma *= 1.12

            # Keep sigma within reasonable limits
            if use_bounds:
                # Use fraction of domain span for upper cap.
                # If span is tiny, sigma cap becomes small.
                sigma_cap = 0.6 * float(np.median(ub - lb))
                sigma_cap = max(1e-12, sigma_cap)
                sigma = float(np.clip(sigma, 1e-12, sigma_cap))
            else:
                sigma = float(np.clip(sigma, 1e-12, 1e6))

            # If sigma becomes extremely small, re-inject exploration a bit.
            if sigma < 1e-10 and eval_count < budget:
                sigma = max(sigma * 10.0, 1e-3)

        return best_x, best_y
