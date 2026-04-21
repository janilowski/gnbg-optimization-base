from __future__ import annotations

import numpy as np


class Algorithm:
    """
    Hybrid restart-evolution strategy with elite-guided local refinement.

    Idea summary:
    - We maintain a small population and evolve it using DE-style mutant proposals
      (global exploration) while preserving a compact elite archive (memory).
    - We also add a skip-connection-inspired residual proposal: besides the
      standard mutation/crossover path, we occasionally propose a shortcut move
      that blends the current point directly with elite/global-best directions.
      This creates an alternate update path that can still make progress when the
      main mutation path is temporarily unproductive.
    - In parallel, we run short stochastic local-search bursts around the current
      best and around archived elites with shrinking step sizes (local exploitation).
    - If progress stalls, we trigger soft restarts that re-seed part of the
      population from elites + broad random samples, which helps escape local traps
      without discarding all good information.

    Why this is robust for black-box GNBG-like landscapes:
    1) Differential mutations are scale-free and work decently across dimensions.
    2) The elite archive reuses historically strong points rather than only the
       current generation, improving reliability under noisy progress patterns.
    3) Local bursts with adaptive radius can quickly polish promising basins.
    4) A stall-driven restart policy reduces brittle dependence on one population.

    The implementation is intentionally lightweight (NumPy only), conservative with
    function evaluations, and explicitly budget-safe via a guarded evaluator.
    """

    def __init__(self, budget: int, dim: int):
        # Public interface required by harness.
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # Read bounds through either API supported by the harness.
        lower = np.asarray(getattr(func, "lower", func.bounds.lb), dtype=float)
        upper = np.asarray(getattr(func, "upper", func.bounds.ub), dtype=float)

        # Safety against malformed bounds; if any width is zero we still work.
        span = np.maximum(upper - lower, 1e-12)

        # Hard evaluation counter to guarantee budget compliance.
        evals = 0

        # Global incumbent (best seen solution/value).
        best_x = None
        best_y = float("inf")

        def clamp(x: np.ndarray) -> np.ndarray:
            """Project a candidate back into domain bounds."""
            return np.minimum(np.maximum(x, lower), upper)

        def evaluate(x: np.ndarray):
            """
            Budget-guarded evaluation.

            Returns:
                (y, ok) where ok=False means budget exhausted and y=None.
            """
            nonlocal evals, best_x, best_y
            if evals >= self.budget:
                return None, False

            x_eval = clamp(np.asarray(x, dtype=float))
            y = float(func(x_eval))
            evals += 1

            if y < best_y:
                best_y = y
                best_x = x_eval.copy()

            return y, True

        # If budget is too tiny, gracefully fall back to pure random sampling.
        if self.budget <= 4:
            for _ in range(self.budget):
                x = np.random.uniform(lower, upper, size=self.dim)
                evaluate(x)
            return best_x, best_y

        # Population size: small-to-moderate, dimension aware, budget aware.
        pop_size = int(np.clip(6 + 3 * np.log2(self.dim + 2), 8, 24))
        pop_size = min(pop_size, max(4, self.budget // 6))

        # Deterministic warm-start probes (cheap and often useful):
        # - box center
        # - two mirrored center-offset points along a random direction
        # This can outperform pure random starts on some shifted GNBG instances.
        center = 0.5 * (lower + upper)
        cy, ok = evaluate(center)
        if ok:
            # Keep a tiny warm pool so the initial population can include these.
            warm_x = [center.copy()]
            warm_y = [cy]
        else:
            warm_x = []
            warm_y = []

        if evals + 2 <= self.budget:
            direction = np.random.normal(0.0, 1.0, size=self.dim)
            norm = np.linalg.norm(direction) + 1e-12
            direction /= norm
            offset = 0.2 * span * direction
            x1 = clamp(center + offset)
            x2 = clamp(center - offset)
            y1, ok1 = evaluate(x1)
            if ok1:
                warm_x.append(x1.copy())
                warm_y.append(y1)
            y2, ok2 = evaluate(x2)
            if ok2:
                warm_x.append(x2.copy())
                warm_y.append(y2)

        # Initialize population with opposition-based sampling:
        # for each random sample x, also consider its opposite point
        # x' = lower + upper - x and keep the better one.
        # This often improves starting quality on GNBG-style domains.
        pop_x = list(warm_x)
        pop_y = list(warm_y)
        for _ in range(pop_size):
            x = np.random.uniform(lower, upper, size=self.dim)
            xo = lower + upper - x
            y, ok = evaluate(x)
            if not ok:
                break
            yo, ok2 = evaluate(xo)
            if not ok2:
                pop_x.append(x)
                pop_y.append(y)
                break
            if yo < y:
                pop_x.append(clamp(xo))
                pop_y.append(yo)
            else:
                pop_x.append(x)
                pop_y.append(y)

        if not pop_x:
            # Ultra-defensive return path (should rarely happen).
            return lower.copy(), float("inf")

        pop_x = np.asarray(pop_x, dtype=float)
        pop_y = np.asarray(pop_y, dtype=float)

        # Elite archive stores a small set of best historically seen points.
        elite_k = min(6, len(pop_x))
        elite_idx = np.argsort(pop_y)[:elite_k]
        elite_x = pop_x[elite_idx].copy()
        elite_y = pop_y[elite_idx].copy()

        def refresh_archive(cand_x: np.ndarray, cand_y: float):
            """Insert candidate into elite archive if it improves the worst elite."""
            nonlocal elite_x, elite_y
            worst = int(np.argmax(elite_y))
            if cand_y < elite_y[worst]:
                elite_x[worst] = cand_x
                elite_y[worst] = cand_y

        # Adaptive controls for search behavior.
        no_improve_iters = 0
        prev_best = best_y
        # Local step factor is relative to box width.
        local_sigma = 0.15

        # Main loop: alternate DE-style evolution, local bursts, and restart checks.
        while evals < self.budget:
            n = len(pop_x)
            if n < 4:
                # Repopulate if we ever become too small.
                while evals < self.budget and len(pop_x) < 4:
                    x = np.random.uniform(lower, upper, size=self.dim)
                    y, ok = evaluate(x)
                    if not ok:
                        break
                    pop_x = np.vstack([pop_x, x])
                    pop_y = np.append(pop_y, y)
                n = len(pop_x)
                if n < 4:
                    break

            # Progress ratio in [0, 1]; used to smoothly anneal parameters.
            progress = evals / max(1, self.budget)

            # DE control parameters: slightly more explorative early, stable late.
            F = 0.75 - 0.25 * progress
            CR = 0.9 - 0.2 * progress

            # Perform one generation of DE/current-to-best/1 with binomial crossover.
            order = np.random.permutation(n)
            for i in order:
                if evals >= self.budget:
                    break

                # Choose three distinct donors different from i.
                idx_pool = [j for j in range(n) if j != i]
                a, b, c = np.random.choice(idx_pool, size=3, replace=False)

                xi = pop_x[i]

                # Use a random elite as guidance (archive-informed exploitation).
                guide = elite_x[np.random.randint(len(elite_x))]

                # Mutation: combines current-to-guide and differential term.
                # c is intentionally used to pick an alternative difference source
                # stochastically, which adds diversity without extra parameters.
                if np.random.rand() < 0.5:
                    diff = pop_x[a] - pop_x[b]
                else:
                    diff = pop_x[a] - pop_x[c]
                mutant = xi + F * (guide - xi) + F * diff
                mutant = clamp(mutant)

                # Binomial crossover with at least one changed coordinate.
                cross_mask = np.random.rand(self.dim) < CR
                jrand = np.random.randint(self.dim)
                cross_mask[jrand] = True
                trial = np.where(cross_mask, mutant, xi)
                trial = clamp(trial)

                ty, ok = evaluate(trial)
                if not ok:
                    break

                # Greedy selection (minimization).
                if ty < pop_y[i]:
                    pop_x[i] = trial
                    pop_y[i] = ty
                    refresh_archive(trial, ty)

                # Skip-connection-inspired shortcut proposal:
                # Create an alternative residual-like update path that bypasses
                # some mutation complexity and directly links xi to guide/best_x.
                # This is analogous to residual shortcuts in deep nets: if the
                # primary path underperforms, the shortcut can still carry useful
                # signal (good directional information) forward.
                if (
                    self.budget <= 2000
                    and evals < self.budget
                    and np.random.rand() < 0.3
                    and best_x is not None
                ):
                    alpha = 0.35 + 0.25 * (1.0 - progress)
                    beta = 0.25
                    skip_trial = xi + alpha * (guide - xi) + beta * (best_x - xi)
                    # Keep some coordinate-wise identity passthrough from xi.
                    skip_mask = np.random.rand(self.dim) < 0.5
                    skip_trial = np.where(skip_mask, skip_trial, xi)
                    skip_trial = clamp(skip_trial)

                    sy, oks = evaluate(skip_trial)
                    if not oks:
                        break
                    if sy < pop_y[i]:
                        pop_x[i] = skip_trial
                        pop_y[i] = sy
                        refresh_archive(skip_trial, sy)

            # Periodic local-search bursts around best + one random elite.
            if evals < self.budget:
                # Shrink local search radius over time; keep a floor for robustness.
                local_sigma = max(0.02, local_sigma * 0.96)

                anchors = [(best_x, best_y)]
                if len(elite_x) > 1:
                    ridx = np.random.randint(len(elite_x))
                    anchors.append((elite_x[ridx].copy(), float(elite_y[ridx])))

                for anchor, ay in anchors:
                    if evals >= self.budget:
                        break

                    # 2-4 local attempts, budget-dependent and lightweight.
                    n_steps = 3 + (1 if self.dim <= 25 else 0)
                    for _ in range(n_steps):
                        if evals >= self.budget:
                            break

                        # Gaussian perturbation scaled by domain span.
                        step = np.random.normal(0.0, local_sigma, size=self.dim) * span
                        # Occasional sparse heavy-tail jump for escaping flat valleys.
                        if np.random.rand() < 0.2:
                            k = max(1, self.dim // 8)
                            idx = np.random.choice(self.dim, size=k, replace=False)
                            step[idx] += (
                                np.random.standard_cauchy(size=k) * 0.01 * span[idx]
                            )

                        candidate = clamp(anchor + step)
                        cy, ok = evaluate(candidate)
                        if not ok:
                            break

                        refresh_archive(candidate, cy)
                        # If this move improves the anchor-local objective, walk from there.
                        if cy < ay:
                            anchor = candidate
                            ay = cy

                        # Very cheap directional probing:
                        # try a half-step opposite direction from the anchor.
                        if evals < self.budget and np.random.rand() < 0.35:
                            probe = clamp(anchor - 0.5 * step)
                            py, okp = evaluate(probe)
                            if not okp:
                                break
                            refresh_archive(probe, py)
                            if py < ay:
                                anchor = probe
                                ay = py

            # Stall tracking and restart policy.
            if best_y < prev_best - 1e-12:
                prev_best = best_y
                no_improve_iters = 0
            else:
                no_improve_iters += 1

            # Soft restart after prolonged stall: keep elites, refresh others.
            stall_limit = 4 + self.dim // 8
            if no_improve_iters >= stall_limit and evals < self.budget:
                no_improve_iters = 0

                # Re-seed roughly half the population.
                keep = max(2, n // 2)
                keep_idx = np.argsort(pop_y)[:keep]
                new_x = [pop_x[j].copy() for j in keep_idx]
                new_y = [pop_y[j] for j in keep_idx]

                while len(new_x) < n and evals < self.budget:
                    # Mix archive-centered and global samples.
                    if np.random.rand() < 0.6 and len(elite_x) > 0:
                        base = elite_x[np.random.randint(len(elite_x))]
                        x = clamp(
                            base + np.random.normal(0.0, 0.25, size=self.dim) * span
                        )
                    else:
                        x = np.random.uniform(lower, upper, size=self.dim)

                    y, ok = evaluate(x)
                    if not ok:
                        break
                    new_x.append(x)
                    new_y.append(y)
                    refresh_archive(x, y)

                pop_x = np.asarray(new_x, dtype=float)
                pop_y = np.asarray(new_y, dtype=float)
        return best_x, best_y
