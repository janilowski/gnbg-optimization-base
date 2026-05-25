import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer using a
# (1+λ)-style evolutionary strategy with coordinate-wise step-size adaptation.
# It maintains a single incumbent solution and generates multiple offspring each
# iteration, selecting the best offspring to replace the incumbent.
# Search state: Incumbent x, its objective value f(x), current global step size
# sigma, number of evaluations used, and a history of recent improvements for
# mild adaptation heuristics.
# Candidate generation: Each generation samples λ offspring around the incumbent:
# x_child = x + sigma * N(0, I). If a reflection-style bound hit is detected,
# the coordinate is nudged back inside bounds while preserving direction.
# Selection and replacement: Among valid (clipped) offspring, the best objective
# value replaces the incumbent if it improves (elitist selection).
# Adaptation: Uses a success-based 1/5-like rule and also scales sigma using
# how many offspring improved the incumbent in the current generation. sigma
# is decreased on failure and increased on consistent success.
# Exploration mechanisms: Gaussian perturbations controlled by sigma; occasional
# larger perturbations ("bursts") when progress stalls to escape local minima.
# Exploitation mechanisms: When improvements occur, sigma is reduced to focus
# the search near the incumbent and exploitation continues around the best point.
# Boundary handling: Reads variable bounds from func.lower/upper or func.bounds.lb/ub.
# Offspring are clipped to bounds (and a small corrective nudge is applied to
# reduce sticking at edges).
# Budget strategy: Converts the evaluation budget into a fixed number of
# generations plus a remainder. Each generation evaluates exactly λ candidates,
# and remaining evaluations are used for one final partial generation. Never
# exceeds the provided budget.
# Closest known influences: Similar in spirit to (1+λ)-ES and CMA-free
# coordinate-free step adaptation, including a success-rate heuristic and elitism.
# Novelty or unusual aspects: Uses a small deterministic "burst schedule" tied
# to progress (no improvement counter) and a simple edge-nudging mechanism after
# clipping to keep the search from collapsing onto boundaries.
# Failure modes: Can stagnate on rugged landscapes or flat regions if sigma
# becomes too small too quickly; may require adequate budget for high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")

    def __call__(self, func):
        d = self.dim
        lb, ub = self._read_bounds(func)

        # Ensure proper shapes/dtypes
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != d or ub.size != d:
            raise ValueError("Bounds dimensionality mismatch with dim")

        # If any bounds are invalid, attempt to correct minor numerical issues
        # (assumes lb <= ub).
        swapped = lb > ub
        if np.any(swapped):
            tmp = lb.copy()
            lb[swapped] = ub[swapped]
            ub[swapped] = tmp[swapped]

        span = ub - lb
        # Prevent zero-span dimensions from causing zero perturbations everywhere.
        span_safe = np.where(span > 0, span, 1.0)

        # Initial point: center with slight random jitter to break ties.
        x = lb + 0.5 * span_safe
        x = self._clip_nudge(x, lb, ub)
        x = x + 0.01 * span_safe * np.random.randn(d)
        x = self._clip_nudge(x, lb, ub)

        evals = 0
        best_y = float(func(x))
        evals += 1

        # Choose offspring count based on dimension and budget.
        # Keep λ moderate so we can adapt within budget.
        # Typical values: 4..20 depending on d/budget.
        lam = int(np.clip(4 + d // 2, 4, 32))
        lam = min(lam, max(1, self.budget - 1))  # must allow at least one eval/update

        # Step size initialization: fraction of typical scale.
        sigma = 0.25 * float(np.mean(span_safe))
        if sigma <= 0:
            sigma = 1.0

        # Progress tracking for mild adaptation and bursts.
        no_improve = 0
        history_impr = 0  # count of improvements in recent steps (binary)

        # Compute number of full generations with λ evaluations each.
        # We already used 1 evaluation for incumbent.
        remaining = self.budget - evals
        if remaining <= 0:
            return x, best_y

        full_gens = remaining // lam
        rem = remaining % lam

        gens = full_gens + (1 if rem > 0 else 0)

        # Main loop: elitist (1+λ)-ES with adaptive sigma.
        for gen_idx in range(gens):
            # Determine how many offspring we can evaluate this generation.
            cur_lam = lam if gen_idx < full_gens else rem
            if cur_lam <= 0:
                break

            # Adaptive "burst" when stuck: occasionally enlarge sigma.
            # Deterministic schedule depends on dimension and generation index.
            burst = False
            if no_improve >= max(6, 2 * int(np.sqrt(d))) and (gen_idx % 3 == 2):
                burst = True
                burst_factor = 2.0 + 0.5 * np.random.rand()
            else:
                burst_factor = 1.0

            sigma_eff = sigma * burst_factor

            # Sample gaussian perturbations around incumbent.
            # Offspring matrix: (cur_lam, d)
            Z = np.random.randn(cur_lam, d)
            Xcand = x[None, :] + sigma_eff * Z

            # Boundary handling: clip and nudge to avoid exact sticking on bounds.
            Xcand = np.apply_along_axis(lambda row: self._clip_nudge(row, lb, ub), 1, Xcand)

            # Evaluate offspring
            ys = np.empty(cur_lam, dtype=float)
            for i in range(cur_lam):
                ys[i] = float(func(Xcand[i]))
            evals += cur_lam

            # Select best
            idx = int(np.argmin(ys))
            y_best_child = float(ys[idx])
            x_best_child = Xcand[idx]

            # Count number of improvements this generation
            improved_mask = ys < best_y
            success_count = int(np.sum(improved_mask))

            # Replacement (elitist): only if strictly better
            if y_best_child < best_y:
                # Update incumbent
                improvement = best_y - y_best_child
                x, best_y = x_best_child, y_best_child
                no_improve = 0
                history_impr = min(10, history_impr + 1)
                # Exploitation: shrink sigma on improvement
                # Use improvement success magnitude lightly.
                shrink = 0.84 ** (1.0 + 0.5 * (success_count / max(1, cur_lam)))
                sigma *= shrink
            else:
                no_improve += 1
                history_impr = max(0, history_impr - 1)
                # Exploration: increase sigma slightly on failure, strongly if no offspring improved
                if success_count == 0:
                    sigma *= 1.18
                else:
                    sigma *= 1.06

            # Additional 1/5-ish success rate adjustment
            success_rate = success_count / max(1, cur_lam)
            if success_rate > 0.2:
                sigma *= 1.03
            elif success_rate < 0.1:
                sigma *= 0.98

            # Hard cap to prevent sigma exploding relative to bounds span.
            # Also ensure sigma doesn't vanish.
            mean_span = float(np.mean(span_safe))
            sigma = float(np.clip(sigma, 1e-12 * mean_span, 5.0 * mean_span + 1e-12))

            # Safety: never exceed budget (should already be ensured)
            if evals >= self.budget:
                break

        return x, best_y

    @staticmethod
    def _read_bounds(func):
        # Priority:
        # 1) func.lower/func.upper
        # 2) func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return func.lower, func.upper
        if hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            return func.bounds.lb, func.bounds.ub
        raise AttributeError(
            "Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
        )

    @staticmethod
    def _clip_nudge(x, lb, ub):
        # Clip into feasible region
        xc = np.minimum(np.maximum(x, lb), ub)
        # Nudge slightly away from exact boundaries when clipped, to reduce getting
        # stuck on edges (using deterministic tiny random perturbation).
        # This is robust even if bounds are tight.
        eps = 1e-12
        at_lb = xc <= lb + eps
        at_ub = xc >= ub - eps
        if np.any(at_lb) or np.any(at_ub):
            span = np.where(ub > lb, ub - lb, 1.0)
            # Move inward by a tiny fraction of span plus a tiny random term.
            nudge = 1e-9 * span * np.random.randn(xc.size)
            xc = np.where(at_lb, lb + eps * span + np.abs(nudge), xc)
            xc = np.where(at_ub, ub - eps * span - np.abs(nudge), xc)
            xc = np.minimum(np.maximum(xc, lb), ub)
        return xc
