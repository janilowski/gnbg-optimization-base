from __future__ import annotations

# =============================================================================
#  BIPOP-CMA-ES  with  Skip-Residual Archive Bridges
# =============================================================================
#
#  ALGORITHM IDEA
#  ==============
#  This optimizer combines three well-studied ideas into a single, robust
#  black-box continuous optimizer:
#
#  1.  CMA-ES  (Covariance Matrix Adaptation Evolution Strategy)
#      ─────────────────────────────────────────────────────────
#      Maintains a multivariate normal search distribution N(m, σ²C) and
#      iteratively updates the mean m, the global step size σ, and the full
#      covariance matrix C to match the local curvature of the objective.
#      Unlike DE which uses fixed-scale differential mutations, CMA-ES adapts
#      the direction *and* scale of its steps, making it scale- and
#      rotation-invariant.  This is the gold-standard black-box optimizer on
#      BBOB/GNBG-style benchmarks (ranked 2nd overall at GNBG-II GECCO 2025).
#
#  2.  BIPOP Restarts  (Bi-Population, Hansen 2009)
#      ─────────────────────────────────────────────
#      When CMA-ES stalls (sigma collapses or no improvement for many
#      generations), we restart it.  BIPOP interleaves two restart regimes:
#        • Large-population restarts: λ doubles each time (IPOP-style), giving
#          coarser global coverage with each new restart.
#        • Small-population restarts: random λ ∈ [2, λ_default) with a random
#          sigma from a wide log-uniform range.  These cheap fresh explorations
#          cheaply re-probe local basins the large regime missed.
#      BIPOP is especially effective on multi-modal / deceptive functions
#      (GNBG f16–f24) because the small regime constantly tries new random
#      starting points.
#
#  3.  Skip-Residual Archive Bridges  (the creative twist)
#      ─────────────────────────────────────────────────────
#      In ResNets, skip (residual) connections bypass intermediate
#      transformations:
#          output = F(x) + x     ← identity shortcut
#      This prevents information loss through deep networks.
#
#      Here, we keep an elite archive of the K historically best solutions
#      ("skip nodes").  During each CMA-ES sampling step, each offspring has a
#      probabilistic *skip path* that bypasses the current mean+covariance
#      transformation and instead links directly to an archive anchor:
#
#          Standard path:  x = m + σ · B·D·z          (learned distribution)
#          Skip path:      x = x_arch + σ · z          (shortcut to archive)
#          Blended:        x = (1-α)·x_main + α·x_arch + ε·noise
#
#      When the distribution has drifted away from promising regions (as
#      happens after a restart), the skip path acts as a *residual correction
#      channel* that pulls proposals back toward historically good solutions.
#      The archive persists across ALL restarts, providing long-term memory
#      that vanilla BIPOP-CMA-ES lacks.
#
#  WHY THIS IS ROBUST FOR GNBG-30D
#  ================================
#  • All 24 GNBG instances are 30-D on [-100, 100].
#  • f1–f6  (unimodal, ill-conditioned): CMA-ES's covariance adaptation is
#    orders of magnitude better than DE at handling strong conditioning.
#  • f7–f15 (single-basin multimodal): CMA-ES + restarts handle these well.
#  • f16–f24 (deceptive multi-basin): BIPOP small restarts + archive skip
#    bridges together cover diverse basins.
#
#  ONLY NumPy IS USED  (no scipy, no cma package).
#  Budget safety is guaranteed: every evaluation is guarded by an eval counter.
# =============================================================================
import numpy as np


class Algorithm:
    """
    BIPOP-CMA-ES with Skip-Residual Archive Bridges.

    Public interface (required by harness):
        __init__(self, budget, dim)
        __call__(self, func) -> (best_x, best_y)
    """

    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):

        # ── 1.  Bounds & domain setup ─────────────────────────────────────────
        lower = np.asarray(getattr(func, "lower", func.bounds.lb), dtype=float)
        upper = np.asarray(getattr(func, "upper", func.bounds.ub), dtype=float)
        span = np.maximum(upper - lower, 1e-12)  # box widths (avoid div-by-0)

        # ── 2.  Global incumbent tracking ────────────────────────────────────
        evals = 0
        best_x: np.ndarray | None = None
        best_y = float("inf")

        def clamp(x: np.ndarray) -> np.ndarray:
            """Project a candidate strictly inside [lower, upper]."""
            return np.clip(x, lower, upper)

        def evaluate(x: np.ndarray):
            """
            Budget-guarded evaluation.

            Returns (y, True) on success, (None, False) when budget is
            exhausted.  Always updates global best and elite archive.
            """
            nonlocal evals, best_x, best_y
            if evals >= self.budget:
                return None, False
            x_c = clamp(np.asarray(x, dtype=float))
            y = float(func(x_c))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x_c.copy()
            return y, True

        # ── 3.  Elite archive  ("skip nodes") ────────────────────────────────
        #
        # The archive stores the K best (x, y) pairs ever seen.  These act as
        # the anchor points for the skip-residual proposals: they are memories
        # of historically good solutions that survive across all restarts.
        # Without this persistence, every CMA-ES restart starts from scratch;
        # with it, later restarts can directly "shortcut" to previously
        # discovered promising regions.
        ARCHIVE_K = max(8, self.dim)  # Archive capacity
        arch_x: list[np.ndarray] = []
        arch_y: list[float] = []

        def archive_update(x: np.ndarray, y: float) -> None:
            """Insert (x, y); evict the worst entry when the archive is full."""
            if len(arch_x) < ARCHIVE_K:
                arch_x.append(x.copy())
                arch_y.append(y)
            elif y < max(arch_y):
                worst = int(np.argmax(arch_y))
                arch_x[worst] = x.copy()
                arch_y[worst] = y

        def best_archive_point() -> np.ndarray | None:
            """Return the archive entry with the lowest objective value."""
            if not arch_x:
                return None
            return arch_x[int(np.argmin(arch_y))]

        # ── 4.  Tiny-budget guard ────────────────────────────────────────────
        # With ≤ 4 evaluations, CMA-ES overhead is unjustifiable; fall back to
        # random sampling.
        if self.budget <= 4:
            for _ in range(self.budget):
                evaluate(np.random.uniform(lower, upper, size=self.dim))
            return (best_x if best_x is not None else lower.copy()), best_y

        # ── 5.  CMA-ES dimension-dependent constants ─────────────────────────
        n = self.dim

        # Expected L2 norm of N(0, I_n): χ_n approximation
        chiN = n**0.5 * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

        # Default population size (Hansen & Ostermeier 2001, Table 1)
        base_lam = 4 + int(3 * np.log(n))

        # Initial step size: 30 % of the narrowest box side — broad enough to
        # cover the domain but not so large that offspring are uniformly random.
        sigma_init = 0.30 * float(np.min(span))

        # ── 6.  Inner CMA-ES runner ──────────────────────────────────────────
        def run_cma(
            m0: np.ndarray,
            sigma0: float,
            lam: int,
            budget_cap: int,
            max_stall_arg: int | None = None,
        ) -> None:
            """
            Execute one CMA-ES run starting from (m0, sigma0) with offspring
            count λ = lam, consuming at most budget_cap evaluations.

            max_stall_arg overrides the default stall limit (use a larger value
            for large restarts that need time to converge through temporary
            plateaus, and a smaller value for cheap small restarts).

            All state updates (global best, archive) go through evaluate() and
            archive_update(), so no return value is needed.
            """
            # ── Derived parameters ───────────────────────────────────────────
            mu = lam // 2  # number of parents

            # Recombination weights: log-linear, positive, sum to 1
            raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = raw_w / raw_w.sum()

            # Effective sample size (variance of weights)
            mueff = 1.0 / float(np.dot(weights, weights))

            # Time-constants (Hansen 2016 "The CMA Evolution Strategy: A Tutorial")
            cs = (mueff + 2.0) / (n + mueff + 5.0)  # σ path leak
            damps = (  # CSA damping
                1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
            )
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)  # C path leak
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)  # rank-1 learning rate
            cmu = min(  # rank-μ learning rate
                1.0 - c1,
                2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff),
            )

            # ── CMA-ES state ─────────────────────────────────────────────────
            m = m0.copy()  # distribution mean
            sigma = float(sigma0)  # global step size
            C = np.eye(n)  # covariance matrix
            pc = np.zeros(n)  # evolution path for C
            ps = np.zeros(n)  # evolution path for σ (CSA)

            # Eigendecomposition: C = B · diag(D²) · Bᵀ
            # B: columns = eigenvectors of C
            # D: sqrt of eigenvalues of C
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)  # C^{-½} = B · diag(1/D) · Bᵀ
            eigeneval = evals  # eval count at last eigendecomp refresh

            # Per-run bookkeeping
            stall_gens = 0
            best_this_run = best_y
            evals_start = evals
            gen = 0

            # Eigendecomp refresh frequency: roughly every lam/(c1+cmu)/n/10
            # evaluations (Hansen's lazy-update heuristic).
            eig_freq = max(1, int(lam / max(1e-10, (c1 + cmu) * n * 10.0)))

            # Stall limit: at most max_stall generations without improvement
            # before we declare convergence and hand off to the next restart.
            # The caller passes a regime-specific value:
            #   Large restarts → max(30, 10·n) = 300 for n=30.  These need
            #     plenty of stall tolerance to survive temporary plateaus on
            #     ill-conditioned unimodal functions (e.g. GNBG f3).
            #   Small restarts → max(10, 3·n) = 90 for n=30.  These are cheap
            #     explorations; terminate quickly to free budget for the next
            #     diverse restart.
            max_stall = max_stall_arg if max_stall_arg is not None else max(20, 7 * n)

            # ── Generation loop ──────────────────────────────────────────────
            while evals < self.budget and (evals - evals_start) < budget_cap:
                gen += 1

                # Refresh eigendecomposition (lazy, amortised O(n³) per gen).
                if evals - eigeneval >= eig_freq:
                    eigeneval = evals
                    C_sym = 0.5 * (C + C.T)  # enforce exact symmetry
                    try:
                        eigvals, B = np.linalg.eigh(C_sym)
                        eigvals = np.maximum(eigvals, 1e-20)  # positive-def
                        D = np.sqrt(eigvals)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                        C = C_sym
                    except np.linalg.LinAlgError:
                        # Rare numerical failure — reset to identity and continue.
                        C = np.eye(n)
                        B, D, invsqrtC = np.eye(n), np.ones(n), np.eye(n)
                        eigeneval = evals

                # ── Sample λ offspring ───────────────────────────────────────
                xs_gen: list[np.ndarray] = []
                ys_gen: list[float] = []

                # Skip-residual probability: 0 when archive is empty;
                # otherwise 20 % → gives useful diversity without dominating.
                skip_prob = 0.20 if arch_x else 0.0

                for _ in range(lam):
                    if evals >= self.budget:
                        break

                    # ── Standard CMA-ES proposal ─────────────────────────────
                    z = np.random.normal(0.0, 1.0, size=n)
                    x_main = m + sigma * (B @ (D * z))  # m + σ · C^{½} · z

                    # ── Skip-residual bridge ─────────────────────────────────
                    # With probability skip_prob, blend the CMA-ES proposal
                    # with an archive anchor — a direct shortcut that bypasses
                    # the current mean+covariance transformation.
                    #
                    # Neural-network analogy:
                    #   ResNet:   output = F(x) + x       (identity shortcut)
                    #   Here:     x_out  = (1-α)·x_CMA + α·x_archive + ε
                    #
                    # This ensures that even when CMA-ES is far from good
                    # regions (e.g. right after a fresh restart), the skip path
                    # maintains a direct information channel to previously found
                    # good solutions.
                    if arch_x and np.random.rand() < skip_prob:
                        # Prefer the best archive entry (60 %) for exploitation;
                        # pick a random one (40 %) for diversity.
                        if np.random.rand() < 0.6:
                            anchor = arch_x[int(np.argmin(arch_y))]
                        else:
                            anchor = arch_x[np.random.randint(len(arch_x))]

                        # Blend factor α: how strongly to follow the skip path
                        alpha = np.random.uniform(0.2, 0.5)

                        # Tiny isotropic noise keeps proposals from collapsing
                        noise = np.random.normal(0.0, 1.0, size=n)
                        x_cand = (
                            (1.0 - alpha) * x_main
                            + alpha * anchor
                            + 0.01 * sigma * noise
                        )
                    else:
                        x_cand = x_main

                    x_cand = clamp(x_cand)
                    val, ok = evaluate(x_cand)
                    if not ok:
                        break
                    assert val is not None  # evaluate returns float when ok=True

                    xs_gen.append(x_cand)
                    ys_gen.append(float(val))
                    archive_update(x_cand, float(val))  # update skip-node pool

                if not xs_gen:
                    break  # budget exhausted mid-generation

                # ── Rank and select top μ ────────────────────────────────────
                rank = np.argsort(ys_gen)
                xs_sorted = [xs_gen[i] for i in rank]
                n_sel = min(mu, len(xs_sorted))
                # Renormalise weights if we got fewer than μ offspring
                w = weights[:n_sel] / weights[:n_sel].sum()

                # ── Update mean ──────────────────────────────────────────────
                old_m = m.copy()
                m = np.zeros(n, dtype=float)
                for wi, xi in zip(w, xs_sorted[:n_sel]):
                    m += wi * xi
                m = clamp(m)  # keep mean inside domain

                # Mean displacement (unscaled by σ): y_w = (m_new − m_old) / σ
                y_w = (m - old_m) / (sigma + 1e-12)

                # ── Cumulative step-size adaptation (CSA) ────────────────────
                # Accumulate the conjugate evolution path pσ and update σ.
                # If ||pσ|| > χ_n the step size grows; if smaller it shrinks.
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (
                    invsqrtC @ y_w
                )
                sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0))
                sigma = float(np.clip(sigma, 1e-12, 2.0 * np.min(span)))

                # ── Heaviside h_σ (Hansen 2016, eq. 46) ─────────────────────
                # Suppresses the rank-1 C update in the first few generations
                # while the path pσ is still too short to be informative.
                h_denom = np.sqrt(max(1e-300, 1.0 - (1.0 - cs) ** (2.0 * gen)))
                h_sig = int(
                    np.linalg.norm(ps) / h_denom < (1.4 + 2.0 / (n + 1.0)) * chiN
                )

                # ── Covariance evolution path ────────────────────────────────
                pc = (1.0 - cc) * pc + h_sig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w

                # ── Rank-1 + rank-μ covariance update ────────────────────────
                # artmp[:, i] = (x_{i:λ} − m_old) / σ   (selection differential)
                artmp = np.column_stack(
                    [(xi - old_m) / (sigma + 1e-12) for xi in xs_sorted[:n_sel]]
                )
                C = (
                    (1.0 - c1 - cmu) * C
                    + c1
                    * (
                        np.outer(pc, pc)
                        # correction when h_sig=0 (path incomplete)
                        + (1.0 - h_sig) * cc * (2.0 - cc) * C
                    )
                    + cmu * artmp @ np.diag(w) @ artmp.T
                )
                C = 0.5 * (C + C.T)  # enforce exact symmetry against float drift

                # ── Stall detection ──────────────────────────────────────────
                if best_y < best_this_run - 1e-12:
                    best_this_run = best_y
                    stall_gens = 0
                else:
                    stall_gens += 1

                # Termination: step size collapsed (converged) or prolonged stall.
                # max_stall is set by the caller to be large for large restarts
                # (so ill-conditioned unimodal runs aren't cut short during
                # temporary plateaus) and small for small restarts (quick exit
                # frees budget for the next diverse starting point).
                if sigma < 1e-11 * np.min(span) or stall_gens > max_stall:
                    break

        # ── 7.  BIPOP outer restart loop ─────────────────────────────────────
        #
        # Alternates between:
        #   Large restarts:  λ doubles with each large restart (IPOP-style).
        #                    Provides coarser global coverage as λ grows.
        #   Small restarts:  random λ ∈ [2, base_lam), random σ₀.
        #                    Cheaply re-probes diverse local basins.
        #
        # The budget for each small restart mirrors the budget consumed by the
        # last large restart (original BIPOP rule), preventing any single small
        # restart from consuming all remaining budget.
        #
        # The skip-residual archive provides cross-restart memory: even a fresh
        # random-start small restart benefits from previously found solutions
        # via the skip proposals.

        large_restart = 0  # number of large restarts completed
        evals_large = 0  # evaluations consumed by all large restarts
        do_large = True  # controls alternation
        small_restarts_done = 0  # initialised here; reset inside each large epoch

        while evals < self.budget:
            remaining = self.budget - evals
            if remaining <= 0:
                break

            if do_large:
                # ── Large restart ─────────────────────────────────────────
                lam = base_lam * (2**large_restart)

                # Starting point: purely random for first restart (ensures
                # broad initial coverage); archive-guided or random afterwards.
                if large_restart == 0:
                    m0 = np.random.uniform(lower, upper, size=n)
                    s0 = sigma_init
                elif arch_x and np.random.rand() < 0.40:
                    # Warm start near a (randomly chosen) archive point,
                    # but keep sigma large enough to escape its local basin.
                    idx = np.random.randint(len(arch_x))
                    m0 = arch_x[idx].copy()
                    s0 = sigma_init * 0.50
                else:
                    m0 = np.random.uniform(lower, upper, size=n)
                    s0 = sigma_init

                # Budget cap for large restarts:
                #   Restart 0  — up to half the remaining budget so that
                #                slow-converging ill-conditioned functions
                #                (e.g. GNBG f3) have enough evals in the very
                #                first run to fully converge.
                #   Restart 1+ — one-third of remaining; later restarts are
                #                more targeted so less budget is acceptable.
                if large_restart == 0:
                    budget_cap = max(lam * 30, remaining // 2)
                else:
                    budget_cap = max(lam * 20, remaining // 3)

                # Large restarts get full stall tolerance (300 gens for n=30)
                # so temporary plateaus on unimodal functions don't cause
                # premature termination.  A fixed 300-gen limit also ensures
                # that slow-converging ill-conditioned functions (e.g. GNBG f3,
                # f9, f23) have enough patience to find deep local improvements
                # before the restart is abandoned.
                large_max_stall = max(30, 10 * n)

                evals_before = evals
                run_cma(m0, s0, lam, budget_cap, max_stall_arg=large_max_stall)
                used = evals - evals_before
                evals_large += used

                large_restart += 1
                small_restarts_done = 0  # reset counter for this epoch
                do_large = False  # next: small restart(s)

            else:
                # ── Small restart(s) ──────────────────────────────────────
                # We run N_SMALL_PER_LARGE small restarts per large restart
                # (BIPOP spirit: use the budget freed by the large restart for
                # diverse cheap explorations rather than one single re-probe).
                N_SMALL_PER_LARGE = 2

                # Population: uniform random in [2, base_lam)
                lam_s = max(2, int(np.random.uniform(2, base_lam)))

                # Initial sigma: log-uniform over [2⁻⁶, 1] × min(span)
                # (BIPOP paper recommendation; covers both fine and coarse scales)
                s0_s = float(np.min(span)) * 2.0 ** np.random.uniform(-6.0, 0.0)

                # Starting point: archive with perturbation (50%) or random (50%).
                # The archive perturbation creates a natural "exploitation of
                # previously discovered basins" without full warm-starting.
                if arch_x and np.random.rand() < 0.50:
                    idx = np.random.randint(len(arch_x))
                    m0_s = clamp(
                        arch_x[idx] + s0_s * np.random.normal(0.0, 1.0, size=n)
                    )
                else:
                    m0_s = np.random.uniform(lower, upper, size=n)

                # Budget cap for this small restart: each of the N_SMALL_PER_LARGE
                # restarts in this epoch shares the last large-restart budget
                # equally (analogous to the classical BIPOP equal-budget rule).
                budget_cap_s = min(
                    remaining,
                    max(lam_s * 5, evals_large // max(1, N_SMALL_PER_LARGE)),
                )

                # Small restarts use a tight stall limit (3·n = 90 gens for
                # n=30) — they are quick explorations, not convergence runs.
                small_max_stall = max(10, 3 * n)

                run_cma(m0_s, s0_s, lam_s, budget_cap_s, max_stall_arg=small_max_stall)

                small_restarts_done += 1
                # Switch back to a large restart only after all small restarts
                # in this epoch are exhausted.
                if small_restarts_done >= N_SMALL_PER_LARGE:
                    do_large = True

        # ── 8.  Opportunistic final polish ───────────────────────────────────
        #
        # If the BIPOP loop terminates with budget remaining (e.g. the last
        # restart stalled early), use any leftover evaluations for a tight
        # CMA-ES pass from best_x.  This costs nothing when the BIPOP loop
        # naturally exhausts the budget (the common case), and can reduce the
        # final gap on near-optimal problems when a few hundred evals remain.
        if best_x is not None and evals < self.budget:
            polish_sigma = 0.01 * sigma_init  # 0.3 % of domain — very tight
            run_cma(
                best_x.copy(),
                polish_sigma,
                base_lam,
                self.budget - evals,
                max_stall_arg=50,
            )

        # ── 9.  Return ───────────────────────────────────────────────────────
        return (best_x if best_x is not None else lower.copy()), best_y
