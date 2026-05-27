import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (μ+λ)-Evolution Strategy with self-adaptive step sizes and an elitist
#          selection mechanism. Designed for compactness and robust behaviour across
#          a wide range of continuous black‑box problems.
# Search state: A population of μ candidate solutions, each represented by a vector
#               x and an individual step size σ. The objective values are stored
#               alongside the candidates.
# Candidate generation: λ offspring are created by copying a randomly chosen parent
#                       (uniform among the μ best) and mutating both its step size
#                       and its position. The step size is updated via log‑normal
#                       mutation with learning rate τ=1/√(2·dim), and the position
#                       is perturbed with isotropic Gaussian noise of magnitude σ.
# Selection and replacement: The next population is formed by selecting the μ best
#                            individuals from the union of the current μ parents and
#                            the λ offspring (elitist (μ+λ) selection).
# Adaptation: Step sizes are self‑adapted: each offspring inherits its parent’s σ
#             and mutates it; successful σ values (those belonging to selected
#             individuals) propagate to future generations automatically.
# Exploration mechanisms: Large initial σ (0.2 of the box range) and log‑normal
#                         step size mutation allow wide exploration early on.
# Exploitation mechanisms: As the population converges, the step sizes shrink
#                          because poorly adapted σ are discarded by selection.
#                          The elitist replacement ensures that the best found
#                          solution is never lost.
# Boundary handling: All candidate positions are clamped to the search box.
# Budget strategy: The number of offspring per generation is reduced when the
#                  remaining budget is less than λ, so that the algorithm can
#                  use every remaining evaluation efficiently.
# Closest known influences: Classic Evolution Strategy with self‑adaptation
#                           (Schwefel 1995) and the (μ+λ) selection scheme.
# Novelty or unusual aspects: None; the implementation is a straightforward,
#                             textbook version for reliability and readability.
# Failure modes: On high‑dimensional (>100) or extremely rugged landscapes the
#                isotropic mutation may converge slowly; no restart mechanism is
#                included, so the algorithm can stagnate in local optima.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """
    (μ+λ)-ES with self‑adaptive isotropic step sizes.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

        # Population sizes (heuristic from CMA‑ES)
        self.lambda_ = max(4, 4 + int(3 * np.log(dim)))
        self.mu = self.lambda_ // 2

        # Learning rate for step‑size mutation
        self.tau = 1.0 / np.sqrt(2 * dim)

        # Seeded by the harness; we just use numpy's global generator
        self._rng = np.random.default_rng()

    def __call__(self, func):
        # ---------- read bounds ----------
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        # Ensure correct shape
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)
        lb = lb.ravel()
        ub = ub.ravel()
        box_range = ub - lb

        # ---------- initialisation ----------
        pop = []
        for _ in range(self.lambda_):
            x = lb + self._rng.uniform(0, 1, self.dim) * box_range
            sigma = 0.2 * box_range  # per‑dimension sigma kept as scalar mean
            # Use a single sigma per individual (mean range)
            sigma = np.mean(sigma)
            y = func(x)
            pop.append((x, sigma, y))
        # sort by objective value
        pop.sort(key=lambda ind: ind[2])

        evals = self.lambda_
        best_x, best_y = pop[0][0].copy(), pop[0][2]

        # ---------- main loop ----------
        while evals < self.budget:
            # Number of offspring for this generation
            remaining = self.budget - evals
            offspring_count = min(self.lambda_, remaining)

            # Generate offspring
            offspring = []
            for _ in range(offspring_count):
                # Select a parent uniformly from the μ best
                parent_idx = self._rng.integers(self.mu)
                parent_x, parent_sigma, _ = pop[parent_idx]

                # Mutate step size
                child_sigma = parent_sigma * np.exp(
                    self.tau * self._rng.normal()
                )

                # Mutate position
                child_x = parent_x + child_sigma * self._rng.normal(size=self.dim)

                # Clamp to box
                child_x = np.clip(child_x, lb, ub)

                # Evaluate
                child_y = func(child_x)
                evals += 1

                offspring.append((child_x, child_sigma, child_y))

                # Update best
                if child_y < best_y:
                    best_y = child_y
                    best_x = child_x.copy()

            # (μ+λ) selection: combine parents and offspring, keep best μ
            combined = pop + offspring
            combined.sort(key=lambda ind: ind[2])
            pop = combined[:self.mu]

            # Safety check – should not happen but guard against budget overshoot
            if evals >= self.budget:
                break

        return best_x, best_y
