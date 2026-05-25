import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+5) evolution strategy with isotropic Gaussian mutation and
#   step‑size adaptation via the 1/5th success rule.
# Search state: A single parent solution (vector) and its fitness, plus a
#   global step‑size that is adapted each generation.
# Candidate generation: 5 offspring are created by adding Gaussian noise
#   scaled by the current step‑size. Noise is independently drawn per
#   coordinate.
# Selection and replacement: Plus selection – the best among parent and
#   offspring becomes the new parent. (If an offspring matches the parent
#   fitness, the parent is retained.)
# Adaptation: The step‑size is updated using the 1/5th rule: if the fraction
#   of offspring that are strictly better than the parent exceeds 0.2,
#   step‑size is multiplied by exp(1/3); if below 0.2, by exp(-1/4);
#   if exactly 0.2, unchanged. The step‑size is clamped to [1e-12, 1e10].
# Exploration mechanisms: Random initialisation, mutation with an adaptive
#   step‑size that can grow or shrink.
# Exploitation mechanisms: Retention of the best solution found so far;
#   selection pressure from plus selection.
# Boundary handling: Offspring coordinates that violate the bounds are
#   reflected back into the domain.
# Budget strategy: The total number of function evaluations (including the
#   initial one) is strictly counted and never exceeds the provided budget.
# Closest known influences: Classic evolution strategies (Rechenberg,
#   Schwefel) with isotropic mutations and the 1/5th rule.
# Novelty or unusual aspects: None – a straightforward, textbook
#   implementation designed for clarity and robustness across dimensions.
# Failure modes: On highly multimodal or ill‑conditioned problems the
#   algorithm may converge prematurely if the step‑size collapses. The
#   1/5th rule can also struggle when the optimum lies near the boundary.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ---------- Read bounds ----------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lower = np.asarray(b.lb, dtype=float)
            upper = np.asarray(b.ub, dtype=float)
        else:
            raise ValueError("Cannot find bounds from func")

        # ---------- Initialisation ----------
        rng = np.random.default_rng()  # harness sets seed, so this is fine
        parent = lower + rng.random(self.dim) * (upper - lower)
        parent_fitness = func(parent)
        evals = 1

        best_x = parent.copy()
        best_y = parent_fitness

        step_size = 0.2 * (upper - lower).mean()  # initial step size

        # population size (offspring per generation)
        lam = 5
        # 1/5th rule parameters
        target_success = 0.2
        success_factor_increase = np.exp(1.0 / 3.0)
        success_factor_decrease = np.exp(-1.0 / 4.0)

        # ---------- Main loop ----------
        while evals < self.budget:
            # remaining evaluations cannot accommodate a full generation?
            # We generate lam offspring, but if we cannot evaluate all,
            # we reduce lam accordingly.
            gen_size = min(lam, self.budget - evals)
            if gen_size <= 0:
                break

            offspring = rng.normal(0, step_size, size=(gen_size, self.dim))
            offspring += parent   # each row is candidate
            # ---------- Boundary reflection ----------
            # reflect any coordinate that lies outside [lower, upper]
            # two reflections suffice: reflect across the violated bound
            lo = lower[None, :]
            hi = upper[None, :]
            # first reflection
            reflection = 2 * np.where(offspring < lo, lo, hi) - offspring
            offspring = np.where((offspring < lo) | (offspring > hi), reflection, offspring)
            # second reflection (guarantees inside after two)
            reflection = 2 * np.where(offspring < lo, lo, hi) - offspring
            offspring = np.where((offspring < lo) | (offspring > hi), reflection, offspring)

            # ---------- Evaluation ----------
            fits = np.full(gen_size, np.inf)
            for i in range(gen_size):
                fits[i] = func(offspring[i])
                evals += 1
                # Update global best
                if fits[i] < best_y:
                    best_y = fits[i]
                    best_x = offspring[i].copy()
                # Early stop if budget exhausted (though we pre-scheduled gen_size)
                if evals >= self.budget:
                    break

            # ---------- Selection ----------
            # (1+5) selection: best among parent and offspring
            all_candidates = np.concatenate(([parent], offspring))
            all_fits = np.concatenate(([parent_fitness], fits))
            best_idx = np.argmin(all_fits)
            new_parent = all_candidates[best_idx]
            new_parent_fitness = all_fits[best_idx]

            # ---------- Step‑size adaptation ----------
            success_count = np.sum(fits < parent_fitness)
            success_rate = success_count / gen_size
            if success_rate > target_success:
                step_size *= success_factor_increase
            elif success_rate < target_success:
                step_size *= success_factor_decrease
            # else unchanged

            # clamp step‑size
            step_size = np.clip(step_size, 1e-12, 1e10)

            parent = new_parent.copy()
            parent_fitness = new_parent_fitness

        return best_x, best_y
