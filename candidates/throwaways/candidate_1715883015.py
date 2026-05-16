# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Harmony Search algorithm featuring dynamic pitch adjustment rates and exponentially decaying search bandwidth.
# Search state: Retains a harmony memory archive of top candidate vectors, their objective values, and global best solution.
# Candidate generation: Generates new harmony vectors coordinate-by-coordinate by selecting from harmony memory, applying pitch adjustments, or random sampling.
# Selection and replacement: Replaces the worst member in harmony memory whenever a newly generated harmony vector achieves superior fitness.
# Adaptation: Pitch adjustment rate increases linearly while adjustment bandwidth exponentially decays over the course of the evaluation budget.
# Exploration mechanisms: Random coordinate sampling (1 - HMCR) and initial wide pitch bandwidths explore unvisited regions of the domain.
# Exploitation mechanisms: Selecting coordinates directly from successful harmony memory vectors concentrates search in elite sub-regions.
# Boundary handling: All pitch adjustments and random samples are explicitly clipped inside domain bounds.
# Budget strategy: Iterates single harmony vector evaluations sequentially until the exact budget ceiling is reached.
# Closest known influences: Harmony Search (Geem).
# Novelty or unusual aspects: Employs dynamic exponential bandwidth decay per coordinate axis to ensure stable convergence in continuous parameter spaces.
# Failure modes: Independent coordinate selection can struggle to follow highly diagonal or non-separable landscape valleys.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.hms = int(min(self.budget // 5, max(15, 2 * self.dim)))
        if self.hms > 50:
            self.hms = 50

    def __call__(self, func):
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        self.eval_count = 0
        domain_range = ub - lb

        best_x = None
        best_y = float("inf")

        # Initialize Harmony Memory
        hm = np.random.uniform(lb, ub, size=(self.hms, self.dim))
        hm_y = np.full(self.hms, float("inf"))

        for i in range(self.hms):
            if self.eval_count >= self.budget:
                break
            y = float(func(hm[i]))
            self.eval_count += 1
            hm_y[i] = y
            if y < best_y:
                best_y = y
                best_x = hm[i].copy()

        hmcr = 0.90
        par_start = 0.3
        par_end = 0.8
        bw_start = 0.1 * domain_range
        bw_end = 1e-5 * domain_range

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            par = par_start + (par_end - par_start) * progress
            bw = bw_start * ((bw_end / (bw_start + 1e-12)) ** progress)

            new_x = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < hmcr:
                    # Choose from HM
                    mem_idx = np.random.randint(self.hms)
                    new_x[j] = hm[mem_idx, j]

                    if np.random.rand() < par:
                        # Pitch adjustment
                        direction = np.random.choice([-1, 1])
                        new_x[j] += direction * np.random.rand() * bw[j]
                else:
                    # Random sampling
                    new_x[j] = np.random.uniform(lb[j], ub[j])

            new_x = np.clip(new_x, lb, ub)
            y = float(func(new_x))
            self.eval_count += 1

            worst_idx = np.argmax(hm_y)
            if y < hm_y[worst_idx]:
                hm[worst_idx] = new_x.copy()
                hm_y[worst_idx] = y
                if y < best_y:
                    best_y = y
                    best_x = new_x.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
