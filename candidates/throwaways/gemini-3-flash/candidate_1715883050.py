# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous sampling algorithm generating candidate solutions around the rank-weighted center of gravity of elite incumbents.
# Search state: Stores an archive of the top K elite candidate solutions, their exact objective values, and global optimum.
# Candidate generation: Generates offspring batches via Gaussian sampling around the rank-weighted elite centroid and perturbations around the absolute best.
# Selection and replacement: Merges newly evaluated batch into the archive and truncates to preserve the top K elite individuals.
# Adaptation: Coordinate standard deviation dynamically tracks the coordinate span across the elite archive.
# Exploration mechanisms: Additive minimum variance lower bounds ensure continuous stochastic exploration across all dimensions.
# Exploitation mechanisms: Rank-weighted center of gravity and direct sampling around the best incumbent concentrate search in elite basins.
# Boundary handling: All sampled candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates sample batches sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Estimation of Distribution Algorithms / Covariance Sampling (Hansen).
# Novelty or unusual aspects: Pre-computes exact linear ranking weights for rapid centroid estimation without full covariance matrix inversions.
# Failure modes: Disregards variable correlations, leading to potential inefficiency along highly non-separable diagonal valleys.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.k_elites = max(5, min(10, self.dim))
        self.batch_size = int(min(self.budget // 6, max(15, 2 * self.dim)))
        if self.batch_size > 50:
            self.batch_size = 50

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

        archive_x = np.random.uniform(lb, ub, size=(self.k_elites, self.dim))
        archive_y = np.full(self.k_elites, float("inf"))

        for i in range(self.k_elites):
            if self.eval_count >= self.budget:
                break
            y = float(func(archive_x[i]))
            self.eval_count += 1
            archive_y[i] = y
            if y < best_y:
                best_y = y
                best_x = archive_x[i].copy()

        # Sort archive
        sorted_idx = np.argsort(archive_y)
        archive_x = archive_x[sorted_idx]
        archive_y = archive_y[sorted_idx]

        min_std = 1e-6 * domain_range

        # Rank weights
        ranks = np.arange(self.k_elites)
        weights = 2.0 * (self.k_elites - ranks) / (self.k_elites * (self.k_elites + 1.0))

        while self.eval_count < self.budget:
            # Centroid
            centroid = np.sum(archive_x * weights[:, np.newaxis], axis=0)
            
            # Span
            span = np.max(archive_x, axis=0) - np.min(archive_x, axis=0)
            std = np.maximum(span, min_std)

            batch_x = np.zeros((self.batch_size, self.dim))
            batch_y = np.full(self.batch_size, float("inf"))

            half = self.batch_size // 2

            for i in range(self.batch_size):
                if self.eval_count >= self.budget:
                    break

                if i < half:
                    # Sample around centroid
                    z = np.random.normal(0, 1, size=self.dim)
                    cand = centroid + z * std
                else:
                    # Sample around absolute best
                    z = np.random.normal(0, 1, size=self.dim)
                    cand = best_x + z * (0.5 * std)

                cand = np.clip(cand, lb, ub)
                y = float(func(cand))
                self.eval_count += 1
                batch_x[i] = cand
                batch_y[i] = y

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            if self.eval_count >= self.budget:
                break

            merged_x = np.vstack((archive_x, batch_x))
            merged_y = np.concatenate((archive_y, batch_y))

            sorted_idx = np.argsort(merged_y)[:self.k_elites]
            archive_x = merged_x[sorted_idx]
            archive_y = merged_y[sorted_idx]

            # Check for collapse
            if np.max(std / domain_range) < 1e-5:
                # Randomize bottom half of archive
                for i in range(self.k_elites // 2, self.k_elites):
                    if self.eval_count >= self.budget:
                        break
                    archive_x[i] = np.random.uniform(lb, ub, size=self.dim)
                    y = float(func(archive_x[i]))
                    self.eval_count += 1
                    archive_y[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = archive_x[i].copy()
                
                sorted_idx = np.argsort(archive_y)
                archive_x = archive_x[sorted_idx]
                archive_y = archive_y[sorted_idx]

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
