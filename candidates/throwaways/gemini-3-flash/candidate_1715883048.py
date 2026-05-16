# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An Ant Colony Optimization algorithm for continuous domains (ACOR) maintaining a Gaussian kernel density archive.
# Search state: Stores a ranked archive of K elite candidate solutions, their exact objective values, and the global optimum.
# Candidate generation: Generates new solutions by probabilistically selecting an archive solution kernel and sampling Gaussian perturbations scaled by average coordinate distances.
# Selection and replacement: Merges newly evaluated solutions into the archive and truncates to retain the top K elite individuals.
# Adaptation: Kernel bandwidth (coordinate standard deviation) contracts naturally as archive members cluster in promising basins.
# Exploration mechanisms: Gaussian kernel density sampling across diverse archive members maintains multimodal domain exploration.
# Exploitation mechanisms: Archive selection weights follow a steep exponential ranking distribution favoring the top elite incumbents.
# Boundary handling: All sampled kernel candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates ant colony sample batches sequentially while strictly checking remaining evaluation budget.
# Closest known influences: Ant Colony Optimization for Continuous Domains ACOR (Socha & Dorigo).
# Novelty or unusual aspects: Pre-computes exact Gaussian ranking weights to ensure stable probabilistic selection across iterations.
# Failure modes: Kernel variance estimation can become computationally expensive for large archive sizes in high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.k_archive = int(min(self.budget // 10, max(15, 2 * self.dim)))
        if self.k_archive > 50:
            self.k_archive = 50
        self.m_ants = max(5, self.k_archive // 2)

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

        # Initialize archive
        archive_x = np.random.uniform(lb, ub, size=(self.k_archive, self.dim))
        archive_y = np.full(self.k_archive, float("inf"))

        for i in range(self.k_archive):
            if self.eval_count >= self.budget:
                break
            y = float(func(archive_x[i]))
            self.eval_count += 1
            archive_y[i] = y
            if y < best_y:
                best_y = y
                best_x = archive_x[i].copy()

        # Sort initial archive
        sorted_idx = np.argsort(archive_y)
        archive_x = archive_x[sorted_idx]
        archive_y = archive_y[sorted_idx]

        q = 0.1
        xi = 0.85
        min_std = 1e-6 * domain_range

        # Precompute weights and selection probabilities
        ranks = np.arange(self.k_archive)
        weights = np.exp(-(ranks ** 2) / (2.0 * (q ** 2) * (self.k_archive ** 2))) / (q * self.k_archive * math.sqrt(2.0 * math.pi) + 1e-12)
        probs = weights / np.sum(weights)

        while self.eval_count < self.budget:
            ants_x = np.zeros((self.m_ants, self.dim))
            ants_y = np.full(self.m_ants, float("inf"))

            for i in range(self.m_ants):
                if self.eval_count >= self.budget:
                    break

                # Select kernel l from archive
                l = np.random.choice(self.k_archive, p=probs)
                kernel_center = archive_x[l]

                # Compute std for each dimension
                diffs = np.abs(archive_x - kernel_center)
                std = xi * np.sum(diffs, axis=0) / max(1, self.k_archive - 1)
                std = np.maximum(std, min_std)

                step = np.random.normal(0, 1, size=self.dim) * std
                trial = np.clip(kernel_center + step, lb, ub)

                y = float(func(trial))
                self.eval_count += 1
                ants_x[i] = trial
                ants_y[i] = y

                if y < best_y:
                    best_y = y
                    best_x = trial.copy()

            if self.eval_count >= self.budget:
                break

            # Merge archive and ants
            merged_x = np.vstack((archive_x, ants_x))
            merged_y = np.concatenate((archive_y, ants_y))

            # Retain top K
            sorted_idx = np.argsort(merged_y)[:self.k_archive]
            archive_x = merged_x[sorted_idx]
            archive_y = merged_y[sorted_idx]

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
