# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Surrogate-Assisted optimization algorithm employing an inverse-distance weighted k-Nearest Neighbors regression model to filter candidate samples.
# Search state: Stores an archive of historical evaluated points and their exact objective values, along with the global optimum.
# Candidate generation: Generates large batches of candidate points via stochastic sampling around the incumbent and uniform domain sampling.
# Selection and replacement: Evaluates the single candidate point predicted most promising by the k-NN surrogate against the true objective function.
# Adaptation: Surrogate model accuracy improves continuously as newly evaluated points expand the historical sample archive.
# Exploration mechanisms: Allocates half of the candidate screening pool to uniform random domain samples to discover unexplored regions.
# Exploitation mechanisms: Inverse-distance k-NN prediction accurately interpolates local objective topology around existing elite samples.
# Boundary handling: All generated candidate points are strictly clipped inside valid domain boundaries.
# Budget strategy: Evaluates exactly one true objective function call per surrogate filtering cycle until budget exhaustion.
# Closest known influences: Surrogate-Assisted Optimization / Kriging / k-NN Surrogates (Jin).
# Novelty or unusual aspects: Lightweight non-parametric surrogate filtering eliminating matrix inversion overhead common in Gaussian Processes.
# Failure modes: Distance metrics become less discriminative in extremely high dimensions (curse of dimensionality).
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0

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

        archive_x = []
        archive_y = []
        max_archive = 400
        init_samples = max(10, min(self.budget // 10, 20))
        k_neigh = 3

        # Initial random sampling
        for _ in range(init_samples):
            if self.eval_count >= self.budget:
                break
            cand = np.random.uniform(lb, ub, size=self.dim)
            y = float(func(cand))
            self.eval_count += 1
            archive_x.append(cand)
            archive_y.append(y)
            if y < best_y:
                best_y = y
                best_x = cand.copy()

        sigma = 0.15

        while self.eval_count < self.budget:
            # Generate 50 candidate points to screen
            n_candidates = 50
            candidates = np.zeros((n_candidates, self.dim))

            # Half around best_x, half random
            half = n_candidates // 2
            for j in range(half):
                step = np.random.normal(0, 1, size=self.dim) * (sigma * domain_range)
                candidates[j] = np.clip(best_x + step, lb, ub)
            for j in range(half, n_candidates):
                candidates[j] = np.random.uniform(lb, ub, size=self.dim)

            # Predict objective using k-NN
            X_mat = np.array(archive_x)
            Y_mat = np.array(archive_y)
            pred_y = np.zeros(n_candidates)

            for j in range(n_candidates):
                diff = X_mat - candidates[j]
                dists = np.linalg.norm(diff, axis=1)
                
                sorted_idx = np.argsort(dists)[:k_neigh]
                nearest_dists = dists[sorted_idx]
                nearest_y = Y_mat[sorted_idx]

                weights = 1.0 / (nearest_dists + 1e-12)
                pred_y[j] = np.sum(weights * nearest_y) / np.sum(weights)

            # Pick best predicted candidate
            best_cand_idx = np.argmin(pred_y)
            chosen_x = candidates[best_cand_idx]

            # True objective evaluation
            y = float(func(chosen_x))
            self.eval_count += 1

            if len(archive_x) >= max_archive:
                # Remove oldest non-best
                worst_idx = np.argmax(archive_y)
                archive_x.pop(worst_idx)
                archive_y.pop(worst_idx)

            archive_x.append(chosen_x)
            archive_y.append(y)

            if y < best_y:
                best_y = y
                best_x = chosen_x.copy()
                sigma = min(sigma * 1.05, 0.3)
            else:
                sigma = max(sigma * 0.95, 0.01)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
