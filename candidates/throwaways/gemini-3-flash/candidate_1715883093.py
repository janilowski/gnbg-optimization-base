# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Gaussian Quantum-Behaved Particle Swarm Optimization algorithm replacing standard velocity equations with quantum delta potential well wavefunctions.
# Search state: Retains agent population positions, personal best positions and fitnesses, mean best position vector, and global optimum.
# Candidate generation: Proposes positions via Gaussian stochastic sampling around attractor coordinates scaled by absolute differences from the swarm mean best.
# Selection and replacement: Standard personal best updates; updates global best optimum whenever a candidate achieves superior objective fitness.
# Adaptation: Contraction-expansion coefficient alpha decreases linearly over iterations to tighten quantum potential wells around leading attractors.
# Exploration mechanisms: Gaussian sampling around attractor coordinates scaled by swarm diversity (mbest) ensures multi-directional exploration.
# Exploitation mechanisms: Attractor coordinates weighted towards the global best anchor pull particles directly into the elite optimum basin.
# Boundary handling: All agent candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates swarm members sequentially in generational iterations while strictly checking remaining evaluation budget.
# Closest known influences: Quantum-Behaved Particle Swarm Optimization QPSO (Sun et al.) / Gaussian QPSO.
# Novelty or unusual aspects: Directly replaces standard Laplacian double-exponential sampling with smooth multivariate Gaussian perturbations.
# Failure modes: Can experience premature variance collapse if all personal best positions converge on a degenerate flat plane.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 8, max(20, 2 * self.dim)))
        if self.pop_size > 50:
            self.pop_size = 50

    def __call__(self, func):
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        self.eval_count = 0

        best_x = None
        best_y = float("inf")

        pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        pbest_x = pop.copy()
        pbest_y = np.full(self.pop_size, float("inf"))

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop[i]))
            self.eval_count += 1
            pbest_y[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        alpha_start, alpha_end = 1.0, 0.5

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            alpha = alpha_start - (alpha_start - alpha_end) * progress
            mbest = np.mean(pbest_x, axis=0)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                phi = np.random.rand(self.dim)
                attractor = phi * pbest_x[i] + (1.0 - phi) * best_x

                z = np.random.normal(0, 1, size=self.dim)
                sign_flip = np.where(np.random.rand(self.dim) < 0.5, 1.0, -1.0)

                step = alpha * np.abs(mbest - pop[i]) * np.abs(z) * sign_flip
                trial = np.clip(attractor + step, lb, ub)

                y = float(func(trial))
                self.eval_count += 1
                pop[i] = trial

                if y <= pbest_y[i]:
                    pbest_y[i] = y
                    pbest_x[i] = trial.copy()
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
