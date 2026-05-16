# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Coordinate Descent algorithm incorporating directional momentum and individual dimension step size adaptation.
# Search state: Retains incumbent solution point, objective fitness value, coordinate step size vector, momentum velocity vector, and global optimum.
# Candidate generation: Proposes candidate moves along randomly permuted coordinates combined with inertia from past successful steps.
# Selection and replacement: Evaluates forward and backward coordinate steps, replacing the incumbent solution immediately upon finding an improvement.
# Adaptation: Multiplicatively expands coordinate step size upon success and contracts coordinate step size upon bi-directional failure.
# Exploration mechanisms: Random coordinate permutations and directional momentum prevent stagnation along diagonal ridge corridors.
# Exploitation mechanisms: Exact coordinate descent updates and rapid step size contraction pinpoint local coordinate minima.
# Boundary handling: All coordinate candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates coordinate steps sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Coordinate Descent / Solis & Wets / Momentum Optimization.
# Novelty or unusual aspects: Directly embeds continuous momentum inertia into discrete axial coordinate probes.
# Failure modes: Can experience zig-zagging inefficiency on highly non-separable rotated elliptic valleys.
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

        curr_x = np.random.uniform(lb, ub, size=self.dim)
        curr_y = float(func(curr_x))
        self.eval_count += 1

        best_x = curr_x.copy()
        best_y = curr_y

        sigma = 0.1 * domain_range
        min_sigma = 1e-6 * domain_range
        max_sigma = 0.4 * domain_range
        vel = np.zeros(self.dim)
        momentum = 0.5

        stagnation = 0

        while self.eval_count < self.budget:
            perm = np.random.permutation(self.dim)
            improved_gen = False

            for j in perm:
                if self.eval_count >= self.budget:
                    break

                # Forward step with momentum
                step_f = np.zeros(self.dim)
                step_f[j] = sigma[j]
                cand_f = np.clip(curr_x + step_f + momentum * vel, lb, ub)
                yf = float(func(cand_f))
                self.eval_count += 1

                if yf < best_y:
                    best_y = yf
                    best_x = cand_f.copy()

                if yf < curr_y:
                    diff = cand_f - curr_x
                    curr_x = cand_f.copy()
                    curr_y = yf
                    vel = momentum * vel + diff
                    sigma[j] = min(sigma[j] * 1.2, max_sigma[j])
                    improved_gen = True
                    continue

                if self.eval_count >= self.budget:
                    break

                # Backward step with momentum
                step_b = np.zeros(self.dim)
                step_b[j] = -sigma[j]
                cand_b = np.clip(curr_x + step_b + momentum * vel, lb, ub)
                yb = float(func(cand_b))
                self.eval_count += 1

                if yb < best_y:
                    best_y = yb
                    best_x = cand_b.copy()

                if yb < curr_y:
                    diff = cand_b - curr_x
                    curr_x = cand_b.copy()
                    curr_y = yb
                    vel = momentum * vel + diff
                    sigma[j] = min(sigma[j] * 1.2, max_sigma[j])
                    improved_gen = True
                else:
                    sigma[j] = max(sigma[j] * 0.75, min_sigma[j])
                    vel[j] *= 0.5

            if not improved_gen:
                stagnation += 1
            else:
                stagnation = 0

            if stagnation > 15 or np.max(sigma / domain_range) < 1e-5:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                sigma = 0.1 * domain_range
                vel = np.zeros(self.dim)
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
