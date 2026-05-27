# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Hooke-Jeeves Pattern Search algorithm combining exploratory axial coordinate moves with directional pattern acceleration jumps.
# Search state: Retains incumbent solution position, objective fitness value, previous base position, step size parameter, and global optimum.
# Candidate generation: Alternates between directional pattern acceleration moves along successful trajectories and exploratory coordinate axial steps.
# Selection and replacement: Replaces incumbent position whenever exploratory moves following pattern jumps discover improved objective values.
# Adaptation: Halves coordinate step size upon exploratory search failure; resets base positions upon successful pattern moves.
# Exploration mechanisms: Directional pattern moves jump over shallow local valleys, accelerating search along descent corridors.
# Exploitation mechanisms: Exploratory coordinate axial moves systematically probe orthogonal axes to pinpoint exact local minima.
# Boundary handling: All pattern acceleration moves and axial test points are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates exploratory coordinate moves sequentially while strictly checking remaining evaluation budget limits.
# Closest known influences: Hooke-Jeeves Pattern Search (Hooke & Jeeves).
# Novelty or unusual aspects: Integrates exact boundary clipping directly into the pattern acceleration move to prevent constraint locking.
# Failure modes: Can experience sluggish convergence along narrow non-separable valleys not aligned with standard coordinate axes.
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

        base_x = curr_x.copy()
        sigma = 0.1 * domain_range
        min_sigma = 1e-6 * domain_range
        stagnation = 0

        def exploratory_move(start_x, start_y):
            ex_x = start_x.copy()
            ex_y = start_y
            improved = False

            for i in range(self.dim):
                if self.eval_count >= self.budget:
                    break

                step = np.zeros(self.dim)
                step[i] = sigma[i]

                # Test forward
                cand_f = np.clip(ex_x + step, lb, ub)
                yf = float(func(cand_f))
                self.eval_count += 1

                nonlocal best_y, best_x
                if yf < best_y:
                    best_y = yf
                    best_x = cand_f.copy()

                if yf < ex_y:
                    ex_x = cand_f
                    ex_y = yf
                    improved = True
                    continue

                if self.eval_count >= self.budget:
                    break

                # Test backward
                cand_b = np.clip(ex_x - step, lb, ub)
                yb = float(func(cand_b))
                self.eval_count += 1

                if yb < best_y:
                    best_y = yb
                    best_x = cand_b.copy()

                if yb < ex_y:
                    ex_x = cand_b
                    ex_y = yb
                    improved = True

            return ex_x, ex_y, improved

        while self.eval_count < self.budget:
            # Pattern move
            pattern_x = np.clip(2.0 * curr_x - base_x, lb, ub)
            if self.eval_count >= self.budget:
                break
            pattern_y = float(func(pattern_x))
            self.eval_count += 1

            if pattern_y < best_y:
                best_y = pattern_y
                best_x = pattern_x.copy()

            # Exploratory move from pattern point
            ex_x, ex_y, improved = exploratory_move(pattern_x, pattern_y)

            if ex_y < curr_y:
                base_x = curr_x.copy()
                curr_x = ex_x.copy()
                curr_y = ex_y
                stagnation = 0
            else:
                # Failed from pattern point, try exploratory move from curr_x directly
                if self.eval_count >= self.budget:
                    break
                ex_x, ex_y, improved = exploratory_move(curr_x, curr_y)
                
                if improved:
                    base_x = curr_x.copy()
                    curr_x = ex_x.copy()
                    curr_y = ex_y
                    stagnation = 0
                else:
                    sigma = np.maximum(sigma * 0.5, min_sigma)
                    stagnation += 1

            if stagnation > 20 or np.max(sigma / domain_range) < 1e-5:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                base_x = curr_x.copy()
                sigma = 0.1 * domain_range
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
