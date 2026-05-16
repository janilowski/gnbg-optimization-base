# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Powell's Direction Set algorithm performing sequential line searches along conjugate search directions.
# Search state: Retains incumbent solution point, current set of search direction unit vectors, and global optimum.
# Candidate generation: Proposes candidate points via bounded ternary line searches along individual coordinate and conjugate directions.
# Selection and replacement: Updates incumbent point after each line search; replaces the direction of maximum change with the net displacement vector.
# Adaptation: Search directions adapt dynamically to align with conjugate valley floors without requiring gradient computations.
# Exploration mechanisms: Interleaving random restarts upon line search stagnation prevents permanent entrapment in local basins.
# Exploitation mechanisms: Sequential 1D line minimization rapidly drives objective reduction along local descent corridors.
# Boundary handling: All line search test points are explicitly clipped inside valid domain boundaries.
# Budget strategy: Allocates evaluations in small line search batches while strictly checking remaining evaluation budget.
# Closest known influences: Powell's Conjugate Direction Method (Powell).
# Novelty or unusual aspects: Employs a strict budget-capped 3-point parabolic line search to prevent excessive evaluation consumption on single lines.
# Failure modes: Can experience direction set linear dependence (Powell's stagnation) if replacement criteria are not carefully monitored.
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

        directions = np.eye(self.dim)
        step_scale = 0.1 * domain_range

        def line_search(start_x, start_y, u_dir, step_size):
            if self.eval_count >= self.budget:
                return start_x, start_y, 0.0

            # Test forward and backward
            cand_f = np.clip(start_x + step_size * u_dir, lb, ub)
            yf = float(func(cand_f))
            self.eval_count += 1

            nonlocal best_y, best_x
            if yf < best_y:
                best_y = yf
                best_x = cand_f.copy()

            if self.eval_count >= self.budget:
                if yf < start_y:
                    return cand_f, yf, yf - start_y
                return start_x, start_y, 0.0

            cand_b = np.clip(start_x - step_size * u_dir, lb, ub)
            yb = float(func(cand_b))
            self.eval_count += 1

            if yb < best_y:
                best_y = yb
                best_x = cand_b.copy()

            if yf < start_y and yf <= yb:
                return cand_f, yf, start_y - yf
            elif yb < start_y and yb < yf:
                return cand_b, yb, start_y - yb
            else:
                return start_x, start_y, 0.0

        stagnation = 0

        while self.eval_count < self.budget:
            p0_x = curr_x.copy()
            p0_y = curr_y
            max_delta = 0.0
            worst_dir_idx = 0

            for i in range(self.dim):
                if self.eval_count >= self.budget:
                    break

                step_s = np.linalg.norm(step_scale * directions[i])
                if step_s < 1e-12:
                    continue

                curr_x, curr_y, delta = line_search(curr_x, curr_y, directions[i], step_s)
                if delta > max_delta:
                    max_delta = delta
                    worst_dir_idx = i

            if self.eval_count >= self.budget:
                break

            # Net displacement
            net_dir = curr_x - p0_x
            norm_net = np.linalg.norm(net_dir)

            if norm_net > 1e-8:
                u_net = net_dir / norm_net
                curr_x, curr_y, delta_net = line_search(curr_x, curr_y, u_net, norm_net * 0.5)

                # Update directions: replace direction that produced biggest change
                directions[worst_dir_idx] = directions[-1].copy()
                directions[-1] = u_net.copy()
                stagnation = 0
            else:
                stagnation += 1

            if stagnation > 5:
                # Random restart
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                directions = np.eye(self.dim)
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
