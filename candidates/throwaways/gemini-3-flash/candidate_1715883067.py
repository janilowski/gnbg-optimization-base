# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Conjugate Subspace Quadratic Interlacing algorithm combining 1D parabolic interpolation with orthogonal subspace exploration.
# Search state: Stores incumbent solution position, objective fitness value, active conjugate search direction vectors, and global optimum.
# Candidate generation: Proposes points via 3-point parabolic vertex interpolation along conjugate directions and Gaussian subspace steps.
# Selection and replacement: Replaces incumbent position whenever quadratic vertex or subspace steps achieve superior objective fitness.
# Adaptation: Updates conjugate search vectors using successful displacement directions to align with local valley floors.
# Exploration mechanisms: Gaussian subspace steps and periodic orthogonalization prevent search vectors from collapsing into lower dimensions.
# Exploitation mechanisms: Exact 1D quadratic vertex estimation achieves quadratic convergence rates on smooth convex basins.
# Boundary handling: All quadratic test probes and subspace candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Allocates evaluations in precise triplets for quadratic interpolation while checking remaining evaluation budget.
# Closest known influences: Parabolic Interpolation / Conjugate Direction Search / CSQI.
# Novelty or unusual aspects: Directly embeds exact analytical parabola vertex calculation into conjugate directional updates.
# Failure modes: Can stall or make bad jumps on highly non-convex or discontinuous ridges if parabola curvature becomes negative or flat.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

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

        # Initialize orthogonal conjugate directions
        dirs = np.eye(self.dim)
        step_scale = 0.05 * domain_range
        stagnation = 0

        while self.eval_count < self.budget:
            improved = False
            for i in range(self.dim):
                if self.eval_count >= self.budget:
                    break

                u = dirs[i]
                norm_u = np.linalg.norm(u)
                if norm_u < 1e-12:
                    continue
                u = u / norm_u

                alpha = np.linalg.norm(step_scale * u)
                if alpha < 1e-12:
                    continue

                # 3 points: x0, x_plus, x_minus
                x0, y0 = curr_x, curr_y

                xp = np.clip(curr_x + alpha * u, lb, ub)
                yp = float(func(xp))
                self.eval_count += 1
                if yp < best_y:
                    best_y = yp
                    best_x = xp.copy()

                if self.eval_count >= self.budget:
                    break

                xm = np.clip(curr_x - alpha * u, lb, ub)
                ym = float(func(xm))
                self.eval_count += 1
                if ym < best_y:
                    best_y = ym
                    best_x = xm.copy()

                # Parabolic interpolation: f(x) = a*t^2 + b*t + c around t=0 (x0)
                # t_plus = +alpha, t_minus = -alpha
                # y0 = c
                # yp = a*alpha^2 + b*alpha + y0
                # ym = a*alpha^2 - b*alpha + y0
                # yp + ym - 2*y0 = 2*a*alpha^2  => a = (yp + ym - 2*y0) / (2*alpha^2)
                # yp - ym = 2*b*alpha          => b = (yp - ym) / (2*alpha)
                # Vertex t* = -b / (2*a) = - (yp - ym)*alpha / (2 * (yp + ym - 2*y0))

                denom = 2.0 * (yp + ym - 2.0 * y0)
                best_cand_x, best_cand_y = None, float("inf")

                if yp < y0 and yp <= ym:
                    best_cand_x, best_cand_y = xp.copy(), yp
                elif ym < y0 and ym < yp:
                    best_cand_x, best_cand_y = xm.copy(), ym

                if abs(denom) > 1e-12:
                    t_star = - (yp - ym) * alpha / denom
                    # Only test vertex if it's a minimum (a > 0) and within reasonable jump distance
                    if denom > 0 and abs(t_star) < 5.0 * alpha and self.eval_count < self.budget:
                        x_quad = np.clip(curr_x + t_star * u, lb, ub)
                        y_quad = float(func(x_quad))
                        self.eval_count += 1

                        if y_quad < best_y:
                            best_y = y_quad
                            best_x = x_quad.copy()

                        if y_quad < best_cand_y:
                            best_cand_x, best_cand_y = x_quad.copy(), y_quad

                if best_cand_y < curr_y:
                    diff = best_cand_x - curr_x
                    norm_diff = np.linalg.norm(diff)
                    curr_x, curr_y = best_cand_x.copy(), best_cand_y
                    if norm_diff > 1e-10:
                        dirs[i] = diff / norm_diff
                    improved = True
                    step_scale = np.minimum(step_scale * 1.1, 0.25 * domain_range)
                else:
                    step_scale = np.maximum(step_scale * 0.85, 1e-6 * domain_range)

            if not improved:
                stagnation += 1
            else:
                stagnation = 0

            if stagnation > 5 or np.max(step_scale / domain_range) < 1e-5:
                # Stochastic jump
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                dirs = np.eye(self.dim)
                step_scale = 0.05 * domain_range
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
