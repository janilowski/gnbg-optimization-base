# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Quasi-Newton optimization algorithm utilizing finite difference gradient estimation and BFGS inverse Hessian updates.
# Search state: Retains incumbent position, current gradient vector, approximate inverse Hessian matrix, and global optimum.
# Candidate generation: Generates search directions via matrix multiplication of the inverse Hessian and estimated gradient, followed by backtracking line search.
# Selection and replacement: Moves to new line search positions if the objective value satisfies sufficient decrease or strict improvement.
# Adaptation: Adapts curvature information by updating the inverse Hessian matrix using secant equation rank-two updates (BFGS formula).
# Exploration mechanisms: Re-seeds the search at random domain coordinates whenever the gradient vanishes or curvature curvature updates collapse.
# Exploitation mechanisms: Quadratic convergence rate in smooth convex basins via second-order curvature approximation.
# Boundary handling: All finite difference probes and line search steps are strictly clipped inside valid variable bounds.
# Budget strategy: Allocates budget to finite difference stencils and line searches sequentially, checking limits prior to every function evaluation.
# Closest known influences: BFGS Quasi-Newton method with numerical differentiation.
# Novelty or unusual aspects: Fallback restart mechanism when non-smooth landscapes cause non-positive definite Hessian updates.
# Failure modes: High evaluation overhead per gradient step in large dimensions; struggles on highly discontinuous or noisy surfaces.
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
        h = 1e-6 * domain_range

        curr_x = np.random.uniform(lb, ub, size=self.dim)
        curr_y = float(func(curr_x))
        self.eval_count += 1

        best_x = curr_x.copy()
        best_y = curr_y

        H = np.eye(self.dim)
        I = np.eye(self.dim)

        while self.eval_count < self.budget:
            # Estimate gradient via forward finite differences
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                if self.eval_count >= self.budget:
                    break
                probe = curr_x.copy()
                probe[i] = np.clip(probe[i] + h[i], lb[i], ub[i])
                
                # Ensure actual step size is non-zero
                actual_h = probe[i] - curr_x[i]
                if abs(actual_h) < 1e-12:
                    probe[i] = np.clip(curr_x[i] - h[i], lb[i], ub[i])
                    actual_h = probe[i] - curr_x[i]

                if abs(actual_h) < 1e-12:
                    grad[i] = 0.0
                    continue

                y_probe = float(func(probe))
                self.eval_count += 1

                if y_probe < best_y:
                    best_y = y_probe
                    best_x = probe.copy()

                grad[i] = (y_probe - curr_y) / actual_h

            if self.eval_count >= self.budget:
                break

            norm_g = np.linalg.norm(grad)
            if norm_g < 1e-8:
                # Gradient vanished: restart
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                curr_y = float(func(curr_x))
                self.eval_count += 1
                if curr_y < best_y:
                    best_y = curr_y
                    best_x = curr_x.copy()
                H = np.eye(self.dim)
                continue

            # Compute search direction
            direction = -H @ grad

            # Backtracking line search
            step_len = 1.0
            found_step = False
            next_x = curr_x.copy()
            next_y = curr_y

            for _ in range(6):
                if self.eval_count >= self.budget:
                    break

                cand = np.clip(curr_x + step_len * direction, lb, ub)
                y_cand = float(func(cand))
                self.eval_count += 1

                if y_cand < best_y:
                    best_y = y_cand
                    best_x = cand.copy()

                if y_cand < curr_y - 1e-4 * step_len * (grad @ direction):
                    found_step = True
                    next_x = cand.copy()
                    next_y = y_cand
                    break
                step_len *= 0.5

            if not found_step:
                # Line search failed: restart
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                H = np.eye(self.dim)
                continue

            # Estimate next gradient for BFGS update
            next_grad = np.zeros(self.dim)
            for i in range(self.dim):
                if self.eval_count >= self.budget:
                    break
                probe = next_x.copy()
                probe[i] = np.clip(probe[i] + h[i], lb[i], ub[i])
                actual_h = probe[i] - next_x[i]
                if abs(actual_h) < 1e-12:
                    probe[i] = np.clip(next_x[i] - h[i], lb[i], ub[i])
                    actual_h = probe[i] - next_x[i]
                if abs(actual_h) < 1e-12:
                    next_grad[i] = 0.0
                    continue

                y_probe = float(func(probe))
                self.eval_count += 1

                if y_probe < best_y:
                    best_y = y_probe
                    best_x = probe.copy()

                next_grad[i] = (y_probe - next_y) / actual_h

            if self.eval_count >= self.budget:
                break

            # BFGS update
            s = next_x - curr_x
            y_diff = next_grad - grad
            rho_denom = np.dot(y_diff, s)

            if rho_denom > 1e-10:
                rho = 1.0 / rho_denom
                A = I - rho * np.outer(s, y_diff)
                B = I - rho * np.outer(y_diff, s)
                H = A @ H @ B + rho * np.outer(s, s)
            else:
                H = np.eye(self.dim)

            curr_x = next_x
            curr_y = next_y

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
