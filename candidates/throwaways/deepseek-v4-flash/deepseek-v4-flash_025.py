import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Simple (1+1)-Evolution Strategy with Rechenberg's 1/5 success rule for adaptive step-size. Uses isotropic Gaussian mutation. Boundaries handled by reflection.
# Search state: Single parent solution (current best candidate) and its objective value. Also stores global best ever found.
# Candidate generation: Mutation by adding isotropic Gaussian noise scaled by adaptive step size sigma.
# Selection and replacement: Greedy (1+1) - if offspring is better than parent, replace parent. Otherwise parent unchanged.
# Adaptation: Step size sigma adjusted every iteration using success rate over recent window (last 50 or budget//2 events). If success rate > 0.2, sigma increased; if < 0.2, sigma decreased.
# Exploration mechanisms: Step size adaptation maintains diversity; initial sigma 20% of domain range; potential increase if many successes indicates too small steps.
# Exploitation mechanisms: Once converged, sigma becomes small; greedy selection focuses on best point.
# Boundary handling: Reflection on each coordinate. If out-of-bounds, reflect back into domain; may cause multiple reflections if step large.
# Budget strategy: Uses entire budget; evaluations counted strictly. No extra overhead.
# Closest known influences: Rechenberg's (1+1)-ES with 1/5 rule.
# Novelty or unusual aspects: None; standard textbook algorithm.
# Failure modes: May get stuck in local optima if domain is highly multimodal and sigma shrinks prematurely. Not suitable for separable or constrained problems. Reflection may bias near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Read bounds
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb, dtype=float)
                ub = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Cannot determine bounds from func")

        if lb.ndim == 0:
            lb = np.broadcast_to(lb, self.dim).copy()
            ub = np.broadcast_to(ub, self.dim).copy()

        domain_range = ub - lb
        # Initial point
        parent = lb + np.random.uniform(0, 1, size=self.dim) * domain_range
        f_parent = func(parent)
        best_x = parent.copy()
        best_y = f_parent
        evals = 1

        if self.budget <= 1:
            return best_x, best_y

        # Step size parameters
        sigma = 0.2 * np.mean(domain_range)
        sigma_min = 1e-8 * np.mean(domain_range)
        sigma_max = 0.5 * np.mean(domain_range)

        # Success record window
        window_size = min(50, self.budget // 2)
        successes = []   # list of 0/1 success indicators

        while evals < self.budget:
            # Generate candidate via isotropic Gaussian mutation
            candidate = parent + sigma * np.random.randn(self.dim)

            # Boundary reflection
            out_low = candidate < lb
            out_high = candidate > ub
            candidate[out_low] = 2 * lb[out_low] - candidate[out_low]
            candidate[out_high] = 2 * ub[out_high] - candidate[out_high]
            candidate = np.clip(candidate, lb, ub)

            # Evaluate
            f_candidate = func(candidate)
            evals += 1

            # Success?
            success = f_candidate < f_parent
            successes.append(1 if success else 0)
            if len(successes) > window_size:
                successes.pop(0)

            if success:
                parent = candidate
                f_parent = f_candidate
                if f_candidate < best_y:
                    best_x = candidate.copy()
                    best_y = f_candidate

            # Step size adaptation (once window is filled)
            if evals >= window_size and (evals % 1 == 0):
                rate = np.mean(successes)
                if rate > 0.2:
                    sigma *= 1.2
                elif rate < 0.2:
                    sigma *= 0.85
                sigma = np.clip(sigma, sigma_min, sigma_max)

        return best_x, best_y
