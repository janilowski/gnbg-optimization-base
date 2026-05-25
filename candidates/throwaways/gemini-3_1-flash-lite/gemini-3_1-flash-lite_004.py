# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free local search algorithm based on a symmetric sampling pattern (Pattern Search).
# Search state: Tracks the current best point (incumbent) and current step size (sigma).
# Candidate generation: Generates a set of 2*dim orthogonal candidates around the current incumbent by perturbing one coordinate at a time by sigma.
# Selection and replacement: Greedy strategy; if a candidate improves the incumbent, the incumbent is updated to the best found candidate.
# Adaptation: If an iteration fails to find an improvement, sigma is halved; if it succeeds, sigma is maintained or slightly increased.
# Exploration mechanisms: Initial search explores a large region dictated by the problem bounds; subsequent iterations refine locally.
# Exploitation mechanisms: Hill climbing via coordinate-wise probes.
# Boundary handling: Candidates are clipped to the valid domain [lb, ub] before evaluation.
# Budget strategy: Terminates exactly when the evaluation counter reaches the remaining budget.
# Closest known influences: Hooke-Jeeves pattern search / Coordinate Descent.
# Novelty or unusual aspects: None; chosen for robustness and simplicity in black-box optimization.
# Failure modes: Susceptible to getting trapped in local minima in rugged landscapes due to the greedy, local nature of the search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialize state
        curr_x = np.random.uniform(lb, ub)
        curr_y = func(curr_x)
        evals_used = 1
        
        # Initial step size as a fraction of the search space
        sigma = (ub - lb) * 0.2
        
        best_x = curr_x.copy()
        best_y = curr_y

        while evals_used < self.budget:
            improved = False
            # Generate 2*dim orthogonal candidates
            directions = np.eye(self.dim)
            candidates = []
            for d in range(self.dim):
                for sign in [-1, 1]:
                    step = sign * sigma * directions[d]
                    candidates.append(np.clip(curr_x + step, lb, ub))

            # Evaluate candidates
            for cand in candidates:
                if evals_used >= self.budget:
                    break
                
                y = func(cand)
                evals_used += 1
                
                if y < best_y:
                    best_y = y
                    best_x = cand.copy()
                    curr_x = cand.copy()
                    improved = True

            # Adaptive step size
            if improved:
                # If improvement found, keep step size or accelerate
                sigma = np.clip(sigma * 1.1, (ub - lb) * 1e-6, (ub - lb) * 0.5)
            else:
                # Shrink step size if no improvement to refine locally
                sigma *= 0.5
                
            # Convergence check: stop if sigma is effectively zero
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
