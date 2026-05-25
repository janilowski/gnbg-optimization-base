# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adapting (1+1)-ES with covariance matrix adaptation (CMA-lite) for black-box minimization.
# Search state: Tracks current best solution (center) and a step size (sigma).
# Candidate generation: New candidates are sampled from a multivariate normal distribution N(center, sigma^2 * I).
# Selection and replacement: Deterministic replacement; if the new candidate is better, it becomes the new center.
# Adaptation: Sigma follows the 1/5th success rule of Rechenberg; it increases on success and decreases on failure to balance exploration/exploitation.
# Exploration mechanisms: Gaussian mutation driven by the current step size (sigma).
# Exploitation mechanisms: Local hill-climbing via successful steps and step size reduction.
# Boundary handling: Candidates are clipped to the provided search space boundaries.
# Budget strategy: A strict counter ensures the loop terminates exactly when the budget is reached.
# Closest known influences: (1+1)-CMA-ES, Rechenberg's 1/5th success rule.
# Novelty or unusual aspects: Lightweight, dependency-free implementation using only numpy.
# Failure modes: Can get stuck in local optima; performance is limited by the fixed identity-based covariance without full matrix adaptation.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialize state
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        self.eval_count += 1
        
        best_x, best_y = np.copy(x), y
        
        # Initial step size (approx 1/5 of range)
        sigma = 0.2 * (ub - lb)
        
        # Success tracking for 1/5 rule
        success_count = 0
        gen_count = 0

        while self.eval_count < self.budget:
            # Generate candidate
            z = np.random.normal(0, 1, self.dim)
            candidate = x + sigma * z
            
            # Boundary handling: clipping
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            f_candidate = func(candidate)
            self.eval_count += 1
            gen_count += 1
            
            # Selection
            if f_candidate < y:
                x, y = candidate, f_candidate
                success_count += 1
                if f_candidate < best_y:
                    best_x, best_y = np.copy(candidate), f_candidate
            
            # Adaptation of sigma (1/5th success rule)
            if gen_count >= 10:
                ratio = success_count / gen_count
                if ratio > 0.2:
                    sigma *= 1.2
                elif ratio < 0.2:
                    sigma *= 0.8
                
                # Reset
                success_count = 0
                gen_count = 0
                
                # Prevent sigma from vanishing
                sigma = np.maximum(sigma, 1e-9 * (ub - lb))

        return best_x, best_y
