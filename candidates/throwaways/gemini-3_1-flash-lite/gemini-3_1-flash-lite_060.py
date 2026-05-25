# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple, robust Adaptive Random Search with a shrinking hypersphere radius.
# Search state: Maintains the current 'best_x' found so far and the current search radius 'sigma'.
# Candidate generation: Generates new candidates by sampling from a multivariate normal distribution centered at 'best_x' with scale 'sigma'.
# Selection and replacement: Simple greedy update; if a candidate yields a lower function value, it replaces 'best_x'.
# Adaptation: 'sigma' is adapted using the one-fifth success rule: if the success rate is > 20%, 'sigma' increases; if < 20%, 'sigma' decreases.
# Exploration mechanisms: The initial large sigma allows for global exploration, while successive adaptation transitions toward local refinement.
# Exploitation mechanisms: The algorithm narrows down around the current best candidate as the budget decreases.
# Boundary handling: Candidates are clipped to the domain bounds using numpy.clip.
# Budget strategy: Iterates until the evaluation count hits the budget exactly.
# Closest known influences: Adaptive Step-Size Random Search / 1/5th Rule.
# Novelty or unusual aspects: Extremely lightweight, zero-dependency implementation suitable for black-box problems in any dimension.
# Failure modes: Can get stuck in narrow local minima; performance degrades in highly rugged landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds safely
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        else:
            raise ValueError("Function must define search bounds.")

        # Initialize state
        current_x = np.random.uniform(lb, ub)
        current_y = func(current_x)
        budget_remaining = self.budget - 1
        
        # Sigma is roughly 1/5th of the search space reach
        sigma = (ub - lb) * 0.2
        
        best_x, best_y = np.copy(current_x), current_y
        
        # Tracking for adaptation
        successes = 0
        total_steps = 0
        
        while budget_remaining > 0:
            # Generate candidate
            step = np.random.normal(0, sigma, self.dim)
            candidate = np.clip(current_x + step, lb, ub)
            
            candidate_y = func(candidate)
            budget_remaining -= 1
            total_steps += 1
            
            # Acceptance logic
            if candidate_y < current_y:
                current_x, current_y = np.copy(candidate), candidate_y
                successes += 1
                if current_y < best_y:
                    best_x, best_y = np.copy(current_x), current_y
            
            # 1/5th rule adaptation every few steps
            if total_steps >= 10:
                rate = successes / total_steps
                if rate > 0.2:
                    sigma *= 1.1  # Expand
                elif rate < 0.2:
                    sigma *= 0.9  # Contract
                successes, total_steps = 0, 0
                
        return best_x, best_y
