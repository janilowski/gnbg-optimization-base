# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A stochastic local search algorithm utilizing an annealed random walk with periodic restarts.
# Search state: Current best position, current local center, current step size (sigma), and evaluation counter.
# Candidate generation: Perturbations are drawn from a Gaussian distribution centered at the current best point, scaled by the domain range and a decaying temperature factor.
# Selection and replacement: Greedy selection is used; the algorithm only updates its center of search when a point with a lower objective value is found.
# Adaptation: The step size (sigma) decays exponentially within each restart cycle to transition from exploration to exploitation.
# Exploration mechanisms: The algorithm uses random restarts from fresh coordinates in the search space when local progress stalls or a specific cycle length is reached.
# Exploitation mechanisms: Local search with decreasing variance (annealing) focuses the search on promising regions.
# Boundary handling: All generated candidates are clipped to the hyper-rectangle defined by the function's lower and upper bounds.
# Budget strategy: A strict counter is maintained to terminate exactly when the provided budget is exhausted.
# Closest known influences: Simulated Annealing (without probabilistic uphill moves), Local Hill Climbing, and Multi-start Local Search.
# Novelty or unusual aspects: The marriage of exponential step-size decay with a budget-aware restart schedule ensures both global coverage and local refinement.
# Failure modes: Highly needle-in-a-haystack landscapes or extremely high-dimensional spaces where the random walk covers a negligible fraction of the volume.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    An annealed random walk algorithm with restarts for black-box minimization.
    It balances exploration via restarts and exploitation via shrinking Gaussian perturbations.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Determine boundaries from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        else:
            # Fallback for unexpected interface, though GNBG standards apply
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)

        bounds_range = ub - lb
        best_x = None
        best_y = float('inf')
        
        # Hyperparameters for the annealed walk
        # Restart period: how often we jump to a new random location
        # A simple heuristic based on dimension and budget
        restart_period = max(20, int(self.budget / (5 + np.log(self.dim + 1))))
        
        while self.eval_count < self.budget:
            # Restart: Pick a new random start point within bounds
            curr_x = lb + np.random.rand(self.dim) * bounds_range
            curr_y = func(curr_x)
            self.eval_count += 1
            
            if curr_y < best_y:
                best_y = curr_y
                best_x = np.copy(curr_x)
            
            if self.eval_count >= self.budget:
                break

            # Local annealed walk
            # Each restart gets its own annealing schedule
            steps_in_cycle = min(restart_period, self.budget - self.eval_count)
            
            for i in range(steps_in_cycle):
                # Annealing schedule: sigma decreases from 0.5 to 0.0001
                # The decay is relative to the progress through the current restart cycle
                t = i / steps_in_cycle
                sigma = 0.5 * (0.0001 / 0.5)**t
                
                # Generate candidate
                noise = np.random.normal(0, sigma, self.dim)
                candidate_x = curr_x + noise * bounds_range
                
                # Boundary handling: clip to bounds
                candidate_x = np.clip(candidate_x, lb, ub)
                
                # Evaluate
                candidate_y = func(candidate_x)
                self.eval_count += 1
                
                # Greedy selection
                if candidate_y < curr_y:
                    curr_y = candidate_y
                    curr_x = candidate_x
                    
                    # Global best update
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = np.copy(curr_x)
                
                if self.eval_count >= self.budget:
                    break
                    
        return best_x, best_y
