import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An adaptive pattern search algorithm that performs greedy coordinate-wise descent 
# with a shrinking search radius. It identifies local improvements by probing along 
# individual axes and contracts the search space when no improvements are found.
# Search state: Current best position (x), current best value (y), and the current 
# step size (sigma) relative to the domain range.
# Candidate generation: Generates candidates by perturbing the current best solution 
# along one dimension at a time, using both positive and negative offsets scaled by sigma.
# Selection and replacement: Uses a greedy selection strategy; any candidate that 
# strictly improves the objective function becomes the new search center immediately.
# Adaptation: The step size (sigma) is halved whenever a complete cycle through all 
# dimensions fails to produce any improvement.
# Exploration mechanisms: Starts with a random point in the bounded space and a 
# relatively large initial step size (0.5 of the range) to cover the domain.
# Exploitation mechanisms: Progressively reduces the step size, allowing the search 
# to refine the solution in increasingly smaller neighborhoods of the current best.
# Boundary handling: All generated candidates are clipped to the lower and 
# upper bounds of the search space using numpy.clip.
# Budget strategy: The algorithm tracks evaluations and terminates immediately 
# once the budget is reached, returning the best solution found so far.
# Closest known influences: Hooke-Jeeves Pattern Search, Luus-Jaakola heuristic.
# Novelty or unusual aspects: Employs a simplified "greedy" variant of pattern 
# search that updates the center mid-cycle, potentially speeding up convergence 
# on axis-aligned slopes.
# Failure modes: May converge to local optima in highly multi-modal landscapes 
# or struggle on narrow, non-axis-aligned ridges (the "canyon" problem).
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        else:
            # Fallback for unbounded or unspecified
            lb = np.full(self.dim, -100.0)
            ub = np.full(self.dim, 100.0)

        # Initial random point
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        self.eval_count += 1
        
        # Search parameters
        # sigma is the step size relative to the range of each dimension
        sigma = 0.5
        ranges = ub - lb
        
        # Main search loop
        while self.eval_count < self.budget:
            improved = False
            
            # Create a permutation of indices to avoid directional bias
            indices = np.random.permutation(self.dim)
            
            for d in indices:
                if self.eval_count >= self.budget:
                    break
                
                # Calculate the actual step for this dimension
                step = sigma * ranges[d]
                
                # Try two directions (positive and negative)
                for direction in [1, -1]:
                    if self.eval_count >= self.budget:
                        break
                        
                    candidate_x = best_x.copy()
                    candidate_x[d] = np.clip(candidate_x[d] + direction * step, lb[d], ub[d])
                    
                    # If clipping resulted in no change, skip evaluation
                    if np.array_equal(candidate_x, best_x):
                        continue
                        
                    val = func(candidate_x)
                    self.eval_count += 1
                    
                    if val < best_y:
                        best_y = val
                        best_x = candidate_x
                        improved = True
                        # If we found an improvement, move the center and 
                        # continue searching from the new position
                        break 

            # Adaptation: If no improvement was found in any dimension, shrink the step
            if not improved:
                sigma *= 0.5
                
            # If sigma becomes infinitesimally small, reset to encourage exploration
            # or just continue if budget is nearly exhausted.
            if sigma < 1e-12:
                sigma = 0.1

        return best_x, best_y
