# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A multi-start local search algorithm using an adaptive (1+1) Evolution Strategy logic.
# Search state: Current best position in the local trajectory, current step size (sigma), and the global best found across all restarts.
# Candidate generation: New candidates are generated using Gaussian perturbations centered at the current local best, scaled by a coordinate-wise step size or a global scalar.
# Selection and replacement: Greedy selection; a candidate replaces the current point if its objective value is lower or equal.
# Adaptation: The step size (sigma) is adjusted using a success-based rule: it expands when a better point is found and contracts when a proposal fails.
# Exploration mechanisms: Restarts from random locations within the search space whenever the local step size becomes negligible or a trajectory length limit is reached.
# Exploitation mechanisms: Local hill-climbing with adaptive step size refinement allows for fine-tuning solutions once a promising basin is found.
# Boundary handling: Candidates are clipped to the hypercube defined by the problem bounds.
# Budget strategy: The algorithm tracks the total number of function evaluations and terminates immediately once the provided budget is exhausted.
# Closest known influences: (1+1)-ES (Evolution Strategy), Matyas search, and Randomized Local Search (RLS).
# Novelty or unusual aspects: Combines a simple success-rule adaptation with a restart mechanism to handle multi-modal landscapes without complex population dynamics.
# Failure modes: May struggle with highly ridge-like or non-separable functions where diagonal Gaussian mutations are inefficient, or extremely high-dimensional spaces where restarts are too infrequent.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the optimization algorithm.
        
        Args:
            budget: Total number of allowed function evaluations.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Runs the optimization process on the provided function.
        
        Args:
            func: Objective function to minimize.
        
        Returns:
            tuple: (best_x, best_y) found during the search.
        """
        # Extract bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback to defaults if bounds are not provided
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        rng_range = ub - lb
        evals = 0
        
        global_best_x = None
        global_best_y = float('inf')

        # Multi-start loop
        while evals < self.budget:
            # Initialize a new local search trajectory
            curr_x = lb + np.random.rand(self.dim) * rng_range
            
            if evals >= self.budget: break
            curr_y = func(curr_x)
            evals += 1
            
            if curr_y < global_best_y:
                global_best_y = curr_y
                global_best_x = np.copy(curr_x)

            # Local search parameters
            sigma = 0.2 * np.max(rng_range) # Initial step size
            stagnation_limit = 20 + 2 * self.dim
            stagnation_counter = 0
            
            # Local search (1+1)-ES style loop
            while evals < self.budget:
                # Generate candidate
                noise = np.random.normal(0, 1, self.dim)
                candidate_x = curr_x + sigma * noise
                
                # Boundary handling: clip to search space
                candidate_x = np.clip(candidate_x, lb, ub)
                
                # Evaluate
                candidate_y = func(candidate_x)
                evals += 1
                
                # Greedy selection and adaptation
                if candidate_y <= curr_y:
                    # Success: move to new point and expand search radius
                    if candidate_y < curr_y:
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                        
                    curr_x = candidate_x
                    curr_y = candidate_y
                    sigma *= 1.1 # Expansion factor
                    
                    if curr_y < global_best_y:
                        global_best_y = curr_y
                        global_best_x = np.copy(curr_x)
                else:
                    # Failure: contract search radius
                    sigma *= 0.8 # Contraction factor
                    stagnation_counter += 1

                # Restart criteria: step size too small or too many failures
                if sigma < 1e-9 * np.max(rng_range) or stagnation_counter > stagnation_limit:
                    break

        return global_best_x, global_best_y
