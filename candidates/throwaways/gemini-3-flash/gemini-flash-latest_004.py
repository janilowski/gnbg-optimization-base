# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A multi-start adaptive (1+1)-Evolution Strategy (ES). It combines global exploration via random restarts and local exploitation via success-based step-size adaptation.
# Search state: Current local iterate (x_curr), current local objective value (y_curr), current step size (sigma), and the global best found so far.
# Candidate generation: Gaussian perturbation centered at the current local iterate: x_new = x_curr + sigma * N(0, I).
# Selection and replacement: Greedy selection; the candidate replaces the current iterate only if it results in a lower (or equal) objective value.
# Adaptation: Step size (sigma) is adjusted based on a simplified 1/5th success rule. Sigma increases on success and decreases on failure to maintain an optimal search pressure.
# Exploration mechanisms: Random restarts are triggered whenever the local step size becomes negligibly small or a local stagnation criterion is met, ensuring different regions of the search space are sampled.
# Exploitation mechanisms: The (1+1) structure behaves like a stochastic hill-climber, focusing the search around promising areas with decreasing step sizes for fine-tuning.
# Boundary handling: Candidates are clipped to the hyper-box defined by the problem bounds before evaluation.
# Budget strategy: The algorithm tracks evaluations strictly, terminating immediately when the budget is exhausted. Evaluations are distributed across multiple local search trajectories.
# Closest known influences: (1+1)-ES, Luus-Jaakola heuristic, Random Search.
# Novelty or unusual aspects: Uses a robust bound-detection logic to accommodate varying benchmark API styles (e.g., direct attributes vs nested bounds objects).
# Failure modes: May struggle with highly deceptive landscapes or extremely high-dimensional spaces where the budget is insufficient to allow for enough restarts.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the optimization algorithm.
        
        :param budget: Total number of function evaluations allowed.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.evals_count = 0
        self.best_x = None
        self.best_y = float('inf')

    def __call__(self, func):
        """
        Executes the optimization process.
        
        :param func: The objective function to minimize.
        :return: (best_x, best_y) found during the search.
        """
        # 1. Bounds extraction
        lb, ub = self._get_bounds(func)
        range_width = ub - lb
        
        # 2. Main Search Loop (Multi-start)
        while self.evals_count < self.budget:
            # Random Restart
            x_curr = lb + (ub - lb) * np.random.rand(self.dim)
            y_curr = self._safe_eval(func, x_curr)
            if self.evals_count >= self.budget:
                break
            
            # Local Search Parameters (1+1-ES style)
            sigma = 0.2 * np.max(range_width)  # Initial step size
            sigma_min = 1e-9 * np.min(range_width) if np.min(range_width) > 0 else 1e-12
            
            # Success adaptation factors
            inc = 1.5
            dec = inc**(-0.25) # 1/5th rule approximation
            
            # 3. Local Search Loop
            stagnation_counter = 0
            max_stagnation = 20 + 2 * self.dim
            
            while self.evals_count < self.budget and sigma > sigma_min:
                # Generate candidate
                noise = np.random.randn(self.dim)
                x_new = np.clip(x_curr + sigma * noise, lb, ub)
                
                y_new = self._safe_eval(func, x_new)
                if self.evals_count >= self.budget:
                    break
                
                if y_new <= y_curr:
                    # Success
                    if y_new < y_curr:
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                        
                    x_curr, y_curr = x_new, y_new
                    sigma *= inc
                else:
                    # Failure
                    sigma *= dec
                    stagnation_counter += 1
                
                # Check for local stagnation to trigger early restart
                if stagnation_counter > max_stagnation:
                    break
                    
        return self.best_x, self.best_y

    def _safe_eval(self, func, x):
        """Evaluates the function and updates the global best state."""
        if self.evals_count >= self.budget:
            return float('inf')
        
        y = func(x)
        self.evals_count += 1
        
        if y < self.best_y:
            self.best_y = y
            self.best_x = np.copy(x)
            
        return y

    def _get_bounds(self, func):
        """Robustly extracts lower and upper bounds from the function object."""
        lb = None
        ub = None
        
        # Strategy A: func.lower / func.upper
        if hasattr(func, 'lower') and func.lower is not None:
            lb = func.lower
            ub = func.upper
        # Strategy B: func.bounds.lb / func.bounds.ub
        elif hasattr(func, 'bounds') and func.bounds is not None:
            lb = func.bounds.lb
            ub = func.bounds.ub
            
        # Ensure they are numpy arrays
        lb = np.atleast_1d(lb).astype(float)
        ub = np.atleast_1d(ub).astype(float)
        
        # Handle scalar broadcast to dimension
        if lb.shape[0] == 1 and self.dim > 1:
            lb = np.full(self.dim, lb[0])
        if ub.shape[0] == 1 and self.dim > 1:
            ub = np.full(self.dim, ub[0])
            
        return lb, ub

# End of Algorithm implementation.
