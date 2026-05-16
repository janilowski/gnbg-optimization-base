# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A hybrid optimization strategy combining Opposition-Based Learning (OBL) for global exploration and an adaptive (1+1) Evolution Strategy for local exploitation.
# Search state: Tracks the current best position (best_x), its objective value (best_y), and a vector of mutation step sizes (sigma).
# Candidate generation: Initial candidates are generated as random uniform samples paired with their "opposites" (mirror points across the search space center). Subsequent candidates are produced via Gaussian mutation of the current best point.
# Selection and replacement: Uses a strict elitist selection rule where a candidate replaces the current best only if it yields a strictly lower objective value.
# Adaptation: The mutation step size (sigma) is adapted using a simplified success-based rule: expanding by 10% after a successful improvement and contracting by 5% after a failure.
# Exploration mechanisms: Exploration is primarily handled in the first phase by OBL, which samples both random points and their point-reflections to increase the probability of landing in the global optimum's basin.
# Exploitation mechanisms: Exploitation is performed by a greedy local hill-climber with an adaptive step size that narrows down on the optima.
# Boundary handling: All generated points are clipped to the feasible hypercube defined by the problem bounds.
# Budget strategy: Roughly 20% of the budget is allocated to the opposition-based sampling phase, with the remainder dedicated to local refinement.
# Closest known influences: Opposition-Based Learning (OBL) and (1+1)-ES with the 1/5th success rule logic.
# Novelty or unusual aspects: Combines OBL's symmetry-breaking properties with a simple adaptive local search in a compact, dependency-minimal implementation.
# Failure modes: May prematurely converge if the initial sampling phase fails to find the basin of the global optimum in highly non-convex, high-dimensional landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    A black-box minimization algorithm utilizing Opposition-Based Learning 
    for initialization followed by an Adaptive Local Search.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Handle bounds extraction from different possible attribute names
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Default fallback if bounds are missing (not expected in GNBG)
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)

        evals_done = 0
        best_x = (lb + ub) / 2.0
        best_y = float('inf')

        def evaluate(x):
            nonlocal evals_done, best_x, best_y
            if evals_done >= self.budget:
                return None
            
            # Ensure candidate is within bounds
            x_clipped = np.clip(x, lb, ub)
            y = func(x_clipped)
            evals_done += 1
            
            if y < best_y:
                best_y = y
                best_x = np.copy(x_clipped)
                return y, True
            return y, False

        # Phase 1: Opposition-Based Initialization
        # We use approx 20% of budget or a fixed multiple of dimension for OBL
        n_init = min(self.budget // 5, 2 * self.dim + 10)
        
        # Always evaluate the center first if we have budget
        if self.budget > 0:
            evaluate(best_x)

        # Generate pairs (random + opposite)
        for _ in range(n_init // 2):
            if evals_done >= self.budget:
                break
            
            p_rand = np.random.uniform(lb, ub)
            evaluate(p_rand)
            
            if evals_done < self.budget:
                # Opposite point calculation: lb + ub - x
                p_opp = lb + ub - p_rand
                evaluate(p_opp)

        # Phase 2: Adaptive Local Search (Greedy 1+1 ES flavor)
        # Initialize sigma as a fraction of the search range
        sigma = 0.1 * (ub - lb)
        
        while evals_done < self.budget:
            # Generate candidate using Gaussian mutation
            noise = np.random.standard_normal(self.dim) * sigma
            candidate = best_x + noise
            
            result = evaluate(candidate)
            if result is None:
                break
                
            _, improved = result
            
            # Simple adaptation logic:
            # If we improved, try to move further (expand sigma)
            # If we failed, focus the search (contract sigma)
            if improved:
                sigma *= 1.1
            else:
                sigma *= 0.95
            
            # Prevent sigma from becoming too small or too large relative to the domain
            sigma = np.clip(sigma, 1e-9 * (ub - lb), 0.5 * (ub - lb))

        return best_x, best_y
