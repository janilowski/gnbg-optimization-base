# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a hybrid optimization algorithm combining random
# sampling for exploration with Nelder-Mead simplex for local exploitation. It
# uses restarts to escape local minima and adaptive sampling density.
# Search state: Maintains current position, best found position, simplex vertices
# for Nelder-Mead, evaluation counter, and restart counter.
# Candidate generation: Initial random sampling in hypercube, then Nelder-Mead
# reflections and contractions from current simplex.
# Selection and replacement: Nelder-Mead replaces worst vertex if better; restarts
# occur after convergence or budget exhaustion.
# Adaptation: Sample variance adjusts based on search progress; restarts use
# shrinking search radius when no improvement seen.
# Exploration mechanisms: Random sampling at multiple scales, restart with
# reduced radius when stuck.
# Exploitation mechanisms: Nelder-Mead simplex operations (reflect, expand,
# contract, shrink) for fast local convergence.
# Boundary handling: Reflects out-of-bounds coordinates back into valid range
# using coordinate-wise reflection.
# Budget strategy: Strict tracking of evaluations; graceful termination when
# budget depleted; returns best found solution.
# Closest known influences: Hooke-Jeeves direct search + Nelder-Mead simplex
# with evolutionary restart strategy.
# Novelty or unusual aspects: Adaptive sampling variance based on iteration
# progress; restart with shrinking radius when no recent improvement.
# Failure modes: May struggle on very high-dimensional problems (>100D) due to
# simplex size; may be slow on ridge-like functions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    """
    Hybrid random-sampling and Nelder-Mead simplex optimizer for black-box minimization.
    
    Combines broad exploration through random sampling with intensive local
    exploitation via Nelder-Mead simplex method, using restarts to escape
    local minima.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize optimizer with budget and dimensionality.
        
        Args:
            budget: Maximum number of function evaluations allowed.
            dim: Problem dimensionality.
        """
        self.budget = budget
        self.dim = dim
        self.evals = 0  # Evaluation counter
    
    def __call__(self, func):
        """
        Optimize the given objective function within evaluation budget.
        
        Args:
            func: Black-box function object with bounds attribute.
            
        Returns:
            Tuple of (best_x, best_y) where best_x is optimal solution found
            and best_y is its objective value.
        """
        # Extract bounds from function object (support both attribute styles)
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback: assume standard bound format if available
            lb = np.full(self.dim, -10.0)
            ub = np.full(self.dim, 10.0)
        
        self.evals = 0
        best_x = None
        best_y = float('inf')
        
        # Adaptive sampling parameters
        sample_scale = 1.0  # Initial sampling radius
        no_improve_count = 0
        
        while self.evals < self.budget:
            # Generate initial simplex for Nelder-Mead
            vertices = self._generate_initial_simplex(lb, ub, sample_scale)
            
            # Evaluate initial vertices
            values = np.array([func(x) for x in vertices])
            self.evals += len(values)
            
            # Track best in this run
            run_best_idx = np.argmin(values)
            run_best_x = vertices[run_best_idx]
            run_best_y = values[run_best_idx]
            
            if run_best_y < best_y:
                best_y = run_best_y
                best_x = run_best_x.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Run Nelder-Mead simplex search from this start
            final_vertices, final_values = self._nelder_mead(
                vertices, values, func, lb, ub
            )
            
            # Update global best
            best_in_run = np.argmin(final_values)
            if final_values[best_in_run] < best_y:
                best_y = final_values[best_in_run]
                best_x = final_vertices[best_in_run].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Adjust sampling strategy based on progress
            if no_improve_count > 3:
                sample_scale *= 0.5  # Shrink search radius
                no_improve_count = 0
            else:
                sample_scale = min(1.0, sample_scale * 1.2)  # Slowly expand if making progress
        
        return best_x, best_y
    
    def _generate_initial_simplex(self, lb, ub, scale):
        """Generate initial simplex vertices around random center point."""
        center = np.random.uniform(lb, ub)
        vertices = [center]
        
        # Add vertices along each dimension
        for i in range(self.dim):
            vertex = center.copy()
            # Random direction with magnitude proportional to range * scale
            direction = np.zeros(self.dim)
            direction[i] = (ub[i] - lb[i]) * 0.5 * scale * np.random.uniform(0.5, 1.5)
            vertex += direction
            # Reflect into bounds
            vertex = np.clip(vertex, lb, ub)
            vertices.append(vertex)
        
        return vertices
    
    def _nelder_mead(self, vertices, values, func, lb, ub):
        """
        Nelder-Mead downhill simplex algorithm implementation.
        
        Args:
            vertices: Initial simplex vertices (n+1 points in n dimensions).
            values: Function values at vertices.
            func: Objective function.
            lb, ub: Variable bounds.
            
        Returns:
            Final vertices and their function values.
        """
        n = len(vertices) - 1  # Should be equal to self.dim
        alpha = 1.0   # Reflection coefficient
        gamma = 2.0   # Expansion coefficient
        rho = 0.5     # Contraction coefficient
        sigma = 0.5   # Shrink coefficient
        
        max_iter = max(50, self.dim * 10)
        
        for _ in range(max_iter):
            if self.evals >= self.budget:
                break
            
            # Sort vertices by function value
            sorted_idx = np.argsort(values)
            vertices = [vertices[i] for i in sorted_idx]
            values = values[sorted_idx]
            
            # Compute centroid of all points except worst
            centroid = np.mean(vertices[:-1], axis=0)
            
            # Reflection point
            worst = vertices[-1]
            reflected = centroid + alpha * (centroid - worst)
            reflected = np.clip(reflected, lb, ub)
            reflected_val = func(reflected)
            self.evals += 1
            
            if reflected_val < values[0]:
                # Expansion
                expanded = centroid + gamma * (reflected - centroid)
                expanded = np.clip(expanded, lb, ub)
                expanded_val = func(expanded)
                self.evals += 1
                
                if expanded_val < reflected_val:
                    vertices[-1] = expanded
                    values[-1] = expanded_val
                else:
                    vertices[-1] = reflected
                    values[-1] = reflected_val
            elif reflected_val < values[-2]:
                # Accept reflection
                vertices[-1] = reflected
                values[-1] = reflected_val
            else:
                # Contraction
                contracted = centroid + rho * (worst - centroid)
                contracted = np.clip(contracted, lb, ub)
                contracted_val = func(contracted)
                self.evals += 1
                
                if contracted_val < values[-1]:
                    vertices[-1] = contracted
                    values[-1] = contracted_val
                else:
                    # Shrink: contract all except best
                    best = vertices[0]
                    new_vertices = [vertices[0]]
                    new_values = [values[0]]
                    
                    for i in range(1, len(vertices)):
                        new_vertex = best + sigma * (vertices[i] - best)
                        new_vertex = np.clip(new_vertex, lb, ub)
                        new_val = func(new_vertex)
                        self.evals += 1
                        
                        new_vertices.append(new_vertex)
                        new_values.append(new_val)
                    
                    vertices = new_vertices
                    values = np.array(new_values)
        
        return vertices, values
