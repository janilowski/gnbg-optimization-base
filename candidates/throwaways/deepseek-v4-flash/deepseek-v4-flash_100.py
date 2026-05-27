# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a (1+1)-Evolution Strategy with step-size adaptation
# for black-box minimization. It is designed for the GNBG benchmark and respects
# the evaluation budget. It reads bounds from the objective function object.
# Search state: a single current solution vector and a global step size.
# Candidate generation: Gaussian mutation scaled by the current step size.
# Selection and replacement: the new candidate replaces the current one if it
# improves the objective value (elitist selection). The step size is adapted
# using Rechenberg's 1/5 success rule: after a learning period, if the success
# rate is above 1/5, the step size is increased; if below, it is decreased.
# Adaptation: step size is adjusted multiplicatively using factors
# (exp(+d) or exp(-d)) based on the observed success rate over a sliding window.
# Exploration mechanisms: mutation with a Gaussian distribution allows
# exploration; step-size adaptation tunes the exploration range.
# Exploitation mechanisms: elitist selection and step-size reduction when
# success rate is low focus the search locally.
# Boundary handling: reflections off the domain boundaries to keep all
# candidate points inside the feasible region.
# Budget strategy: the callable counts function evaluations and stops when the
# budget is exhausted; the best solution found so far is returned.
# Closest known influences: (1+1)-ES with Rechenberg's step-size adaptation.
# Novelty or unusual aspects: None; this is a standard implementation.
# Failure modes: may converge prematurely to local optima if step size becomes
# too small; the success‑rate window may cause oscillation; may perform poorly
# on highly non‑separable or multi‑modal problems.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Minimizer using (1+1)-ES with step-size adaptation."""
    
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        
        # Step-size adaptation parameters (Rechenberg's 1/5 rule)
        self.learning_rate = 0.1          # d in exp(±d)
        self.eval_window = 50             # window size for success rate
        self.success_target = 1.0 / 5.0   # target success probability
        self.success_count = 0
        self.eval_count_in_window = 0
        
    def __call__(self, func):
        # Read bounds
        try:
            lb = np.asarray(func.lower, dtype=np.float64)
            ub = np.asarray(func.upper, dtype=np.float64)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=np.float64) if hasattr(func, 'bounds') else -np.ones(self.dim) * 5.0
            ub = np.asarray(func.bounds.ub, dtype=np.float64) if hasattr(func, 'bounds') else np.ones(self.dim) * 5.0
        
        # Domain width for step size initialization
        domain_width = ub - lb
        
        # Initialize point uniformly in the domain
        x = lb + np.random.rand(self.dim) * domain_width
        y = func(x)
        evals = 1
        
        # Initial step size: 20% of the smallest domain dimension
        sigma = 0.2 * np.min(domain_width)
        if sigma <= 0.0:
            sigma = 1.0  # fallback if all dimensions are same bound
        
        best_x = x.copy()
        best_y = y
        
        # Reset success tracking
        self.success_count = 0
        self.eval_count_in_window = 0
        
        while evals < self.budget:
            # Generate offspring: Gaussian mutation
            z = np.random.randn(self.dim)
            offspring = x + sigma * z
            
            # Reflect at boundaries
            for d in range(self.dim):
                while offspring[d] < lb[d] or offspring[d] > ub[d]:
                    if offspring[d] < lb[d]:
                        offspring[d] = 2.0 * lb[d] - offspring[d]
                    if offspring[d] > ub[d]:
                        offspring[d] = 2.0 * ub[d] - offspring[d]
            
            # Evaluate
            f_off = func(offspring)
            evals += 1
            
            # Selection (minimization)
            if f_off < y:
                x = offspring
                y = f_off
                self.success_count += 1
                # Update best
                if y < best_y:
                    best_x = x.copy()
                    best_y = y
            # else: discard offspring, no success
            
            self.eval_count_in_window += 1
            
            # Step-size adaptation every window evaluations
            if self.eval_count_in_window >= self.eval_window:
                success_rate = self.success_count / float(self.eval_window)
                # Adapt sigma
                if success_rate > self.success_target:
                    sigma *= np.exp(self.learning_rate)
                else:
                    sigma *= np.exp(-self.learning_rate)
                # Reset window
                self.success_count = 0
                self.eval_count_in_window = 0
        
        return best_x, best_y
