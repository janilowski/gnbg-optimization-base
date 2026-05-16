# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An archive-based local searcher with global restarts. It maintains a set of elite solutions and iteratively refines them using Gaussian mutation with a decaying step size.
# Search state: A sorted list (archive) of the best (value, position) pairs found, and a local counter for evaluations.
# Candidate generation: New points are generated either by uniform sampling (for exploration and restarts) or by adding Gaussian noise to a randomly selected elite member from the archive.
# Selection and replacement: The algorithm uses a simple archive management where the best solutions found across the entire search are kept in a fixed-size list.
# Adaptation: The mutation strength (sigma) follows a power-law decay based on the fraction of the budget consumed, facilitating a transition from broad exploration to fine-grained local refinement.
# Exploration mechanisms: Initial uniform sampling and a 10% stochastic probability of uniform sampling at any step (global refreshes) to prevent stagnation in local optima.
# Exploitation mechanisms: Focused Gaussian mutation centered on the current best-known points in the search space.
# Boundary handling: All proposed candidates are clipped to the hyper-rectangular bounds defined by the problem.
# Budget strategy: The search loop iterates strictly until the evaluation counter reaches the provided budget, ensuring no overruns.
# Closest known influences: Evolutionary Strategies (ES) and Archive-based Micro-Genetic Algorithms.
# Novelty or unusual aspects: A lean, stateless mutation decay synchronized with the remaining budget, requiring minimal hyperparameter tuning.
# Failure modes: May struggle with extremely high-dimensional landscapes where Gaussian mutation without covariance adaptation is inefficient, or highly deceptive landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the algorithm with a budget and dimensionality.
        """
        self.budget = budget
        self.dim = dim
        self.archive_size = 10

    def __call__(self, func):
        """
        Executes the optimization process on the provided function.
        """
        # Extract bounds from the function object
        if hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.atleast_1d(func.bounds.lb)
            ub = np.atleast_1d(func.bounds.ub)
        elif hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.atleast_1d(func.lower)
            ub = np.atleast_1d(func.upper)
        else:
            # Default to unit hypercube if bounds are missing
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)
        
        range_ = ub - lb
        archive = [] # List of (y, x)
        best_x = None
        best_y = float('inf')
        evals = 0

        while evals < self.budget:
            # Determine search mode: exploration (global) or exploitation (archive-based)
            # We use global sampling if the archive is empty or by chance (10% rate)
            do_global = (len(archive) == 0) or (np.random.rand() < 0.1)
            
            if do_global:
                # Global uniform sample across the search space
                x = lb + np.random.rand(self.dim) * range_
            else:
                # Local search: perturb an existing elite solution
                # Pick a random elite from the archive
                idx = np.random.randint(len(archive))
                _, parent_x = archive[idx]
                
                # Calculate progress-based mutation strength
                progress = evals / self.budget
                # Step size decays from 30% of the range down to a small fraction
                scale = 0.3 * (1.0 - progress)**2
                noise = np.random.standard_normal(self.dim) * scale * range_
                x = parent_x + noise
            
            # Boundary handling: clip candidate to hypercube
            x = np.clip(x, lb, ub)
            
            # Evaluate the candidate
            y = func(x)
            evals += 1
            
            # Update the global best tracker
            if y < best_y:
                best_y = y
                best_x = x.copy()
            
            # Update the elite archive
            archive.append((y, x.copy()))
            archive.sort(key=lambda t: t[0])
            if len(archive) > self.archive_size:
                archive.pop()
        
        return best_x, best_y
