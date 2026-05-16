# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A population-based archive search that combines adaptive Gaussian mutation around elite solutions with periodic global restarts.
# Search state: Maintains an archive of the best-performing parameter vectors and their objective values, alongside a progress-dependent mutation scale.
# Candidate generation: Candidates are generated either via uniform sampling across the search space (exploration/refresh) or via Gaussian perturbations centered on a randomly selected member of the elite archive (exploitation).
# Selection and replacement: The algorithm uses truncation selection, maintaining a fixed-size archive of the best individuals found so far, updated every time a new candidate outperforms an existing archive member.
# Adaptation: The mutation step size (sigma) decreases non-linearly as the evaluation budget is consumed, transitioning from broad exploration to fine-grained local refinement.
# Exploration mechanisms: Exploration is driven by a constant probability of global random sampling and a high initial mutation variance.
# Exploitation mechanisms: Exploitation is achieved by sampling increasingly closer to the elite solutions in the archive and an occasional crossover-like operation with the current best solution.
# Boundary handling: Solutions are strictly enforced within the search space using clipping to the lower and upper bounds.
# Budget strategy: The algorithm tracks evaluations and terminates exactly when the budget is exhausted, ensuring the best result found is returned.
# Closest known influences: Evolutionary Strategies (ES), basic Archive-based Local Search, and elements of Differential Evolution (crossover).
# Novelty or unusual aspects: A lightweight, self-contained implementation that balances global refreshes with archive-based refinement without requiring complex covariance matrix updates.
# Failure modes: May struggle with extremely high-dimensional landscapes where the archive cannot sufficiently cover the search space, or on highly deceptive functions where local optima are far from the global minimum.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the optimization algorithm.
        
        Args:
            budget (int): Total number of function evaluations allowed.
            dim (int): Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.evals_done = 0

    def __call__(self, func):
        """
        Executes the optimization process.
        
        Args:
            func (callable): The objective function to minimize.
            
        Returns:
            tuple: (best_x, best_y) found during the search.
        """
        # Extract bounds from the function object
        if hasattr(func, 'lower') and func.lower is not None:
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            # Fallback to defaults if bounds are not provided
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        span = ub - lb
        
        # Hyperparameters
        elite_size = max(3, min(20, self.dim * 2))
        global_sample_prob = 0.15
        initial_sigma = 0.2
        
        # State initialization
        archive_x = []
        archive_y = []
        best_x = None
        best_y = float('inf')

        while self.evals_done < self.budget:
            # Decide whether to sample globally (refresh) or locally (exploit)
            if len(archive_x) < 2 or np.random.rand() < global_sample_prob:
                # Global random sampling
                candidate = lb + np.random.rand(self.dim) * span
            else:
                # Local sampling around an elite
                # Select an elite using a bias towards the best (rank-based)
                weights = np.arange(len(archive_y), 0, -1)
                probs = weights / weights.sum()
                idx = np.random.choice(len(archive_x), p=probs)
                parent = archive_x[idx]
                
                # Adaptive step size based on budget consumption
                progress = self.evals_done / self.budget
                # Sigma decays from initial_sigma down to a small fraction
                current_sigma = initial_sigma * (0.9 * (1.0 - progress)**1.5 + 0.1)
                
                # Generate offspring with Gaussian mutation
                candidate = parent + np.random.randn(self.dim) * current_sigma * span
                
                # Occasional discrete crossover with the best found so far
                if np.random.rand() < 0.2:
                    mask = np.random.rand(self.dim) < 0.3
                    candidate[mask] = archive_x[0][mask]

            # Boundary handling: clip to search space
            candidate = np.clip(candidate, lb, ub)

            # Evaluation
            y = func(candidate)
            self.evals_done += 1

            # Update archive and global best
            if best_x is None or y < best_y:
                best_y = y
                best_x = candidate.copy()

            # Maintain the archive of elite solutions
            archive_x.append(candidate)
            archive_y.append(y)
            
            # Sort archive by objective value and truncate to elite_size
            if len(archive_x) > elite_size * 2:
                indices = np.argsort(archive_y)
                archive_x = [archive_x[i] for i in indices[:elite_size]]
                archive_y = [archive_y[i] for i in indices[:elite_size]]

        return best_x, best_y

# The harness typically sets the seed for numpy before calling the algorithm.
# The class Algorithm satisfies the public interface requirements.
