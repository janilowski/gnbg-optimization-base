# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1)-Evolution Strategy with cumulative step-size adaptation (1/5th rule) for black-box minimization.
# Search state: A single candidate solution (the current best) and a global step size sigma.
# Candidate generation: Gaussian mutation: new candidate = current best + sigma * randn(dim), scaled by the typical range of the domain (upper-lower bound difference).
# Selection and replacement: Greedy replacement: accept the new candidate if it yields a lower function value (minimization).
# Adaptation: Step-size control using Rechenberg's 1/5 rule: after every adaptation interval (ceil(dim/2) evaluations), compute the success rate over that interval; if rate > 0.2, multiply sigma by 1.2; if rate < 0.2, multiply by 0.85.
# Exploration mechanisms: Gaussian mutations with step size adapted to maintain a target success rate, enabling a balance between exploration and exploitation.
# Exploitation mechanisms: The greedy selection forces convergence toward better regions.
# Boundary handling: Candidate solutions are clipped to the domain bounds. No explicit repair or reflection is applied, which is simple but may bias adaptation.
# Budget strategy: Runs a single chain of mutations until the evaluation budget is exhausted; no restarts or multi-starts.
# Closest known influences: Rechenberg's (1+1)-Evolution Strategy with 1/5 rule (ca. 1973).
# Novelty or unusual aspects: None; implementation is a minimal textbook ES.
# Failure modes: May stagnate on strongly multimodal or non-separable problems due to greedy selection; clipping at bounds can distort mutation distribution; fixed step-size adaptation may not suit all landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Adaptation interval (number of evaluations between sigma updates)
        self.adapt_interval = max(1, int(np.ceil(dim / 2.0)))
        # Step-size scaling factor relative to the domain range; will be set during run
        self.sigma_factor = 0.2  # initial sigma = sigma_factor * domain_range

    def __call__(self, func):
        # Retrieve bounds (support both naming conventions)
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # assumes func.bounds has .lb and .ub
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Function does not expose bounds via lower/upper or bounds.lb/ub")
        dim = self.dim
        # Ensure bounds are 1D arrays of length dim
        if lb.ndim == 0:
            lb = np.full(dim, lb)
        if ub.ndim == 0:
            ub = np.full(dim, ub)
        domain_range = ub - lb
        # Initialise current solution uniformly in bounds
        x = lb + np.random.rand(dim) * domain_range
        y = func(x)
        evals = 1
        best_x = x.copy()
        best_y = y
        # Step size
        sigma = self.sigma_factor * np.mean(domain_range)  # single global step
        # Adaptation state
        successes = 0
        adapt_counter = 0

        while evals < self.budget:
            # Generate candidate by Gaussian mutation
            # Scale mutation by sigma, but also clamp to domain range
            dx = sigma * np.random.randn(dim)
            candidate = x + dx
            # Boundary handling: clip to bounds
            candidate = np.clip(candidate, lb, ub)
            # Evaluate
            cand_y = func(candidate)
            evals += 1
            # Selection: keep if better (minimization)
            if cand_y < y:
                x = candidate
                y = cand_y
                successes += 1
                # Update global best if better
                if cand_y < best_y:
                    best_x = candidate.copy()
                    best_y = cand_y
            # Adaptation step size every adapt_interval evaluations (including this one)
            adapt_counter += 1
            if adapt_counter >= self.adapt_interval:
                success_rate = successes / adapt_counter if adapt_counter > 0 else 0.0
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.85
                # Reset counters
                successes = 0
                adapt_counter = 0
            # Avoid overly large or small sigma
            sigma = np.clip(sigma, 1e-10 * np.mean(domain_range), 0.5 * np.mean(domain_range))

        return best_x, best_y
