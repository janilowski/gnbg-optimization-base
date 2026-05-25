import numpy as np
from collections import deque

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1)-Evolution Strategy with step-size adaptation by the 1/5-th success rule.
# Search state: a single parent vector (normalised to [0,1]^n) and a global step-size sigma.
# Candidate generation: isotropic Gaussian perturbation of the parent, sigma * N(0,I), then clamped to [0,1].
# Selection and replacement: greedy – the candidate replaces the parent only if it yields strictly better objective value (minimisation).
# Adaptation: after every `L` candidate evaluations, the fraction of successful mutations is computed. If it exceeds 0.2, sigma is increased (multiplied by e); if below 0.2, sigma is decreased (divided by e). If it equals 0.2, sigma is unchanged.
# Exploration mechanisms: large sigma permits long jumps; the 1/5-th rule raises sigma when exploration is too successful, thus avoiding premature convergence.
# Exploitation mechanisms: small sigma refines the current solution; the rule decreases sigma when too many mutations are rejected, focusing on local improvement.
# Boundary handling: coordinates are clamped to [0,1] after mutation, then linearly mapped to the original bounds supplied by the objective function.
# Budget strategy: exactly one evaluation per generation after the initial point; stops immediately when the evaluation budget is exhausted.
# Closest known influences: Rechenberg’s (1+1)-ES with the 1/5-th success rule (Evolutionstrategie, 1973).
# Novelty or unusual aspects: None; this is a straightforward implementation of a textbook algorithm.
# Failure modes: Multimodal or highly noisy problems – the greedy selection may cause premature convergence to a poor local optimum; the rule-based sigma adaptation can oscillate if the success rate fluctuates wildly.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        """
        Parameters:
            budget (int): maximum number of function evaluations.
            dim (int): dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        # step-size initial in normalised space
        self.sigma0 = 0.2
        # length of the success-rate window (heuristic)
        self.window_length = max(10, min(200, dim * 5))

    def __call__(self, func):
        """
        Run the (1+1)-ES on the given objective.

        Args:
            func: objective with attributes 'lower' / 'upper' or 'bounds.lb' / 'bounds.ub'.

        Returns:
            (best_x, best_y) where best_x is a 1-D numpy array in the original space,
            and best_y is the corresponding function value (minimisation).
        """
        # ----- read bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float).flatten()
            ub = np.asarray(func.upper, dtype=float).flatten()
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float).flatten()
            ub = np.asarray(func.bounds.ub, dtype=float).flatten()
        else:
            raise AttributeError("The objective must provide lower/upper bounds "
                                 "via 'lower'/'upper' or 'bounds.lb'/'bounds.ub'.")

        # ensure the arrays are the correct dimension
        if lb.ndim == 0:
            lb = np.array([lb.item()])
            ub = np.array([ub.item()])
        # normalise mapping: internal space [0,1] -> original space
        def to_original(x_norm):
            return lb + x_norm * (ub - lb)

        dim = self.dim
        budget = self.budget
        sigma = self.sigma0
        window = self.window_length

        # ----- initial point -----
        x = np.random.uniform(0.0, 1.0, size=dim)          # parent in normalised space
        best_x = x.copy()
        best_y = func(to_original(x))
        evals = 1                                          # one evaluation used

        # success tracking (deque acting as a sliding window)
        successes = deque(maxlen=window)

        # ----- main loop -----
        while evals < budget:
            # generate candidate
            y = x + sigma * np.random.randn(dim)
            y = np.clip(y, 0.0, 1.0)                      # boundary handling
            fy = func(to_original(y))
            evals += 1

            # selection
            success = 0
            if fy < best_y:                                # improvement (minimisation)
                best_y = fy
                best_x = y.copy()
                x = y.copy()
                success = 1

            # step-size adaptation via 1/5-th rule
            successes.append(success)
            if len(successes) == window:
                rate = sum(successes) / window
                if rate > 0.2:
                    sigma *= np.exp(1.0)                   # increase sigma
                elif rate < 0.2:
                    sigma *= np.exp(-1.0)                  # decrease sigma
                successes.clear()                          # restart the window

            # safeguard against numerical extinction
            if sigma < 1e-15:
                sigma = self.sigma0

        # map best point back to original space
        best_x_orig = to_original(best_x)
        return best_x_orig, best_y
