import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (μ+λ)-Evolution Strategy with self-adaptive per-dimension step sizes.
#   Designed for the GNBG black‑box minimization benchmark.  Uses intermediate
#   recombination and uncorrelated log‑normal mutation.  Maintains elitism by
#   keeping the best μ individuals after selection.
# Search state: A set of μ parent vectors (candidates) and their step‑size vectors.
#   Each generation λ offspring are sampled from the parent pool.
# Candidate generation: For each parent, select a parent uniformly at random but
#   with weighted recombination: a new candidate is the average of μ/2 randomly
#   chosen parents (intermediate recombination), then mutated by adding z*sigma
#   where z ~ N(0, I) and sigma is the step‑size vector of that reconstructed parent.
#   Offspring step sizes are mutated from the parent step sizes using the log‑normal
#   rule: sigma' = sigma * exp(tau * N(0,1) + tau' * N(0,1) per dimension).
# Selection and replacement: (μ+λ) selection: combine parents and offspring,
#   sort by fitness, keep the best μ individuals.
# Adaptation: Step sizes adapt via self‑adaptation: the log‑normal mutation.
#   Parameters tau = 1/sqrt(2*d) and tau' = 1/sqrt(2*sqrt(d)) for per‑dim and overall.
# Exploration mechanisms: Large mutation steps when step sizes are large, and
#   recombination mixes genetic material.
# Exploitation mechanisms: Intermediate recombination centers the population,
#   elitist selection preserves the best solutions.
# Boundary handling: Reflection at the bounds. If a component leaves [lb, ub],
#   it is reflected back into the domain (mirror effect).
# Budget strategy: The algorithm stops after using exactly budget evaluations.
#   It does not overshoot; the last candidate is evaluated only if budget remains.
# Closest known influences: Classic ES with self‑adaptive step sizes (Schwefel,
#   Beyer, etc.), similar to the (μ/μ_I,λ)-ES but with (μ+λ).
# Novelty or unusual aspects: None; it is a straightforward robust ES.
# Failure modes: May converge slowly on highly multimodal or ill‑conditioned
#   landscapes.  Step sizes can shrink prematurely if population is too small
#   or recombination reduces diversity too quickly.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # ES parameters (chosen to be robust for small dim, e.g., dim <= 30)
        self.mu = max(4, dim)          # parent population
        self.lam = max(4 * self.mu, budget // 4)  # offspring per generation (at least 4*mu)
        if self.lam < 2:
            self.lam = 2
        # Ensure we don't overshoot the budget too much: we'll limit per‑generation
        self.lam = min(self.lam, budget)  # but budget may be huge, so we keep lam manageable
        # Adaptation parameters
        self.tau = 1.0 / np.sqrt(2.0 * dim)
        self.tau_prime = 1.0 / np.sqrt(2.0 * np.sqrt(dim))
        # Step size boundaries to avoid numerical issues
        self.sigma_min = 1e-8
        self.sigma_max = 1e2

    def __call__(self, func):
        # Read bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # assume bounds.lb, bounds.ub
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds in func")
        bounds = np.vstack([lb, ub]).T   # (dim, 2)

        dim = self.dim
        # Initialize parent population
        # population: list of (x, sigma, fitness)
        pop = []
        for _ in range(self.mu):
            x = np.random.uniform(lb, ub, size=dim)
            sigma = np.full(dim, (ub - lb) / 4.0)  # initial step size ~ 1/4 domain width
            pop.append((x, sigma, None))
        # Evaluate initial population
        evals = 0
        for i in range(self.mu):
            x = pop[i][0]
            y = func(x)
            evals += 1
            pop[i] = (x, pop[i][1], y)
        # Sort by fitness
        pop.sort(key=lambda t: t[2])

        best_x = pop[0][0].copy()
        best_y = pop[0][2]

        # Helper: reflect component into bounds
        def reflect(z, lb_, ub_):
            if z < lb_:
                return lb_ + (lb_ - z)
            elif z > ub_:
                return ub_ - (z - ub_)
            else:
                return z

        # Main loop
        while evals < self.budget:
            # Offspring generation
            offspring = []
            # How many offspring can we still produce without exceeding budget?
            remaining = self.budget - evals
            # We enforce that we don't exceed budget, so limit lam to remaining
            lam_eff = min(self.lam, remaining)

            if lam_eff <= 0:
                break

            for _ in range(lam_eff):
                # Select recombination parents: choose mu/2 distinct parents uniformly
                # For simplicity use intermediate recombination of mu/2 individuals
                r = np.random.choice(range(self.mu), size=max(1, self.mu//2), replace=False)
                # Weighted average
                p_x = np.mean([pop[idx][0] for idx in r], axis=0)
                p_sigma = np.mean([pop[idx][1] for idx in r], axis=0)

                # Mutate step size
                # Global step factor (one for all dimensions)
                tau_g = self.tau_prime
                tau_d = self.tau
                # N(0,1) for global step, per dimension for local
                global_noise = np.random.randn()
                per_dim_noise = np.random.randn(dim)

                sigma_new = p_sigma * np.exp(
                    tau_g * global_noise + tau_d * per_dim_noise
                )
                # Clip to reasonable range
                sigma_new = np.clip(sigma_new, self.sigma_min, self.sigma_max)
                # Step size boundary reflection is not needed, just clamp

                # Mutate position
                z = np.random.randn(dim) * sigma_new
                x_new = p_x + z

                # Boundary handling: reflect
                for d_ in range(dim):
                    l, u = bounds[d_]
                    x_new[d_] = reflect(x_new[d_], l, u)

                # Evaluate (if budget permits – we will only evaluate if we haven't exhausted budget)
                y_new = func(x_new)
                evals += 1
                # Keep track of best
                if y_new < best_y:
                    best_x = x_new.copy()
                    best_y = y_new

                offspring.append((x_new, sigma_new, y_new))

                # Check if we exceed budget in this generation after this evaluation
                if evals >= self.budget:
                    break

            # Combine parents and offspring
            combined = pop + offspring
            # Sort by fitness
            combined.sort(key=lambda t: t[2])
            # Select best mu
            pop = combined[:self.mu]

            # Update best from new parents (already tracked, but do for safety)
            if pop[0][2] < best_y:
                best_x = pop[0][0].copy()
                best_y = pop[0][2]

        return best_x, best_y
