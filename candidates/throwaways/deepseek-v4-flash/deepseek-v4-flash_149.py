import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE) with dither and binomial crossover. Simple, robust black‑box optimizer.
# Search state: A population of NP candidate vectors and their current fitness values.
# Candidate generation: For each target vector, three distinct random vectors a,b,c are chosen.
#   The mutant is a + F*(b-c) with F sampled uniformly per individual in [0.5,1.0] (dither).
#   Binomial crossover (CR=0.9) produces the trial vector.
# Selection and replacement: Greedy replacement – the trial replaces the target if it has lower (better)
#   objective function value (minimization).
# Adaptation: No explicit parameter adaptation; F dithering provides light dynamic behavior.
# Exploration mechanisms: High crossover probability and differential mutation encourage exploration.
# Exploitation mechanisms: Population gradually converges; greedy selection preserves the best so far.
# Boundary handling: If any component of the trial leaves the search domain, it is re‑initialized
#   uniformly at random inside the bounds.
# Budget strategy: Population size NP is set as NP = max(4, min(budget‑1, 50, budget//10)).
#   The algorithm runs full generations while total evaluations + NP <= budget. Remaining evaluations
#   are left unused – the algorithm stops when the next generation would exceed the budget.
# Closest known influences: Classic DE/rand/1/bin with uniform jitter (dither) on the scaling factor.
# Novelty or unusual aspects: None – this is a straightforward, compact implementation.
# Failure modes: On highly multimodal or deceptive landscapes, may converge to a local optimum
#   if the budget is insufficient for global search.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Differential Evolution (DE) with dither for GNBG black‑box minimization."""
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Population size heuristic – scaled to budget, capped, at least 4.
        self.NP = max(4, min(self.budget - 1, 50, self.budget // 10))
        # For very small budgets, fallback to pure random search
        self.random_search = (self.budget < 4)
        
    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.array(func.lower, dtype=float)
            upper = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lower = np.array(func.bounds.lb, dtype=float)
            upper = np.array(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds; expected func.lower/upper or func.bounds.lb/ub")
        
        dim = self.dim
        budget = self.budget
        NP = self.NP

        # Helper: evaluate without exceeding budget
        evals = 0
        best_x = None
        best_y = np.inf

        def evaluate(x):
            nonlocal evals, best_x, best_y
            if evals >= budget:
                return None   # signal that budget exhausted
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y
        
        # Fallback for extremely small budgets
        if self.random_search:
            # just random sampling, avoid DE overhead
            while evals < budget:
                x = lower + np.random.rand(dim) * (upper - lower)
                evaluate(x)
            return best_x, best_y

        # Initialize population
        pop = lower + np.random.rand(NP, dim) * (upper - lower)
        fit = np.full(NP, np.inf)
        for i in range(NP):
            v = pop[i]
            evaluate(v)
            # if budget exhausted during init, we should stop
            # but it's unlikely because NP <= budget-1
            fit[i] = func(v)
            evals += 1
            if fit[i] < best_y:
                best_y = fit[i]
                best_x = v.copy()
        
        # Main DE loop
        while evals < budget:
            # Do not start a new generation if fewer than NP evaluations remain
            if evals + NP > budget:
                break
            for i in range(NP):
                if evals >= budget:
                    break
                # Generate three distinct indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                
                # Mutation with dither: F per individual
                F = 0.5 + 0.5 * np.random.rand()
                mutant = pop[a] + F * (pop[b] - pop[c])
                
                # Binomial crossover
                CR = 0.9
                j_rand = np.random.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Boundary handling – re‑initialize out‑of‑bounds components uniformly
                for j in range(dim):
                    if trial[j] < lower[j] or trial[j] > upper[j]:
                        trial[j] = lower[j] + np.random.rand() * (upper[j] - lower[j])
                
                # Evaluate trial and select
                f_trial = evaluate(trial)
                if f_trial is None:   # budget exhausted during evaluation
                    break
                if f_trial < fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
        
        return best_x, best_y
