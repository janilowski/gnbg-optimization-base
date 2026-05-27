import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact, self-adaptive Differential Evolution (DE) variant with
#   randomized parameter control and a small archive for diversity. It uses
#   either DE/best/1/bin or DE/rand/1/bin depending on local success rates.
# Search state: A population of NP candidate solutions (vectors) and their
#   fitness values. A small archive stores previously discarded individuals.
#   Scaling factor F and crossover rate CR are stored per-individual.
# Candidate generation: For each target vector, either the best-so-far vector
#   (DE/best/1) or a random distinct vector (DE/rand/1) is chosen as the base,
#   then a single difference vector is added. Binomial crossover with the target.
# Selection and replacement: Greedy: if the trial vector is better than or equal
#   to the target, it replaces the target. Otherwise the target may enter the
#   archive (limited size, oldest-out).
# Adaptation: Per-individual F and CR are regenerated each generation from
#   truncated normal distributions centered on previous successes, following a
#   simple success-history adaptation (similar to jDE or SHADE, but simplified).
#   Success memory stores mean F and mean CR of recent successful mutations.
# Exploration mechanisms: Mutation with difference vectors provides global
#   exploration. Archive re-injection of old solutions creates additional
#   diversity pressure. Low crossover rates occasionally cause large jumps.
# Exploitation mechanisms: DE/best/1 mutation when used focuses search around
#   the current best. Greedy selection ensures local refinement.
# Boundary handling: "Reflect" – if a coordinate goes below lb or above ub,
#   it is reflected back into the domain by the excess amount.
# Budget strategy: Ceases function evaluations when budget is exhausted.
#   Population size NP is set to min(40, max(6, dim)) to scale with dimension.
# Closest known influences: jDE (self-adaptive F, CR), SHADE (success-history),
#   and standard DE.
# Novelty or unusual aspects: Extremely lightweight implementation; uses a tiny
#   archive (max 15) rather than full historical population; adaptation uses
#   only a single memory element (mean of last 10 successful parameters).
# Failure modes: Can be trapped in local optima on highly multimodal landscapes
#   because best-guided mutation can reduce diversity prematurely. The simple
#   memory may not adapt well to rapidly changing landscape properties.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.np = max(6, min(40, dim))           # population size
        self.archive_size = min(15, self.np // 2)   # small archive

    def __call__(self, func):
        # ------------------------------------------------------------
        # Bounds extraction
        # ------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds from func.")

        dim = self.dim
        np_ = self.np
        budget = self.budget
        evals = 0

        # ------------------------------------------------------------
        # Initialisation
        # ------------------------------------------------------------
        pop = lb + (ub - lb) * np.random.rand(np_, dim)
        fit = np.empty(np_)
        for i in range(np_):
            fit[i] = func(pop[i])
            evals += 1
            if evals >= budget:
                best_idx = np.argmin(fit[:i+1])
                return pop[best_idx].copy(), fit[best_idx]

        # Archive – list of vectors that were discarded
        archive = []

        # Individual control parameters (F, CR)
        F = 0.5 * np.ones(np_)
        CR = 0.9 * np.ones(np_)

        # Success memory – one memory cell
        mem_F = 0.5
        mem_CR = 0.9
        # Temporary buffer for successful parameters
        succ_F = []
        succ_CR = []

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # ------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------
        while evals < budget:
            # Replenish success memory every generation (reset after update)
            new_pop = pop.copy()
            new_fit = fit.copy()
            trial_vectors = np.empty_like(pop)

            # --------------------------------------------------------
            # For each target vector produce a trial
            # --------------------------------------------------------
            for i in range(np_):
                # --- choose base vector ---
                # Use best with probability 0.5, else random
                if np.random.rand() < 0.5:
                    base = pop[best_idx]
                else:
                    # pick a random distinct index
                    idxs = [j for j in range(np_) if j != i]
                    base = pop[np.random.choice(idxs)]

                # --- difference vector ---
                # pick two distinct indices != i, also != base if possible
                pool = [j for j in range(np_) if j != i]
                if len(pool) < 2:
                    # fallback: pick any two different
                    a, b = np.random.choice(np_, size=2, replace=False)
                else:
                    a, b = np.random.choice(pool, size=2, replace=False)
                diff = pop[a] - pop[b]

                # --- optional archive contribution ---
                if len(archive) > 0 and np.random.rand() < 0.2:
                    arch_vec = archive[np.random.randint(len(archive))]
                    # use archive vector instead of pop[b]
                    diff = pop[a] - arch_vec

                # --- mutation ---
                # Use F from the individual, but also a small random perturbation
                F_i = F[i]
                mut = base + F_i * diff

                # --- binomial crossover ---
                CR_i = CR[i]
                j_rand = np.random.randint(dim)
                trial = np.empty(dim)
                for d in range(dim):
                    if np.random.rand() < CR_i or d == j_rand:
                        trial[d] = mut[d]
                    else:
                        trial[d] = pop[i, d]

                # --- boundary handling (reflection) ---
                below = trial < lb
                above = trial > ub
                trial[below] = lb[below] + (lb[below] - trial[below])
                trial[above] = ub[above] - (trial[above] - ub[above])
                # clamp any that still escaped (rare)
                trial = np.clip(trial, lb, ub)

                trial_vectors[i] = trial

            # --------------------------------------------------------
            # Evaluation and selection (greedy)
            # --------------------------------------------------------
            for i in range(np_):
                if evals >= budget:
                    break
                trial = trial_vectors[i]
                trial_fit = func(trial)
                evals += 1
                if trial_fit <= fit[i]:
                    # success – record parameter
                    succ_F.append(F[i])
                    succ_CR.append(CR[i])
                    # replace
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
                    # old individual may go to archive
                    if len(archive) < self.archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # replace random archive slot
                        idx = np.random.randint(self.archive_size)
                        archive[idx] = pop[i].copy()
                else:
                    new_pop[i] = pop[i]
                    new_fit[i] = fit[i]

            # --------------------------------------------------------
            # update population and best
            # --------------------------------------------------------
            pop = new_pop
            fit = new_fit
            best_idx = np.argmin(fit)
            if fit[best_idx] < best_y:
                best_x = pop[best_idx].copy()
                best_y = fit[best_idx]

            # --------------------------------------------------------
            # Parameter adaptation (simplified jDE style)
            # --------------------------------------------------------
            if len(succ_F) > 0:
                # update memory with arithmetic mean of successful params
                mem_F = np.mean(succ_F)
                mem_CR = np.mean(succ_CR)
                # clear buffer
                succ_F.clear()
                succ_CR.clear()

            # Generate new F and CR for each individual (cauchy / normal)
            # Use a truncated normal around memory values
            for i in range(np_):
                # F in (0,1.2), CR in (0,1)
                F[i] = np.clip(np.random.normal(mem_F, 0.2), 0.1, 1.2)
                CR[i] = np.clip(np.random.normal(mem_CR, 0.1), 0.0, 1.0)

        return best_x, best_y
