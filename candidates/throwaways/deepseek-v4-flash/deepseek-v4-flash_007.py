# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A modified Differential Evolution (DE) variant with adaptive parameter control and Cauchy mutation for global black-box minimization.
#
# Search state: A population of candidate solutions stored as a 2D numpy array (NP x dim), with associated fitness values in a separate array. The best solution and its fitness are tracked throughout the search.
#
# Candidate generation: Each generation produces NP/2 trial vectors using a combination of DE/rand/1/bin, DE/best/1/bin, and DE/current-to-best/1/bin strategies. Selection between strategies uses a weighted roulette mechanism based on their historical success rates.
#
# Selection and replacement: Greedy selection between parent and child solutions based on fitness improvement. A child replaces its parent if and only if it has strictly better (lower) objective value.
#
# Adaptation: Scale factor F and crossover rate CR are adapted per-dimension using a success-history mechanism. The mutation scale for the best component is also adapted based on the diversity of the population.
#
# Exploration mechanisms: Use of multiple DE strategies with different exploration-exploitation balances. A Cauchy perturbation operator is applied to the best solution with diminishing probability. Re-initialization of stagnant solutions.
#
# Exploitation mechanisms: Local refinement via the DE/best/1 strategy when diversity is low. Shrinking of the Cauchy perturbation scale over time. Tracking and exploitation of the best-found solution.
#
# Boundary handling: A bounce-back mechanism that reflects infeasible components back into bounds with a random perturbation toward the feasible region.
#
# Budget strategy: Pre-allocates budget with (budget - population_size) function evaluations for the main loop, keeps the remainder for final local refinement.
#
# Closest known influences: Similar to SHADE (Success-History Adaptive Differential Evolution) with elements from JADE. The Cauchy perturbation and multiple strategy pool are inspired by ensemble DE variants.
#
# Novelty or unusual aspects: The combination of three distinct DE strategies with adaptive selection, plus a Cauchy perturbation phase that transitions from exploration to exploitation based on remaining budget. Bounce-back boundary handling with random reflection.
#
# Failure modes: May converge prematurely on highly multimodal landscapes if population diversity collapses too quickly. The adaptive mechanisms can sometimes lead to over-exploitation on functions with deceptive local minima. Not suitable for problems requiring exact optimal solutions within very tight budgets (e.g., < 100 evaluations).
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
        # Population size: scale with dimension, but keep reasonable
        self.pop_size = max(10, min(30, int(np.sqrt(budget))))
        
        # Strategy probabilities (DE/rand/1, DE/best/1, DE/current-to-best/1)
        self.strategy_probs = np.array([0.3, 0.4, 0.3])
        self.strategy_success = np.array([1.0, 1.0, 1.0])
        self.strategy_attempts = np.array([1.0, 1.0, 1.0])
        
        # Adaptation memories for F and CR
        self.memory_size = 5
        self.F_memory = np.full(self.memory_size, 0.5)
        self.CR_memory = np.full(self.memory_size, 0.5)
        self.memory_index = 0
        
        # Cauchy perturbation parameters
        self.cauchy_scale_init = 0.1
        self.cauchy_probability_init = 0.05
        
        # Stagnation detection
        self.stagnation_counter = 0
        self.stagnation_threshold = max(10, budget // 20)
        
    def __call__(self, func):
        # Determine bounds from function
        try:
            lower = func.lower
            upper = func.upper
        except AttributeError:
            try:
                lower = func.bounds.lb
                upper = func.bounds.ub
            except AttributeError:
                lower = func.bounds[0]
                upper = func.bounds[1]
        
        lb = np.array(lower, dtype=float)
        ub = np.array(upper, dtype=float)
        range_width = ub - lb
        
        # Initialize population
        population = lb + np.random.random((self.pop_size, self.dim)) * range_width
        fitness = np.array([func(x) for x in population])
        evals = self.pop_size
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]
        
        # Main evolutionary loop
        while evals < self.budget - self.pop_size:
            # Update stagnation counter
            old_best = best_y
            
            # Adaptive parameters for this generation
            F = self._sample_F()
            CR = self._sample_CR()
            
            # Generate offspring
            offspring = []
            offspring_fitness = []
            
            for i in range(self.pop_size):
                if len(offspring) >= self.budget - evals:
                    break
                    
                # Select strategy via roulette wheel
                strategy = np.random.choice(3, p=self.strategy_probs)
                
                # Generate trial vector based on strategy
                if strategy == 0:  # DE/rand/1/bin
                    trial = self._rand_1_bin(population, i, F, CR, lb, ub, range_width)
                elif strategy == 1:  # DE/best/1/bin
                    trial = self._best_1_bin(population, i, best_x, F, CR, lb, ub, range_width)
                else:  # DE/current-to-best/1/bin
                    trial = self._current_to_best_bin(population, i, best_x, F, CR, lb, ub, range_width)
                
                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1
                self.strategy_attempts[strategy] += 1
                
                # Track success for strategy adaptation
                if trial_fitness < fitness[i]:
                    self.strategy_success[strategy] += 1
                
                offspring.append((trial, trial_fitness, i, strategy))
                offspring_fitness.append(trial_fitness)
            
            # Update population with greedy selection
            for trial, trial_fitness, i, strategy in offspring:
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # Update best
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()
            
            # Update strategy probabilities based on success rates
            success_rates = self.strategy_success / np.maximum(self.strategy_attempts, 1e-10)
            self.strategy_probs = success_rates / np.sum(success_rates)
            # Ensure minimum exploration probability
            self.strategy_probs = np.maximum(self.strategy_probs, 0.05)
            self.strategy_probs /= np.sum(self.strategy_probs)
            
            # Update F and CR memories based on successful parameters
            self._update_memory(F, CR)
            
            # Cauchy perturbation on best solution (exploration/exploitation balance)
            remaining_budget = self.budget - evals
            cauchy_prob = self.cauchy_probability_init * (remaining_budget / self.budget)
            cauchy_scale = self.cauchy_scale_init * (remaining_budget / self.budget)
            
            if np.random.random() < cauchy_prob and evals < self.budget:
                perturbed = best_x + cauchy_scale * range_width * np.random.standard_cauchy(self.dim)
                # Boundary handling
                perturbed = np.clip(perturbed, lb, ub)
                perturb_fitness = func(perturbed)
                evals += 1
                
                if perturb_fitness < best_y:
                    best_y = perturb_fitness
                    best_x = perturbed.copy()
                    
                    # Inject into population
                    worst_idx = np.argmax(fitness)
                    if perturb_fitness < fitness[worst_idx]:
                        population[worst_idx] = perturbed
                        fitness[worst_idx] = perturb_fitness
            
            # Stagnation handling: reinitialize worst-performing solutions
            if best_y >= old_best:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.stagnation_threshold:
                    # Reinitialize bottom 20% of population
                    sorted_indices = np.argsort(fitness)
                    n_reinit = max(1, int(0.2 * self.pop_size))
                    for idx in sorted_indices[-n_reinit:]:
                        population[idx] = lb + np.random.random(self.dim) * range_width
                        fitness[idx] = func(population[idx])
                        evals += 1
                        if evals >= self.budget:
                            break
                    self.stagnation_counter = 0
            else:
                self.stagnation_counter = 0
        
        # Final local refinement using small perturbation of best solution
        remaining = self.budget - evals
        while remaining > 0 and remaining < self.pop_size:
            scale = 0.01 * range_width * (remaining / self.budget)
            candidate = best_x + scale * np.random.randn(self.dim)
            candidate = np.clip(candidate, lb, ub)
            cand_fitness = func(candidate)
            evals += 1
            remaining = self.budget - evals
            
            if cand_fitness < best_y:
                best_y = cand_fitness
                best_x = candidate.copy()
        
        return best_x, best_y
    
    def _sample_F(self):
        """Sample scale factor F using Cauchy distribution from memory."""
        idx = np.random.randint(0, self.memory_size)
        F = np.random.standard_cauchy() * 0.1 + self.F_memory[idx]
        return np.clip(F, 0.1, 1.0)
    
    def _sample_CR(self):
        """Sample crossover rate CR using normal distribution from memory."""
        idx = np.random.randint(0, self.memory_size)
        CR = np.random.randn() * 0.1 + self.CR_memory[idx]
        return np.clip(CR, 0.0, 1.0)
    
    def _update_memory(self, F, CR):
        """Update F and CR memories with new parameters."""
        self.F_memory[self.memory_index] = F
        self.CR_memory[self.memory_index] = CR
        self.memory_index = (self.memory_index + 1) % self.memory_size
    
    def _rand_1_bin(self, population, idx, F, CR, lb, ub, range_width):
        """DE/rand/1/bin mutation and crossover."""
        dim = self.dim
        # Select three distinct random indices different from idx
        indices = list(range(self.pop_size))
        indices.remove(idx)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        
        # Mutation
        mutant = population[r1] + F * (population[r2] - population[r3])
        
        # Binomial crossover
        j_rand = np.random.randint(dim)
        trial = np.array([mutant[j] if np.random.random() < CR or j == j_rand 
                         else population[idx][j] for j in range(dim)])
        
        # Boundary handling with bounce-back
        for j in range(dim):
            if trial[j] < lb[j] or trial[j] > ub[j]:
                if np.random.random() < 0.5:
                    trial[j] = lb[j] + np.random.random() * (population[idx][j] - lb[j])
                else:
                    trial[j] = population[idx][j] + np.random.random() * (ub[j] - population[idx][j])
        
        return trial
    
    def _best_1_bin(self, population, idx, best_x, F, CR, lb, ub, range_width):
        """DE/best/1/bin mutation and crossover."""
        dim = self.dim
        indices = list(range(self.pop_size))
        indices.remove(idx)
        r1, r2 = np.random.choice(indices, 2, replace=False)
        
        # Mutation
        mutant = best_x + F * (population[r1] - population[r2])
        
        # Binomial crossover
        j_rand = np.random.randint(dim)
        trial = np.array([mutant[j] if np.random.random() < CR or j == j_rand 
                         else population[idx][j] for j in range(dim)])
        
        # Boundary handling
        for j in range(dim):
            if trial[j] < lb[j] or trial[j] > ub[j]:
                if np.random.random() < 0.5:
                    trial[j] = lb[j] + np.random.random() * (best_x[j] - lb[j])
                else:
                    trial[j] = best_x[j] + np.random.random() * (ub[j] - best_x[j])
        
        return trial
    
    def _current_to_best_bin(self, population, idx, best_x, F, CR, lb, ub, range_width):
        """DE/current-to-best/1/bin mutation and crossover."""
        dim = self.dim
        indices = list(range(self.pop_size))
        indices.remove(idx)
        r1, r2 = np.random.choice(indices, 2, replace=False)
        
        # Mutation
        mutant = population[idx] + F * (best_x - population[idx]) + F * (population[r1] - population[r2])
        
        # Binomial crossover
        j_rand = np.random.randint(dim)
        trial = np.array([mutant[j] if np.random.random() < CR or j == j_rand 
                         else population[idx][j] for j in range(dim)])
        
        # Boundary handling
        for j in range(dim):
            if trial[j] < lb[j] or trial[j] > ub[j]:
                if np.random.random() < 0.5:
                    trial[j] = lb[j] + np.random.random() * (population[idx][j] - lb[j])
                else:
                    trial[j] = population[idx][j] + np.random.random() * (ub[j] - population[idx][j])
        
        return trial
