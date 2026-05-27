import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm uses a hybrid approach of Nelder-Mead simplex optimization and random sampling to minimize
#          the given black-box function within the provided budget.
# Search state: The search state is represented by a simplex of N+1 vertices, where N is the problem dimension.
# Candidate generation: New candidates are generated either by performing a Nelder-Mead simplex step or by sampling
#                      uniformly within the variable bounds.
# Selection and replacement: The best candidate from the current simplex and the newly generated candidate are
#                           compared, and the better one replaces the worst vertex in the simplex.
# Adaptation: The simplex size is adaptively adjusted based on the progress of the optimization.
# Exploration mechanisms: Random sampling provides global exploration, while the Nelder-Mead simplex method
#                        provides local exploitation.
# Exploitation mechanisms: The Nelder-Mead simplex method is used to exploit promising regions of the search space.
# Boundary handling: Candidates that violate the variable bounds are projected back onto the feasible region.
# Budget strategy: The available budget is evenly divided between the number of simplex steps and random samples.
# Closest known influences: Nelder-Mead simplex optimization, random search.
# Novelty or unusual aspects: The hybrid approach of combining Nelder-Mead simplex and random sampling is
#                            uncommon for black-box optimization.
# Failure modes: The algorithm may struggle on highly multimodal functions or functions with very steep gradients,
#                as the Nelder-Mead simplex method may get stuck in local minima.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.simplex = None
        self.simplex_steps = 0
        self.random_samples = 0

    def __call__(self, func):
        if self.simplex is None:
            self.initialize_simplex(func)

        while self.simplex_steps + self.random_samples < self.budget:
            if self.simplex_steps < self.budget // 2:
                self.take_simplex_step(func)
                self.simplex_steps += 1
            else:
                self.take_random_sample(func)
                self.random_samples += 1

        best_idx = np.argmin([func(x) for x in self.simplex])
        best_x = self.simplex[best_idx]
        best_y = func(best_x)
        return best_x, best_y

    def initialize_simplex(self, func):
        """Initialize the Nelder-Mead simplex."""
        self.simplex = [np.random.uniform(func.lower, func.upper, self.dim) for _ in range(self.dim + 1)]

    def take_simplex_step(self, func):
        """Perform a Nelder-Mead simplex step."""
        # Compute the centroid of the simplex, excluding the worst vertex
        worst_idx = np.argmax([func(x) for x in self.simplex])
        centroid = np.mean([self.simplex[i] for i in range(len(self.simplex)) if i != worst_idx], axis=0)

        # Reflect the worst vertex through the centroid
        reflected = centroid + (centroid - self.simplex[worst_idx])
        reflected = np.clip(reflected, func.lower, func.upper)
        reflected_value = func(reflected)

        if reflected_value < func(self.simplex[worst_idx]):
            # The reflected vertex is better than the worst, so replace the worst with it
            self.simplex[worst_idx] = reflected
        else:
            # The reflected vertex is not better, so try expanding or contracting
            expanded = centroid + 2 * (reflected - centroid)
            expanded = np.clip(expanded, func.lower, func.upper)
            expanded_value = func(expanded)
            if expanded_value < reflected_value:
                self.simplex[worst_idx] = expanded
            else:
                contracted = centroid + 0.5 * (self.simplex[worst_idx] - centroid)
                contracted = np.clip(contracted, func.lower, func.upper)
                contracted_value = func(contracted)
                if contracted_value < func(self.simplex[worst_idx]):
                    self.simplex[worst_idx] = contracted
                else:
                    # Shrink the simplex towards the best vertex
                    for i in range(len(self.simplex)):
                        if i != np.argmin([func(x) for x in self.simplex]):
                            self.simplex[i] = self.simplex[np.argmin([func(x) for x in self.simplex])] + 0.5 * (self.simplex[i] - self.simplex[np.argmin([func(x) for x in self.simplex])])
                            self.simplex[i] = np.clip(self.simplex[i], func.lower, func.upper)

    def take_random_sample(self, func):
        """Generate a new random sample within the variable bounds."""
        new_sample = np.random.uniform(func.lower, func.upper, self.dim)
        new_value = func(new_sample)
        worst_idx = np.argmax([func(x) for x in self.simplex])
        if new_value < func(self.simplex[worst_idx]):
            self.simplex[worst_idx] = new_sample
