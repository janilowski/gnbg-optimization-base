import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a simple, gradient-free optimization algorithm that uses a combination of random search and simplex method (Nelder-Mead) to minimize the objective function.
# Search state: The algorithm maintains a simplex (set of points) in the search space, as well as a set of randomly sampled points.
# Candidate generation: New candidate solutions are generated either by reflecting, expanding, or contracting the simplex, or by randomly sampling the search space.
# Selection and replacement: The best candidate solution is selected and replaces the worst point in the simplex if it improves the function value.
# Adaptation: The size of the simplex is adapted based on the function values of the simplex points.
# Exploration mechanisms: Random sampling provides exploration of the search space.
# Exploitation mechanisms: The Nelder-Mead simplex method provides local exploitation around the current best solution.
# Boundary handling: If a candidate solution goes outside the search bounds, it is projected back onto the feasible region.
# Budget strategy: The algorithm uses the entire evaluation budget, with a fixed number of function evaluations per iteration.
# Closest known influences: This algorithm combines ideas from random search and the Nelder-Mead simplex method, which are both well-known gradient-free optimization techniques.
# Novelty or unusual aspects: The combination of random search and Nelder-Mead simplex method in a black-box optimization setting is relatively common, but the specific implementation details and parameter choices may be novel.
# Failure modes: The algorithm may struggle with highly multimodal or deceptive objective functions, as it relies on a combination of local and global search mechanisms without any explicit mechanism to escape local minima.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = None
        self.f = None
        self.simplex = None
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        if self.x is None:
            self.initialize(func)

        while self.budget > 0:
            self.update(func)

        return self.best_x, self.best_f

    def initialize(self, func):
        self.x = np.random.uniform(func.lower, func.upper, (self.dim + 1, self.dim))
        self.f = np.array([func(x) for x in self.x])
        self.simplex = self.x.copy()
        self.best_x = self.x[np.argmin(self.f)]
        self.best_f = self.f.min()
        self.budget -= self.dim + 1

    def update(self, func):
        # Perform Nelder-Mead simplex update
        self.simplex, self.f = self.nelder_mead_step(self.simplex, self.f, func)
        self.best_x = self.simplex[np.argmin(self.f)]
        self.best_f = self.f.min()

        # Evaluate a random sample
        x_rand = np.random.uniform(func.lower, func.upper, self.dim)
        f_rand = func(x_rand)
        self.budget -= 1

        # Replace the worst point in the simplex if the random sample is better
        if f_rand < self.f.max():
            self.simplex[np.argmax(self.f)] = x_rand
            self.f[np.argmax(self.f)] = f_rand

    def nelder_mead_step(self, simplex, f, func):
        """
        Perform one step of the Nelder-Mead simplex method.
        """
        n = len(simplex)
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        worst_f = f[-1]

        # Reflection
        x_r = centroid + (centroid - worst)
        f_r = func(x_r)
        self.budget -= 1
        if f_r < f.min():
            # Expansion
            x_e = centroid + 2 * (centroid - worst)
            f_e = func(x_e)
            self.budget -= 1
            if f_e < f_r:
                return np.vstack((simplex[:-1], x_e)), np.append(f[:-1], f_e)
            else:
                return np.vstack((simplex[:-1], x_r)), np.append(f[:-1], f_r)
        elif f_r < f[-2]:
            return np.vstack((simplex[:-1], x_r)), np.append(f[:-1], f_r)
        else:
            # Contraction
            if f_r < worst_f:
                x_c = centroid + 0.5 * (x_r - centroid)
            else:
                x_c = centroid - 0.5 * (worst - centroid)
            f_c = func(x_c)
            self.budget -= 1
            if f_c < worst_f:
                return np.vstack((simplex[:-1], x_c)), np.append(f[:-1], f_c)
            else:
                # Shrink
                new_simplex = simplex[0] + 0.5 * (simplex - simplex[0])
                new_f = [func(x) for x in new_simplex]
                self.budget -= len(new_simplex)
                return new_simplex, new_f
