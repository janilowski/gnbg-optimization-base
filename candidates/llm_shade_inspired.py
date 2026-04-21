from __future__ import annotations

import time
from enum import IntEnum

import numpy as np


class Terminations(IntEnum):
    NO_TERMINATION = 0
    MAX_FUNCTION_EVALUATIONS = 1
    MAX_RUNTIME = 2
    FITNESS_THRESHOLD = 3
    EARLY_STOPPING = 4


def _cauchy_rvs(rng: np.random.Generator, loc: float, scale: float) -> float:
    return float(loc + scale * rng.standard_cauchy())


class Optimizer(object):
    def __init__(self, problem, options):
        self.fitness_function = problem.get("fitness_function")
        self.ndim_problem = problem.get("ndim_problem")
        assert self.ndim_problem > 0

        self.upper_boundary = problem.get("upper_boundary")
        self.lower_boundary = problem.get("lower_boundary")
        self.initial_upper_boundary = problem.get(
            "initial_upper_boundary", self.upper_boundary
        )
        self.initial_lower_boundary = problem.get(
            "initial_lower_boundary", self.lower_boundary
        )
        self.problem_name = problem.get("problem_name")
        if (self.problem_name is None) and hasattr(self.fitness_function, "__name__"):
            self.problem_name = self.fitness_function.__name__

        self.options = options
        self.max_function_evaluations = options.get("max_function_evaluations", np.inf)
        self.max_runtime = options.get("max_runtime", np.inf)
        self.fitness_threshold = options.get("fitness_threshold", -np.inf)
        self.n_individuals = options.get("n_individuals")
        self.n_parents = options.get("n_parents")

        self.seed_rng = options.get("seed_rng")
        if self.seed_rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.seed_rng)
        self.seed_initialization = options.get(
            "seed_initialization", self.rng.integers(np.iinfo(np.int64).max)
        )
        self.rng_initialization = np.random.default_rng(self.seed_initialization)
        self.seed_optimization = options.get(
            "seed_optimization", self.rng.integers(np.iinfo(np.int64).max)
        )
        self.rng_optimization = np.random.default_rng(self.seed_optimization)

        self.saving_fitness = options.get("saving_fitness", 0)
        self.verbose = options.get("verbose", 10)

        self.Terminations, self.termination_signal = Terminations, 0
        self.n_function_evaluations = options.get("n_function_evaluations", 0)
        self.start_function_evaluations = None
        self.time_function_evaluations = options.get("time_function_evaluations", 0)
        self.runtime, self.start_time = options.get("runtime", 0), None
        self.best_so_far_y, self.best_so_far_x = options.get(
            "best_so_far_y", np.inf
        ), None
        self.fitness = None
        self.is_restart = options.get("is_restart", True)
        self.early_stopping_evaluations = options.get(
            "early_stopping_evaluations", np.inf
        )
        self.early_stopping_threshold = options.get("early_stopping_threshold", 0.01)
        self.counter_early_stopping, self.base_early_stopping = (
            0,
            self.best_so_far_y,
        )

    def _evaluate_fitness(self, x, args=None):
        self.start_function_evaluations = time.time()
        if args is None:
            y = self.fitness_function(x)
        else:
            y = self.fitness_function(x, args=args)
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += 1
        if y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y
        if (self.base_early_stopping - y) <= self.early_stopping_threshold and abs(
            self.best_so_far_y - self.fitness_threshold
        ) > 1e-8:
            self.counter_early_stopping += 1
        else:
            self.counter_early_stopping, self.base_early_stopping = 0, y
        return float(y)

    def _check_terminations(self):
        self.runtime = time.time() - self.start_time
        if self.n_function_evaluations >= self.max_function_evaluations:
            termination_signal = True, Terminations.MAX_FUNCTION_EVALUATIONS
        elif self.runtime >= self.max_runtime:
            termination_signal = True, Terminations.MAX_RUNTIME
        elif self.best_so_far_y <= self.fitness_threshold:
            termination_signal = True, Terminations.FITNESS_THRESHOLD
        elif self.counter_early_stopping >= self.early_stopping_evaluations:
            termination_signal = True, Terminations.EARLY_STOPPING
        else:
            termination_signal = False, Terminations.NO_TERMINATION
        self.termination_signal = termination_signal[1]
        return termination_signal[0]

    def _compress_fitness(self, fitness):
        fitness = np.array(fitness)
        for i in range(len(fitness) - 1):
            if fitness[i] < fitness[i + 1]:
                fitness[i + 1] = fitness[i]
        if self.saving_fitness == 1:
            self.fitness = np.stack((np.arange(1, len(fitness) + 1), fitness), 1)
        elif self.saving_fitness > 1:
            index = np.arange(1, len(fitness), self.saving_fitness)
            index = np.append(index, len(fitness)) - 1
            self.fitness = np.stack((index, fitness[index]), 1)
            self.fitness[0, 0], self.fitness[-1, 0] = 1, len(fitness)

    def _check_success(self):
        if (self.upper_boundary is not None) and (self.lower_boundary is not None) and (
            np.any(self.lower_boundary > self.best_so_far_x)
            or np.any(self.best_so_far_x > self.upper_boundary)
        ):
            return False
        elif np.isnan(self.best_so_far_y) or np.any(np.isnan(self.best_so_far_x)):
            return False
        return True

    def _collect(self, fitness):
        if self.saving_fitness:
            self._compress_fitness(fitness[: self.n_function_evaluations])
        return {
            "best_so_far_x": self.best_so_far_x,
            "best_so_far_y": self.best_so_far_y,
            "n_function_evaluations": self.n_function_evaluations,
            "runtime": time.time() - self.start_time,
            "termination_signal": self.termination_signal,
            "time_function_evaluations": self.time_function_evaluations,
            "fitness": self.fitness,
            "success": self._check_success(),
        }

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def optimize(self, fitness_function=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []
        return fitness


class DE(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:
            self.n_individuals = 100
        assert self.n_individuals > 0
        self._n_generations = 0
        self._printed_evaluations = self.n_function_evaluations

    def initialize(self):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def crossover(self):
        raise NotImplementedError

    def select(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, fitness, y, is_print=False):
        if y is not None and self.saving_fitness:
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose:
            is_verbose = self._printed_evaluations != self.n_function_evaluations
            is_verbose_1 = (not self._n_generations % self.verbose) and is_verbose
            is_verbose_2 = self.termination_signal > 0 and is_verbose
            is_verbose_3 = is_print and is_verbose
            if is_verbose_1 or is_verbose_2 or is_verbose_3:
                self._printed_evaluations = self.n_function_evaluations

    def _collect(self, fitness=None, y=None):
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        return results


class JADE(DE):
    def __init__(self, problem, options):
        DE.__init__(self, problem, options)
        self.mu = options.get("mu", 0.5)
        self.median = options.get("median", 0.5)
        self.p = options.get("p", 0.05)
        assert 0.0 <= self.p <= 1.0
        self.c = options.get("c", 0.1)
        assert 0.0 <= self.c <= 1.0
        self.is_bound = options.get("is_bound", False)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(
            self.initial_lower_boundary,
            self.initial_upper_boundary,
            size=(self.n_individuals, self.ndim_problem),
        )
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        a = np.empty((0, self.ndim_problem))
        return x, y, a

    def bound(self, x=None, xx=None):
        if not self.is_bound:
            return x
        for k in range(self.n_individuals):
            idx = np.array(x[k] < self.lower_boundary)
            if idx.any():
                x[k][idx] = (self.lower_boundary + xx[k])[idx] / 2.0
            idx = np.array(x[k] > self.upper_boundary)
            if idx.any():
                x[k][idx] = (self.upper_boundary + xx[k])[idx] / 2.0
        return x

    def mutate(self, x=None, y=None, a=None):
        x_mu = np.empty((self.n_individuals, self.ndim_problem))
        f_mu = np.empty((self.n_individuals,))
        order = np.argsort(y)[: int(np.ceil(self.p * self.n_individuals))]
        x_p = x[self.rng_optimization.choice(order, (self.n_individuals,))]
        x_un = np.vstack((np.copy(x), a))
        for k in range(self.n_individuals):
            f_mu[k] = _cauchy_rvs(self.rng_optimization, self.median, 0.1)
            while f_mu[k] <= 0.0:
                f_mu[k] = _cauchy_rvs(self.rng_optimization, self.median, 0.1)
            if f_mu[k] > 1.0:
                f_mu[k] = 1.0
            r1 = self.rng_optimization.choice(
                [i for i in range(self.n_individuals) if i != k]
            )
            r2 = self.rng_optimization.choice(
                [i for i in range(len(x_un)) if i != k and i != r1]
            )
            x_mu[k] = x[k] + f_mu[k] * (x_p[k] - x[k]) + f_mu[k] * (
                x[r1] - x_un[r2]
            )
        return x_mu, f_mu

    def crossover(self, x_mu=None, x=None):
        x_cr = np.copy(x)
        p_cr = self.rng_optimization.normal(self.mu, 0.1, (self.n_individuals,))
        p_cr = np.minimum(np.maximum(p_cr, 0.0), 1.0)
        for k in range(self.n_individuals):
            i_rand = self.rng_optimization.integers(self.ndim_problem)
            for i in range(self.ndim_problem):
                if (i == i_rand) or (self.rng_optimization.random() < p_cr[k]):
                    x_cr[k, i] = x_mu[k, i]
        return x_cr, p_cr

    def select(self, args=None, x=None, y=None, x_cr=None, a=None, f_mu=None, p_cr=None):
        f = np.empty((0,))
        p = np.empty((0,))
        for k in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(x_cr[k], args)
            if yy < y[k]:
                a = np.vstack((a, x[k]))
                f = np.hstack((f, f_mu[k]))
                p = np.hstack((p, p_cr[k]))
                x[k] = x_cr[k]
                y[k] = yy
        if len(p) != 0:
            self.mu = (1.0 - self.c) * self.mu + self.c * np.mean(p)
        if len(f) != 0:
            self.median = (1.0 - self.c) * self.median + self.c * np.sum(
                np.power(f, 2)
            ) / np.sum(f)
        return x, y, a

    def iterate(self, x=None, y=None, a=None, args=None):
        x_mu, f_mu = self.mutate(x, y, a)
        x_cr, p_cr = self.crossover(x_mu, x)
        x_cr = self.bound(x_cr, x)
        x, y, a = self.select(args, x, y, x_cr, a, f_mu, p_cr)
        if len(a) > self.n_individuals:
            a = np.delete(
                a,
                self.rng_optimization.choice(
                    len(a), (len(a) - self.n_individuals,), False
                ),
                0,
            )
        self._n_generations += 1
        return x, y, a

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)
        x, y, a = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            x, y, a = self.iterate(x, y, a, args)
        return self._collect(fitness, y)


class LSHADE(JADE):
    def __init__(self, problem, options, rand_seed, optimal_value):
        JADE.__init__(self, problem, options)
        self.h = options.get("h", 100)
        assert 0 < self.h
        self.m_mu = np.ones(self.h) * self.mu
        self.m_median = np.ones(self.h) * self.median
        self._k = 0
        self.p_min = 2.0 / self.n_individuals
        self.initial_pop_size = self.n_individuals

        self.rng_initialization = np.random.default_rng(rand_seed)
        self.rng_optimization = np.random.default_rng(rand_seed)

        self.rotation = problem.get("rotation", None)
        self.lambda_ = problem.get("lambda_", None)
        self.omega_ = problem.get("omega_", None)
        self.optimal_value = optimal_value
        self.restart_threshold = 5e4

    def initialize_1(self, args=None):
        x = self.rng_initialization.uniform(
            self.initial_lower_boundary,
            self.initial_upper_boundary,
            size=(self.n_individuals, self.ndim_problem),
        )
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        a = np.empty((0, self.ndim_problem))
        return x, y, a

    def initialize_2(self, args=None):
        x = np.zeros((self.n_individuals, self.ndim_problem))

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        n_seeds = min(max(3, int(np.cbrt(self.n_individuals))), len(primes))

        seeds = np.zeros((n_seeds, self.ndim_problem))
        domain_range = self.initial_upper_boundary - self.initial_lower_boundary

        for s in range(n_seeds):
            for d in range(self.ndim_problem):
                prime_factor = primes[s % len(primes)]
                spiral_param = (
                    prime_factor * s + d * primes[(s + d) % len(primes)]
                ) % 1000 / 1000.0

                log_factor = np.log(1 + spiral_param * np.e)
                angular_component = 2 * np.pi * spiral_param * prime_factor / 10.0

                seeds[s, d] = self.initial_lower_boundary + (
                    log_factor * np.cos(angular_component) * 0.5 + 0.5
                ) * domain_range

        individuals_per_cell = self.n_individuals // n_seeds
        remainder_individuals = self.n_individuals % n_seeds
        cell_assignments = []

        for s in range(n_seeds):
            cell_size = individuals_per_cell + (1 if s < remainder_individuals else 0)
            cell_assignments.extend([s] * cell_size)

        self.rng_optimization.shuffle(cell_assignments)

        current_idx = 0

        for cell_id in range(n_seeds):
            seed_point = seeds[cell_id]
            cell_indices = [i for i, c in enumerate(cell_assignments) if c == cell_id]

            for local_idx, global_idx in enumerate(cell_indices):
                if current_idx >= self.n_individuals:
                    break

                density_progress = (local_idx + 1) / len(cell_indices)
                log_density = np.log(1 + density_progress * (np.e - 1)) / np.log(np.e)

                tessellation_angles = np.zeros(self.ndim_problem)
                for d in range(self.ndim_problem):
                    base_angle = 2 * np.pi * d / self.ndim_problem
                    cell_perturbation = cell_id * np.pi / n_seeds
                    tessellation_angles[d] = (
                        base_angle + cell_perturbation + local_idx * 0.1
                    )

                position_offset = np.zeros(self.ndim_problem)
                max_cell_radius = 0.4 * np.linalg.norm(domain_range) / np.sqrt(n_seeds)

                for d in range(self.ndim_problem):
                    radial_distance = max_cell_radius * log_density
                    angle_d = tessellation_angles[d]

                    dim_scale = 1.0 / np.sqrt(1 + d * 0.1)
                    position_offset[d] = radial_distance * np.sin(angle_d) * dim_scale

                noise_strength = 0.1 * (1 - log_density * 0.6)
                tessellation_noise = self.rng_optimization.laplace(
                    0, noise_strength, self.ndim_problem
                )
                scaled_noise = tessellation_noise * domain_range

                x[current_idx] = seed_point + position_offset + scaled_noise

                for d in range(self.ndim_problem):
                    reflection_count = 0
                    while (
                        x[current_idx, d] < self.initial_lower_boundary
                        or x[current_idx, d] > self.initial_upper_boundary
                    ) and reflection_count < 3:
                        if x[current_idx, d] < self.initial_lower_boundary:
                            overshoot = self.initial_lower_boundary - x[current_idx, d]
                            x[current_idx, d] = self.initial_lower_boundary + overshoot * 0.8
                        elif x[current_idx, d] > self.initial_upper_boundary:
                            overshoot = x[current_idx, d] - self.initial_upper_boundary
                            x[current_idx, d] = self.initial_upper_boundary - overshoot * 0.8

                        reflection_count += 1

                    x[current_idx, d] = np.clip(
                        x[current_idx, d],
                        self.initial_lower_boundary,
                        self.initial_upper_boundary,
                    )

                current_idx += 1

        attraction_fraction = 0.2
        n_attracted = int(attraction_fraction * self.n_individuals)
        attracted_indices = self.rng_optimization.choice(
            self.n_individuals, n_attracted, replace=False
        )

        cell_centroids = np.zeros((n_seeds, self.ndim_problem))
        for cell_id in range(n_seeds):
            cell_members = [
                i for i, c in enumerate(cell_assignments[:current_idx]) if c == cell_id
            ]
            if cell_members:
                cell_centroids[cell_id] = np.mean(x[cell_members], axis=0)
            else:
                cell_centroids[cell_id] = seeds[cell_id]

        for idx in attracted_indices:
            assigned_cell = cell_assignments[idx]
            centroid = cell_centroids[assigned_cell]

            attraction_weight = self.rng_optimization.beta(2, 5)
            direction_to_centroid = centroid - x[idx]

            distance_to_centroid = np.linalg.norm(direction_to_centroid)
            if distance_to_centroid > 0:
                normalized_direction = direction_to_centroid / distance_to_centroid
                attraction_magnitude = attraction_weight * distance_to_centroid * 0.3
                x[idx] += normalized_direction * attraction_magnitude

            x[idx] = np.clip(
                x[idx], self.initial_lower_boundary, self.initial_upper_boundary
            )

        y = np.zeros(self.n_individuals)
        for i in range(self.n_individuals):
            y[i] = self._evaluate_fitness(x[i], args)
            if self._check_terminations():
                break

        a = np.zeros((0, self.ndim_problem))

        return x, y, a

    def initialize_3(self, args=None):
        def sobol_sequence(n_samples, n_dims):
            samples = np.zeros((n_samples, n_dims))
            for i in range(n_dims):
                base = 2 + i
                sequence = []
                for j in range(n_samples):
                    val = 0.0
                    frac = 1.0 / base
                    n = j + 1
                    while n > 0:
                        val += (n % base) * frac
                        n //= base
                        frac /= base
                    sequence.append(val % 1.0)
                samples[:, i] = sequence
            return samples

        sobol_samples = sobol_sequence(self.n_individuals // 2, self.ndim_problem)

        x = np.zeros((self.n_individuals, self.ndim_problem))

        boundary_range = self.initial_upper_boundary - self.initial_lower_boundary
        dynamic_scale = 0.8 + 0.4 * np.exp(-self.ndim_problem / 20.0)

        for i in range(self.n_individuals // 2):
            x[i] = (
                self.initial_lower_boundary
                + sobol_samples[i] * boundary_range * dynamic_scale
            )

        remaining = self.n_individuals - self.n_individuals // 2

        n_clusters = min(5, max(2, self.ndim_problem // 3))
        cluster_centers = np.zeros((n_clusters, self.ndim_problem))

        for c in range(n_clusters):
            cluster_position = (c + 1) / (n_clusters + 1)
            for d in range(self.ndim_problem):
                offset = self.rng_optimization.uniform(-0.2, 0.2)
                cluster_centers[c, d] = self.initial_lower_boundary + (
                    cluster_position + offset
                ) * boundary_range

        cluster_size = remaining // n_clusters
        idx = self.n_individuals // 2

        for c in range(n_clusters):
            members = cluster_size if c < n_clusters - 1 else remaining - c * cluster_size

            for m in range(members):
                if idx < self.n_individuals:
                    radius = 0.1 * boundary_range * (
                        1.0 + 0.5 * self.rng_optimization.uniform()
                    )
                    perturbation = self.rng_optimization.normal(
                        0, radius, self.ndim_problem
                    )
                    x[idx] = cluster_centers[c] + perturbation
                    idx += 1

        if hasattr(self, "m_median") and len(self.m_median) > 0:
            memory_influence = 0.05 * np.mean(np.abs(self.m_median))
            memory_directions = self.rng_optimization.normal(0, memory_influence, x.shape)
            x += memory_directions

        x = np.clip(x, self.initial_lower_boundary, self.initial_upper_boundary)

        y = np.zeros(self.n_individuals)
        for i in range(self.n_individuals):
            y[i] = self._evaluate_fitness(x[i], args)

        a = np.zeros((0, self.ndim_problem))

        return x, y, a

    def crossover_1(self, x_mu=None, x=None, r=None):
        x_cr = np.copy(x)
        p_cr = np.empty((self.n_individuals,))
        for k in range(self.n_individuals):
            p_cr[k] = self.rng_optimization.normal(self.m_mu[r[k]], 0.1)
            p_cr[k] = np.minimum(np.maximum(p_cr[k], 0.0), 1.0)
            i_rand = self.rng_optimization.integers(self.ndim_problem)
            for i in range(self.ndim_problem):
                if (i == i_rand) or (self.rng_optimization.random() < p_cr[k]):
                    x_cr[k, i] = x_mu[k, i]
        return x_cr, p_cr

    def crossover_2(self, x_mu=None, x=None, r=None):
        x_cr = np.copy(x_mu)
        p_cr = np.zeros(self.n_individuals)

        bracket_size = max(4, self.n_individuals // 4)
        n_brackets = max(2, self.n_individuals // bracket_size)

        fitness_ranks = np.argsort(np.argsort(r))
        centroid = np.mean(x_mu, axis=0)
        distances = np.linalg.norm(x_mu - centroid, axis=1)
        distance_ranks = np.argsort(np.argsort(distances))

        total_generations = self.max_function_evaluations / self.initial_pop_size
        cycle_period = total_generations * 0.2
        phase = (self._n_generations % cycle_period) / cycle_period

        oscillation_factor = 0.5 * (1 + np.sin(2 * np.pi * phase))
        exploration_weight = 0.8 * oscillation_factor + 0.2 * (1 - oscillation_factor)

        combined_scores = (
            (1 - exploration_weight) * fitness_ranks
            + exploration_weight * distance_ranks
        )
        sorted_indices = np.argsort(combined_scores)

        brackets = [[] for _ in range(n_brackets)]
        for i, idx in enumerate(sorted_indices):
            bracket_idx = (
                i % n_brackets
                if (i // n_brackets) % 2 == 0
                else n_brackets - 1 - (i % n_brackets)
            )
            brackets[bracket_idx].append(idx)

        momentum = 0.5 * (
            1
            + np.tanh(
                (self._n_generations - total_generations * 0.3)
                / (total_generations * 0.1)
            )
        )

        for bracket_idx, bracket in enumerate(brackets):
            if len(bracket) < 2:
                continue

            bracket_fitness = r[bracket]
            competitiveness = np.std(bracket_fitness) / (np.mean(bracket_fitness) + 1e-8)

            bracket_ranks = np.argsort(bracket_fitness)
            winners = [bracket[i] for i in bracket_ranks[: len(bracket) // 2]]
            losers = [bracket[i] for i in bracket_ranks[len(bracket) // 2 :]]

            for i in range(len(bracket)):
                individual_idx = bracket[i]

                if individual_idx in winners:
                    partner_pool = [w for w in winners if w != individual_idx]
                    base_prob = 0.15 + 0.4 * (1 - momentum)
                else:
                    partner_pool = winners
                    base_prob = 0.65 + 0.15 * momentum

                if partner_pool:
                    partner = self.rng_optimization.choice(partner_pool)
                else:
                    partner = self.rng_optimization.choice(
                        [idx for idx in bracket if idx != individual_idx]
                    )

                oscillation_bonus = 0.1 * np.sin(4 * np.pi * phase)
                competitiveness_factor = np.tanh(2 * competitiveness)
                p_cr[individual_idx] = (
                    base_prob * (0.85 + 0.3 * competitiveness_factor)
                    + oscillation_bonus
                )
                p_cr[individual_idx] *= (
                    0.75 + 0.25 * self.rng_optimization.random()
                )
                p_cr[individual_idx] = np.clip(p_cr[individual_idx], 0.1, 0.9)

                mask = self.rng_optimization.random(self.ndim_problem) < p_cr[individual_idx]
                if not np.any(mask):
                    mask[self.rng_optimization.integers(0, self.ndim_problem)] = True

                x_cr[individual_idx][mask] = x[partner][mask]

        return x_cr, p_cr

    def select_initialize(self, index=0):
        if index == 0:
            return self.initialize_1()
        elif index == 1:
            return self.initialize_2()
        else:
            return self.initialize_3()

    def mutate(self, x=None, y=None, a=None):
        x_mu = np.empty((self.n_individuals, self.ndim_problem))
        f_mu = np.empty((self.n_individuals,))
        x_un = np.vstack((np.copy(x), a))
        r = self.rng_optimization.choice(self.h, (self.n_individuals,))
        order = np.argsort(y)[:]
        p = (0.2 - self.p_min) * self.rng_optimization.random((self.n_individuals,)) + self.p_min
        idx = [order[self.rng_optimization.choice(int(i))] for i in np.ceil(p * self.n_individuals)]
        for k in range(self.n_individuals):
            f_mu[k] = _cauchy_rvs(self.rng_optimization, self.m_median[r[k]], 0.1)
            while f_mu[k] <= 0.0:
                f_mu[k] = _cauchy_rvs(self.rng_optimization, self.m_median[r[k]], 0.1)
            if f_mu[k] > 1.0:
                f_mu[k] = 1.0
            r1 = self.rng_optimization.choice(
                [i for i in range(self.n_individuals) if i != k]
            )
            r2 = self.rng_optimization.choice(
                [i for i in range(len(x_un)) if i != k and i != r1]
            )
            x_mu[k] = x[k] + f_mu[k] * (x[idx[k]] - x[k]) + f_mu[k] * (
                x[r1] - x_un[r2]
            )
        return x_mu, f_mu, r

    def crossover(self, x_mu=None, x=None, r=None):
        x_cr = np.copy(x)
        p_cr = np.empty((self.n_individuals,))
        for k in range(self.n_individuals):
            p_cr[k] = self.rng_optimization.normal(self.m_mu[r[k]], 0.1)
            p_cr[k] = np.minimum(np.maximum(p_cr[k], 0.0), 1.0)
            i_rand = self.rng_optimization.integers(self.ndim_problem)
            for i in range(self.ndim_problem):
                if (i == i_rand) or (self.rng_optimization.random() < p_cr[k]):
                    x_cr[k, i] = x_mu[k, i]
        return x_cr, p_cr

    def select(self, args=None, x=None, y=None, x_cr=None, a=None, f_mu=None, p_cr=None):
        f, p, d = np.empty((0,)), np.empty((0,)), np.empty((0,))
        for k in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(x_cr[k], args)
            if yy < y[k]:
                a = np.vstack((a, x[k]))
                f = np.hstack((f, f_mu[k]))
                p = np.hstack((p, p_cr[k]))
                d = np.hstack((d, y[k] - yy))
                x[k], y[k] = x_cr[k], yy
        if (len(p) != 0) and (len(f) != 0):
            w = d / np.sum(d)
            self.m_mu[self._k] = np.sum(w * p)
            self.m_median[self._k] = np.sum(w * np.power(f, 2)) / np.sum(w * f)
            self._k = (self._k + 1) % self.h
        return x, y, a

    def change_population(self, x=None, y=None, a=None, args=None):
        max_iterations = max(2, self.max_function_evaluations // self.initial_pop_size)
        reduction_factor = (self.initial_pop_size - 4) / (max_iterations - 1)
        self.n_individuals = max(
            4, int(self.initial_pop_size - self._n_generations * reduction_factor)
        )

        if len(a) > self.n_individuals:
            indices = np.argsort(y)[: self.n_individuals]
            x = x[indices]
            y = y[indices]
            a = np.delete(
                a,
                self.rng_optimization.choice(
                    len(a), (len(a) - self.n_individuals,), False
                ),
                0,
            )
        return x, y, a

    def iterate(self, x=None, y=None, a=None, args=None):
        x_mu, f_mu, r = self.mutate(x.copy(), y.copy(), a.copy())
        if (
            self.max_function_evaluations <= 500000
            and np.sum(self.omega_.flatten()) >= 200
        ):
            x_cr, p_cr = self.crossover_2(x_mu.copy(), x.copy(), r.copy())
        else:
            x_cr, p_cr = self.crossover_1(x_mu.copy(), x.copy(), r.copy())
        x_cr = self.bound(x_cr, x)
        x, y, a = self.select(args, x, y, x_cr, a, f_mu, p_cr)
        x, y, a = self.change_population(x.copy(), y.copy(), a.copy())
        self._n_generations += 1
        return x, y, a

    def check_not_improving(self):
        if self.counter_early_stopping >= self.restart_threshold:
            self.counter_early_stopping = 0
            self.base_early_stopping = -np.inf
            return True
        else:
            return False

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)

        if self.max_function_evaluations <= 500000:
            x, y, a = self.select_initialize(0)
            while not self._check_terminations():
                self._print_verbose_info(fitness, y)

                if np.sum(self.lambda_.flatten()) <= 0.05:
                    if self.check_not_improving():
                        x, y, a = self.select_initialize(0)

                x, y, a = self.iterate(x, y, a, args)
        else:
            if np.sum(self.lambda_.flatten()) >= 4:
                x, y, a = self.select_initialize(1)
            elif np.sum(self.lambda_.flatten()) <= 1.5:
                x, y, a = self.select_initialize(0)
            elif np.sum(self.lambda_.flatten()) < 2:
                x, y, a = self.select_initialize(2)
            else:
                x, y, a = self.select_initialize(0)

            while not self._check_terminations():
                self._print_verbose_info(fitness, y)

                if np.sum(self.lambda_.flatten()) >= 4:
                    if self.check_not_improving():
                        x, y, a = self.select_initialize(1)
                elif np.sum(self.lambda_.flatten()) < 2 and np.sum(
                    self.lambda_.flatten()
                ) > 1.5:
                    if self.check_not_improving():
                        x, y, a = self.select_initialize(2)

                x, y, a = self.iterate(x, y, a, args)

        return self._collect(fitness, y)


def _vectorize_bound(value, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(dim, float(arr), dtype=float)
    return arr.reshape(dim)


def _problem_bounds(problem, dim: int) -> tuple[np.ndarray, np.ndarray]:
    lower = getattr(problem, "lower", None)
    upper = getattr(problem, "upper", None)
    if lower is None or upper is None:
        bounds = getattr(problem, "bounds", None)
        if bounds is not None:
            lower = getattr(bounds, "lb", lower)
            upper = getattr(bounds, "ub", upper)
    if lower is None or upper is None:
        lower = -100.0
        upper = 100.0
    return _vectorize_bound(lower, dim), _vectorize_bound(upper, dim)


def _read_problem_meta(problem, names, default):
    for name in names:
        try:
            value = getattr(problem, name)
        except Exception:
            continue
        if value is not None:
            return np.asarray(value, dtype=float)
    return np.asarray(default, dtype=float)


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.last_result = None

    def __call__(self, problem):
        dim = int(getattr(problem, "dim", self.dim))
        lower, upper = _problem_bounds(problem, dim)

        lambda_ = _read_problem_meta(problem, ("lambda_", "Lambda"), [1.0])
        omega_ = _read_problem_meta(problem, ("omega_", "Omega"), [0.0])

        problem_dict = {
            "fitness_function": problem,
            "ndim_problem": dim,
            "upper_boundary": upper,
            "lower_boundary": lower,
            "initial_upper_boundary": upper,
            "initial_lower_boundary": lower,
            "problem_name": type(problem).__name__,
            "rotation": None,
            "lambda_": lambda_,
            "omega_": omega_,
        }
        options = {
            "max_function_evaluations": self.budget,
            "fitness_threshold": -np.inf,
        }

        rand_seed = int(np.random.randint(0, np.iinfo(np.int32).max))
        de = LSHADE(problem_dict, options, rand_seed=rand_seed, optimal_value=np.nan)
        self.last_result = de.optimize()
        return self.last_result


__all__ = ["Algorithm"]
