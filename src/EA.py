import random
import numpy as np
from typing import Callable, Tuple, Dict
from enum import Enum
from functools import partial


class CrossingType(Enum):
    SINGLE_POINT = 0
    UNIFORM = 1
    SIMPLE_INTERMEDIATE = 2
    COMPLEX_INTERMEDIATE = 3


class EvolutionaryAlgorithm:
    def __init__(self,
                 f: Callable[[np.array], np.array],
                 dimension: int,
                 population_size: int,
                 elite_size: int,
                 bounds: Tuple[float, float] = (-10., 10.),
                 mutation_rate: float = 1.0,
                 mutation_prob: float = 0.5,
                 crossing_type: CrossingType = CrossingType.UNIFORM,
                 crossing_prob: float = 0.5) \
            -> None:
        self._f: Callable[[np.array], np.array] = f
        self._population: np.array = self._generate_start_population(
            bounds, population_size, dimension)
        self._population_quality: np.array = self._calculate_quality(
            self._population)
        self._elite_size: int = elite_size

        self._mutation_rate: float = mutation_rate
        self._mutation_prob: float = mutation_prob

        self._crossing_type: CrossingType = crossing_type
        self._crossing_prob: float = crossing_prob

        self._best_individual, self._best_quality = self._find_best(
            self._population, self._population_quality)

        self._current_mutants: np.array = None
        self._current_mutants_quality: np.array = None
        self._best_mutant: np.array = None
        self._best_mutant_quality: float = np.inf

        self._last_avg_population_quality: float = float('inf')
        self._percent_of_successes: float = 0.

        self._reset_population = partial(
            self._generate_start_population,
            bounds=bounds,
            size=population_size,
            dimension=dimension
        )

    def reset(self):
        self._population: np.array = self._reset_population()
        self._population_quality: np.array = self._calculate_quality(
            self._population)
        self._best_individual, self._best_quality = self._find_best(
            self._population, self._population_quality)
        # print('mod:', self._best_quality)

        self._current_mutants: np.array = None
        self._current_mutants_quality: np.array = None
        self._best_mutant: np.array = None
        self._best_mutant_quality: float = np.inf

        self._last_avg_population_quality: float = float('inf')
        self._percent_of_successes: float = 0.

    def step(self, n_epochs: int) -> None:
        successes = 0
        self._population_quality = self._calculate_quality(self._population)
        self._best_individual, self._best_quality = self._find_best(
            self._population, self._population_quality)
        for _ in range(n_epochs):
            self._reproduction()
            self._genetic_operations()
            self._current_mutants_quality = self._calculate_quality(
                self._current_mutants)

            self._best_mutant, self._best_mutant_quality = self._find_best(self._current_mutants,
                                                                           self._current_mutants_quality)
            if self._best_mutant_quality <= self._best_quality:
                self._best_quality = self._best_mutant_quality
                self._best_individual = self._best_mutant

            self._succession()

            new_avg_population_quality = np.average(self._population_quality)
            if new_avg_population_quality < self._last_avg_population_quality:
                successes += 1
            self._last_avg_population_quality = new_avg_population_quality
            self._current_mutants = None
            self._best_mutant = None
        self._percent_of_successes = successes / n_epochs

    def set_crossing_params(self, crossing_type: CrossingType, crossing_prob: float) -> None:
        self._crossing_type = crossing_type
        self._crossing_prob = crossing_prob

    def get_average_distance_between_individuals(self) -> float:
        def distance(x1: np.array, x2: np.array) -> float:
            return np.sqrt(np.sum(np.square(x2 - x1)))

        num_points = len(self._population)
        sum = 0.
        counter = 0
        for i in range(num_points):
            for j in range(i + 1, num_points):
                sum += distance(self._population[i], self._population[j])
                counter += 1
        return sum / counter

    def get_percent_of_successes(self) -> float:
        return self._percent_of_successes

    def _generate_start_population(self, bounds: Tuple[float, float], size: int, dimension: int) -> np.array:
        return np.array([np.random.uniform(bounds[0], bounds[1], dimension) for _ in range(size)])

    def _crossing(self) -> None:
        new_population = []
        while len(new_population) < len(self._current_mutants):
            parent_1 = random.choice(self._current_mutants)
            parent_2 = random.choice(self._current_mutants)
            if random.uniform(0, 1) < self._crossing_prob:
                chosen_crossing = self._get_crossing()
                child_1, child_2 = chosen_crossing(parent_1, parent_2)
                new_population.append(child_1)
                new_population.append(child_2)
            else:
                new_population.append(parent_1)
                new_population.append(parent_2)
        self._current_mutants = np.array(new_population)

    def _mutation(self) -> None:
        new_population: list = []
        for individual in self._current_mutants:
            if random.uniform(0, 1) < self._mutation_prob:
                new_individual = individual + \
                    self._mutation_rate * random.gauss(0, 1)
                new_population.append(new_individual)
            else:
                new_population.append(individual)
        self._current_mutants = np.array(new_population)

    def _reproduction(self) -> None:
        population_size = self._population.shape[0]
        new_population = []
        for _ in range(population_size):
            first_index = random.randint(0, population_size - 1)
            second_index = random.randint(0, population_size - 1)
            winner_index = first_index if self._population_quality[first_index] <= self._population_quality[
                second_index] else second_index
            new_population.append(self._population[winner_index])
        self._current_mutants = np.array(new_population)

    def _succession(self) -> None:
        if self._elite_size == 0:
            self._population = self._current_mutants

        self._population, self._population_quality = self._sort(
            self._population, self._population_quality)
        self._current_mutants, self._current_mutants_quality = self._sort(self._current_mutants,
                                                                          self._current_mutants_quality)
        new_population = np.concatenate(
            [self._population[:self._elite_size], self._current_mutants[:-self._elite_size]])
        new_population_quality = np.concatenate(
            [self._population_quality[:self._elite_size], self._current_mutants_quality[:-self._elite_size]])

        self._population = new_population
        self._population_quality = new_population_quality

    def _genetic_operations(self) -> None:
        self._crossing()
        self._mutation()

    def _sort(self, population: np.array, population_quality: np.array) -> Tuple[np.array, np.array]:
        zipped_lists = list(
            sorted(zip(population_quality, population), key=lambda x: x[0]))
        sorted_quality, sorted_population = [
            a for a, b in zipped_lists], [b for a, b in zipped_lists]
        return np.array(sorted_population), np.array(sorted_quality)

    def _calculate_quality(self, population: np.array) -> np.array:
        return self._f(population)

    def _find_best(self, population: np.array, population_quality: np.array) -> Tuple[np.array, float]:
        best_quality = np.min(population_quality)
        best_index: int = np.argmin(population_quality)
        best_individual: np.array = population[best_index]
        return best_individual, best_quality

    def _get_crossing(self) -> Callable[[np.array, np.array], Tuple[np.array, np.array]]:
        def single_point(parent_1: np.array, parent_2: np.array) -> Tuple[np.array, np.array]:
            """
            Krzyżowanie jednopunktowe. Losujemy punkt rozcięcia. Zamieniamy odcięte części
            """
            cut_index = random.randint(0, len(parent_1) - 1)
            child_1 = np.concatenate(
                [parent_1[:cut_index], parent_2[cut_index:]])
            child_2 = np.concatenate(
                [parent_2[:cut_index], parent_1[cut_index:]])
            return child_1, child_2

        def uniform(parent_1: np.array, parent_2: np.array) -> Tuple[np.array, np.array]:
            """
            Każdy gen jest losowany od jednego z rodziców.
            """
            child_1 = []
            child_2 = []
            for i in range(len(parent_1)):
                if random.randint(0, 1) == 0:
                    child_1.append(parent_1[i])
                    child_2.append(parent_2[i])
                else:
                    child_1.append(parent_2[i])
                    child_2.append(parent_1[i])
            return np.array(child_1), np.array(child_2)

        def simple_intermediate(parent_1: np.array, parent_2: np.array) -> Tuple[np.array, np.array]:
            """
            Losujemy wagę, następnie liczymy średnią ważoną genów od obu rodziców
            """
            weight = random.uniform(0, 1)
            child_1 = weight * parent_1 + (1 - weight) * parent_2
            child_2 = weight * parent_2 + (1 - weight) * parent_1
            return child_1, child_2

        def complex_intermediate(parent_1: np.array, parent_2: np.array) -> Tuple[np.array, np.array]:
            """
            Losujemy wektor wag, średnią ważoną liczymy według wagi dla danego indeksu
            """
            weights = np.random.uniform(0, 1, len(parent_1))
            child_1 = weights * parent_1 + \
                (np.ones(len(parent_1)) - weights) * parent_2
            child_2 = weights * parent_2 + \
                (np.ones(len(parent_1)) - weights) * parent_1
            return child_1, child_2

        functions: Dict[CrossingType, Callable[[np.array, np.array], Tuple[np.array, np.array]]] = {
            CrossingType.SINGLE_POINT: single_point,
            CrossingType.UNIFORM: uniform,
            CrossingType.SIMPLE_INTERMEDIATE: simple_intermediate,
            CrossingType.COMPLEX_INTERMEDIATE: complex_intermediate
        }

        return functions.get(self._crossing_type)
