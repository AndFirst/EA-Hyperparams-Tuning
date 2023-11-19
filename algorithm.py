import random
from enum import Enum
from typing import Callable, Tuple
import numpy as np


class CrossingType(Enum):
    SINGLE_POINT = 0
    UNIFORM = 1
    SIMPLE_INTERMEDIATE = 2
    COMPLEX_INTERMEDIATE = 3


def single_point_crossing(parent_1: np.array, parent_2: np.array) -> Tuple[np.array, np.array]:
    assert len(parent_1) == len(parent_2)
    cut_index = random.randint(0, len(parent_1) - 1)
    child_1 = np.concatenate([parent_1[:cut_index], parent_2[cut_index:]])
    child_2 = np.concatenate([parent_2[:cut_index], parent_1[cut_index:]])
    return child_1, child_2


def uniform_crossing(parent_1: np.array, parent_2: np.array) -> Tuple[np.array, np.array]:
    assert len(parent_1) == len(parent_2)
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


def simple_intermediate_crossing(parent_1: np.array, parent_2: np.array) -> Tuple[np.array, np.array]:
    assert len(parent_1) == len(parent_2)
    weight = random.uniform(0, 1)
    child_1 = weight * parent_1 + (1 - weight) * parent_2
    child_2 = weight * parent_2 + (1 - weight) * parent_1
    return child_1, child_2


def complex_intermediate_crossing(parent_1: np.array, parent_2: np.array) -> Tuple[np.array, np.array]:
    assert len(parent_1) == len(parent_2)
    weights = np.random.uniform(0, 1, len(parent_1))
    child_1 = weights * parent_1 + (np.ones(len(parent_1)) - weights) * parent_2
    child_2 = weights * parent_2 + (np.ones(len(parent_1)) - weights) * parent_1
    return child_1, child_2


def get_crossing(crossing_type: CrossingType) -> Callable:
    if crossing_type == CrossingType.SINGLE_POINT:
        return single_point_crossing
    elif crossing_type == CrossingType.UNIFORM:
        return uniform_crossing
    elif crossing_type == CrossingType.SIMPLE_INTERMEDIATE:
        return simple_intermediate_crossing
    elif crossing_type == CrossingType.COMPLEX_INTERMEDIATE:
        return complex_intermediate_crossing
    else:
        raise ValueError(f"Crossing type: {crossing_type.name} not exist.")


def is_solved(t: int, t_max: int) -> bool:
    return not t < t_max


def calculate_quality(f: Callable, population: np.array) -> np.array:
    return f(population)


def find_best(population: np.array, population_quality: np.array) -> Tuple[np.array, float]:
    best_quality = np.min(population_quality)
    best_index = np.argmin(population_quality)
    best_individual = population[best_index]
    return best_individual, best_quality


def reproduction(population: np.ndarray, population_quality: np.array) -> np.array:
    population_size = population.shape[0]
    new_population = []
    for _ in range(population_size):
        first_index = random.randint(0, population_size - 1)
        second_index = random.randint(0, population_size - 1)
        winner_index = first_index if population_quality[first_index] >= population_quality[
            second_index] else second_index
        new_population.append(population[winner_index])
    return np.array(new_population)


def crossing(population: np.array, crossing_type: CrossingType, crossing_probability: float) -> np.array:
    new_population = []
    while len(new_population) < len(population):
        parent_1 = random.choice(population)
        parent_2 = random.choice(population)
        if random.uniform(0, 1) < crossing_probability:
            new_population.append(parent_1)
            new_population.append(parent_2)
        else:
            chosen_crossing = get_crossing(crossing_type)
            child_1, child_2 = chosen_crossing(parent_1, parent_2)
            new_population.append(child_1)
            new_population.append(child_2)
    return np.array(new_population)


def mutation(population: np.array, mutation_value: float, mutation_probability: float) -> np.array:
    new_population = []
    for individual in population:
        if random.uniform(0, 1) < mutation_probability:
            new_individual = individual + mutation_value * random.gauss(0, 1)
            new_population.append(new_individual)
        else:
            new_population.append(individual)
    return np.array(new_population)


def genetic_operations(population: np.array, mutation_value: float, mutation_probability: float,
                       crossing_type: CrossingType,
                       crossing_probability: float) -> np.array:
    crossed_population = crossing(population, crossing_type, crossing_probability)
    mutants = mutation(crossed_population, mutation_value, mutation_probability)
    return mutants


def sort_population(population: np.array, quality: np.array) -> Tuple[np.array, np.array]:
    zipped_lists = list(sorted(zip(quality, population), key=lambda x: x[0]))
    sorted_quality, sorted_population = [a for a, b in zipped_lists], [b for a, b in zipped_lists]
    return np.array(sorted_population), np.array(sorted_quality)


def succession(population: np.array, mutants: np.array, population_quality: np.array, mutants_quality: np.array,
               elite_size: float) -> Tuple[np.array, np.array]:
    if elite_size == 0:
        return mutants, mutants_quality

    population, population_quality = sort_population(population, population_quality)
    mutants, mutants_quality = sort_population(mutants, mutants_quality)

    new_population = np.concatenate([population[:elite_size], mutants[:-elite_size]])
    new_population_quality = np.concatenate([population_quality[:elite_size], mutants_quality[:-elite_size]])

    return new_population, new_population_quality


def evolutionary_algorithm(f: Callable, population: np.array, mutation_value: float = None,
                           mutation_probability: float = None, elite_size: int = None,
                           t_max: int = None, crossing_type: CrossingType = None, crossing_probability: float = None) -> \
        Tuple[np.array, float]:
    t = 0
    population_quality = calculate_quality(f, population)
    best_individual, best_quality = find_best(population, population_quality)
    while not is_solved(t, t_max):
        current_reproduction = reproduction(population, population_quality)
        current_mutants = genetic_operations(current_reproduction, mutation_value, mutation_probability, crossing_type,
                                             crossing_probability)
        mutants_quality = calculate_quality(f, current_mutants)
        best_mutant, best_mutant_quality = find_best(current_mutants, mutants_quality)
        if best_mutant_quality <= best_quality:
            best_individual = best_mutant
            best_quality = best_mutant_quality
        population, population_quality = succession(population, current_mutants, population_quality, mutants_quality,
                                                    elite_size)
        t += 1
    return best_individual, best_quality
