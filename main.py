import numpy as np
from cec2017.functions import *


def generate_start_population(population_size: int, dimension: int, min_range: float, max_range: float) -> np.array:
    return np.array([np.random.uniform(min_range, max_range, dimension) for _ in range(population_size)])


if __name__ == "__main__":
    population = generate_start_population(200, 10, -50, 50)

    result = evolutionary_algorithm(f=f7,
                                    population=population,
                                    mutation_value=50.,
                                    mutation_probability=1,
                                    elite_size=20,
                                    t_max=1000,
                                    crossing_type=CrossingType.COMPLEX_INTERMEDIATE,
                                    crossing_probability=0
                                    )
    print(result)