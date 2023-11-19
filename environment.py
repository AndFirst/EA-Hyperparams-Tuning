from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
from cec2017.functions import *
from algorithm import evolutionary_algorithm, CrossingType

def generate_start_population(population_size: int, dimension: int, min_range: float, max_range: float) -> np.array:
    return np.array([np.random.uniform(min_range, max_range, dimension) for _ in range(population_size)])

class EvolutionaryEnv(gym.Env):
    def __init__(self) -> None:
        # Przykladowe wartosci graniczne i liczba kubelkow dla dyskretyzacji stanow
        self.success_bins = np.linspace(0, 100, num=10)
        self.distance_bins = np.linspace(0, 10, num=10)

        # Przykladowe prawdopodobienstwa mutacji
        self.mutation_probabilities = [0.1, 0.3, 0.5, 0.7]

        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(4), # 4 rodzaje krzyzowania
            gym.spaces.Discrete(len(self.mutation_probabilities)) # rodzaje p. mutacji
        ))
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(len(self.success_bins) - 1),
            gym.spaces.Discrete(len(self.distance_bins) - 1)
        ))

        self.current_success = 0 # do liczenia procent sukcesow
        self.current_distance = 0 # do liczenia sredniej odleglosci
        self.population = generate_start_population(200, 10, -50, 50) # aktualny stan algorytmu ewolucyjnego

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        selected_crossover = action[0]
        selected_mutation = action[1]

        # Aktualizuj current_success, current_distance
        
        reward = self._calculate_reward()

        return (self.current_success, self.current_distance), reward, False, False, {}
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.current_success = 0
        self.current_distance = 0
        return (self.current_success, self.current_distance), {}
    
    def _calculate_reward(self):
        # do uzupelnienia
        pass


