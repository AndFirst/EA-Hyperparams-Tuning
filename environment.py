import random
from copy import copy, deepcopy
from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from EA import EvolutionaryAlgorithm, CrossingType
from cec2017.functions import *
from q_learning import q_learning_greedy, show_model
from tabulate import tabulate


def _to1DSpace(space: tuple[int, int], coord: tuple[int, int]) -> int:
    return coord[0] * space[1] + coord[1]


def _to2DSpace(space: tuple[int, int], coord: int) -> tuple[int, int]:
    return (coord // space[1], coord % space[1])


class EvolutionaryEnv(gym.Env):
    def __init__(self, max_steps: int, model: EvolutionaryAlgorithm) -> None:
        self._action_dim = (len(CrossingType), 3)
        self._observation_dim = (5, 5)

        self._success_bins = np.linspace(0, 1, num=self._observation_dim[0] - 1)
        self._distance_bins = np.linspace(0, 1000, num=self._observation_dim[1] - 1)
        # self._distance_bins = np.logspace(0, 1000, num=self._observation_dim[1] - 1)

        self._crossover_probabilities = np.linspace(0, 1, num=self._action_dim[1])

        self.action_space = gym.spaces.Discrete(self._action_dim[0] * self._action_dim[1])
        self.observation_space = gym.spaces.Discrete(self._observation_dim[0] * self._observation_dim[1])

        self._max_steps = max_steps
        self._current_step = 0
        self._start_model = deepcopy(model)

        self._last_best_quality = np.inf
        self._model = model

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self._last_best_quality = self._model._best_quality
        selected_crossover, crossover_probability = _to2DSpace(self._action_dim, action)
        # print(selected_crossover, crossover_probability )
        crossover_probability = float(self._crossover_probabilities[crossover_probability])
        selected_crossover = CrossingType(selected_crossover)

        STEP = 20
        self._model.step(STEP)
        self._model.set_crossing_params(selected_crossover, crossover_probability)
        success = np.digitize(self._model.get_percent_of_successes(), self._success_bins)
        distance = np.digitize(self._model.get_average_distance_between_individuals(), self._distance_bins)

        reward = self._calculate_reward(success, distance)
        terminated = self._current_step >= self._max_steps

        self._current_step += 1

        return _to1DSpace(self._observation_dim,
                          (success, distance)), reward, terminated, False, {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self._current_step = 0
        self._model = self._start_model

        success = np.digitize(self._model.get_percent_of_successes(), self._success_bins)
        distance = np.digitize(self._model.get_average_distance_between_individuals(), self._distance_bins)

        return _to1DSpace(self._observation_dim, (success, distance)), {}

    def _calculate_reward(self, success, distance):
        optimum = 700.
        result = optimum - self._last_best_quality
        # if result < -100000:
        #     result = -100000
        return result
    
    def print_Q(self, Q: np.array):
        headers = ["state/action"]

        def _1DtoAction(i: int):
            selected_crossover, crossover_probability = _to2DSpace(self._action_dim, i)
            crossover_probability = float(self._crossover_probabilities[crossover_probability])
            # selected_crossover = CrossingType(selected_crossover)
            return (selected_crossover, crossover_probability)

        headers.extend([_1DtoAction(x) for x in range(0, Q.shape[1])])
        table = []
        for i, row in enumerate(Q):
            success, distance = _to2DSpace(self._observation_dim, i)
            formatted_row = []
            formatted_row.append(f"({success}, {distance})")
            formatted_row.extend(row)
            table.append(formatted_row)
        print(tabulate(table, headers=headers))



if __name__ == "__main__":
    model = EvolutionaryAlgorithm(f7, 10, 20, 2)
    env = EvolutionaryEnv(30, model)
    Q = q_learning_greedy(env, 0.1, 0.9, 10, 0.9, 1)
    print(Q.shape)
    # plt.show()
    # print(list(Q))
    env.print_Q(Q)
