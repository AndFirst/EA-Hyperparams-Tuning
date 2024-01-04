import csv
from copy import deepcopy
from typing import Any, Iterable, SupportsFloat
import gymnasium as gym
import numpy as np
from src.EA import EvolutionaryAlgorithm, CrossingType
from tabulate import tabulate
from typing import Callable
from src.constants import ACTION_DIM, OBSERVATION_DIM


def _to1DSpace(space: tuple[int, int], coord: tuple[int, int]) -> int:
    return coord[0] * space[1] + coord[1]


def _to2DSpace(space: tuple[int, int], coord: int) -> tuple[int, int]:
    return (coord // space[1], coord % space[1])


class EvolutionaryEnv(gym.Env):
    def __init__(self, max_steps: int, step_size: int, model: EvolutionaryAlgorithm, reward_functions: Iterable[Callable[[dict], float]]) -> None:
        self._success_bins = np.linspace(
            0, 1, num=OBSERVATION_DIM[0] - 1)
        self._distance_bins = np.linspace(
            0, 10, num=OBSERVATION_DIM[1] - 1)

        self._crossover_probabilities = np.linspace(
            0, 1, num=ACTION_DIM[1])

        self.action_space = gym.spaces.Discrete(
            ACTION_DIM[0] * ACTION_DIM[1])
        self.observation_space = gym.spaces.Discrete(
            OBSERVATION_DIM[0] * OBSERVATION_DIM[1])

        self._max_steps = max_steps
        self._step_size = step_size
        self._current_step = 0
        self._start_model = deepcopy(model)

        self._last_distance_bin = 0
        self._last_successes_bin = 0
        self._last_best_quality = np.inf

        self._model = model
        self._reward_functions = reward_functions

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self._last_best_quality = self._model._best_quality

        self._last_successes_bin = np.digitize(
            self._model.get_percent_of_successes(), self._success_bins)

        self._last_distance_bin = np.digitize(
            self._model.get_average_distance_between_individuals(), self._distance_bins)

        selected_crossover, crossover_probability = _to2DSpace(
            ACTION_DIM, action)
        crossover_probability = float(
            self._crossover_probabilities[crossover_probability])
        selected_crossover = CrossingType(selected_crossover)

        self._model.step(self._step_size)
        self._model.set_crossing_params(
            selected_crossover, crossover_probability)
        success = np.digitize(
            self._model.get_percent_of_successes(), self._success_bins)
        distance = np.digitize(
            self._model.get_average_distance_between_individuals(), self._distance_bins)

        reward = self._calculate_reward(success, distance)
        terminated = self._current_step >= self._max_steps

        self._current_step += 1

        return _to1DSpace(OBSERVATION_DIM,
                          (success, distance)), reward, terminated, False, {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self._current_step = 0
        self._model = self._start_model
        self._last_best_quality = np.inf
        self._last_successes_bin = 0
        self._last_distance_bin = 0

        success = np.digitize(
            self._model.get_percent_of_successes(), self._success_bins)
        distance = np.digitize(
            self._model.get_average_distance_between_individuals(), self._distance_bins)

        return _to1DSpace(OBSERVATION_DIM, (success, distance)), {}

    def _calculate_reward(self, success, distance):
        kwargs = {"current_distance_bin": distance,
                  "last_distance_bin": self._last_distance_bin,

                  "current_best_quality": self._model._best_quality,
                  "last_best_quality": self._last_best_quality,

                  "last_successes_bin": self._last_successes_bin,
                  "current_successes_bin": success,
                  }
        return sum(function(kwargs) for function in self._reward_functions)

    def print_Q(self, Q: np.array):
        headers = ["state/action"]

        def _1DtoAction(i: int):
            selected_crossover, crossover_probability = _to2DSpace(
                ACTION_DIM, i)
            crossover_probability = float(
                self._crossover_probabilities[crossover_probability])
            # selected_crossover = CrossingType(selected_crossover)
            return (selected_crossover, crossover_probability)

        headers.extend([_1DtoAction(x) for x in range(0, Q.shape[1])])
        table = []
        for i, row in enumerate(Q):
            success, distance = _to2DSpace(OBSERVATION_DIM, i)
            formatted_row = []
            formatted_row.append(f"({success}, {distance})")
            formatted_row.extend(row)
            table.append(formatted_row)
        print(tabulate(table, headers=headers))

    def export_Q_to_csv(self, Q: np.array, csv_file_path: str):
        headers = ["state/action"]

        def _1DtoAction(i: int):
            selected_crossover, crossover_probability = _to2DSpace(
                ACTION_DIM, i)
            crossover_probability = float(
                self._crossover_probabilities[crossover_probability])
            return (selected_crossover, crossover_probability)

        headers.extend([_1DtoAction(x) for x in range(0, Q.shape[1])])
        table = []

        for i, row in enumerate(Q):
            success, distance = _to2DSpace(OBSERVATION_DIM, i)
            formatted_row = []
            formatted_row.append(f"({success}, {distance})")
            formatted_row.extend(row)
            table.append(formatted_row)

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headers)
            writer.writerows(table)

        print(f"Dane zostały zapisane do pliku CSV: {csv_file_path}")
