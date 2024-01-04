from src.EA import EvolutionaryAlgorithm, CrossingType
from src.environment import EvolutionaryEnv
from src.q_learning import q_learning_greedy, use_trained_q_table
from src.reward_functions import distance_reward, quality_reward, successes_reward
from cec2017.functions import f4, f20
from collections import defaultdict
import json
import time

# parametry środowiska
ENV_STEP_SIZE = 20  # z tylu rund środowisko zbiera dane o stanie algorytmu ewolucyjnego
ENV_MAX_STEPS = 20  # tyle razy algorytm ewolucyjny uruchamiany jest na ENV_STEP_SIZE rund

# parametry Q-learningu
LEARNING_RATE = 0.1
EPSILON = 0.1
DISCOUNT_FACTOR = 0.9
Q_LEARNING_ITERATIONS = 50

# parametry algorytmu ewolucyjnego
DIMENSION = 10
POPULATION_SIZE = 20
ELITE_SIZE = 2

# parametry eksperymentów
FUNCTIONS = f4, f20
N_REPEATS = 25  # ilość powtórzeń każdego eksperymentu
CROSSING_TYPES = list(CrossingType)
CROSSING_PROBS = 0.0, 0.25, 0.5, 0.75, 1.0
FUNCTIONS_COMBINATIONS = ((distance_reward, quality_reward, successes_reward),
                          (distance_reward, quality_reward,),
                          (distance_reward, successes_reward),
                          (quality_reward, successes_reward),
                          (distance_reward,),
                          (quality_reward,),
                          (successes_reward,),)

# do porównania z algorytmem bez uczenia ze wzmocnieniem
N_EPOCHS = ENV_MAX_STEPS * ENV_STEP_SIZE


def base_results():
    """
    Zwraca wyniki uruchomień algorytmu ewolucyjnego dla różnych kombinacji parametrów
    bez wspomagania algorytmem uczenia ze wzmocnieniem
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_runs = len(FUNCTIONS) * len(CROSSING_TYPES) * \
        len(CROSSING_PROBS) * N_REPEATS
    i = 1
    start_time = time.time()
    for crossing_type in CROSSING_TYPES:
        for prob in CROSSING_PROBS:
            for function in FUNCTIONS:
                for run in range(N_REPEATS):
                    print(f'{i}/{all_runs} elapsed_time: {time.time()- start_time}')
                    model = EvolutionaryAlgorithm(
                        function, DIMENSION, POPULATION_SIZE, ELITE_SIZE, crossing_type=crossing_type, crossing_prob=prob)
                    model.step(N_EPOCHS)
                    results[crossing_type.name][function.__name__][str(prob)].append(
                        model._best_quality)
                    i += 1
    print(f'{i}/{all_runs} elapsed_time: {time.time()- start_time}')
    with open('results/base_results.json', 'w') as file:
        json.dump(results, file)


def q_learning_results():
    """
    Zwraca wyniki uruchomień algorytmu ewolucyjnego, którego parametry są 
    wybierane na podstawie tablicy Q nauczonej na różnych kombinacjach funkcji nagrody
    """
    results = defaultdict(lambda: defaultdict(list))
    all_runs = len(FUNCTIONS) * len(FUNCTIONS_COMBINATIONS) * N_REPEATS
    i = 1
    start_time = time.time()

    for function in FUNCTIONS:
        for c, combination in enumerate(FUNCTIONS_COMBINATIONS):
            for run in range(N_REPEATS):
                print(f'{i}/{all_runs} elapsed_time: {time.time() - start_time}')
                model = EvolutionaryAlgorithm(
                    function, DIMENSION, POPULATION_SIZE, ELITE_SIZE)
                env = EvolutionaryEnv(
                    ENV_MAX_STEPS, ENV_STEP_SIZE, model, combination)
                Q = q_learning_greedy(
                    env, LEARNING_RATE, DISCOUNT_FACTOR, Q_LEARNING_ITERATIONS, EPSILON, 10)

                use_trained_q_table(env, Q)

                results[function.__name__][f"comb_{c}"].append(
                    env._model._best_quality)
                i += 1
    print(f'{i}/{all_runs} elapsed_time: {time.time()- start_time}')
    with open('data/q_results.json', 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    q_learning_results()
    base_results()
