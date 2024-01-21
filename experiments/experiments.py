import numpy as np
from src.EA import EvolutionaryAlgorithm, CrossingType
from src.environment import EvolutionaryEnv
from src.q_learning import q_learning_greedy, use_trained_q_table, import_Q_from_csv
from src.reward_functions import distance_reward, quality_reward, successes_reward
from cec2017.functions import *
from collections import defaultdict
import json
import time

# parametry środowiska
ENV_STEP_SIZE = 10  # z tylu rund środowisko zbiera dane o stanie algorytmu ewolucyjnego
ENV_MAX_STEPS = 30  # tyle razy algorytm ewolucyjny uruchamiany jest na ENV_STEP_SIZE rund

# parametry Q-learningu
LEARNING_RATE = 0.1
EPSILON = 0.1
DISCOUNT_FACTOR = 0.9
Q_LEARNING_ITERATIONS = 500

# parametry algorytmu ewolucyjnego
DIMENSION = 10
POPULATION_SIZE = 20
ELITE_SIZE = 2

# parametry eksperymentów
FUNCTIONS = f4, f9
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
    history_results = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    all_runs = len(FUNCTIONS) * len(CROSSING_TYPES) * \
        len(CROSSING_PROBS) * N_REPEATS
    i = 1
    start_time = time.time()
    for crossing_type in CROSSING_TYPES:
        for prob in CROSSING_PROBS:
            for function in FUNCTIONS:
                history_for_n_repeats = list()
                for run in range(N_REPEATS):
                    current_run_history = list()
                    print(f'{i}/{all_runs} elapsed_time: {time.time()- start_time}')
                    model = EvolutionaryAlgorithm(
                        function, DIMENSION, POPULATION_SIZE, ELITE_SIZE, crossing_type=crossing_type, crossing_prob=prob)
                    for _ in range(ENV_MAX_STEPS):
                        model.step(ENV_STEP_SIZE)
                        current_run_history.append(model._best_quality)
                    results[crossing_type.name][function.__name__][str(prob)].append(
                        model._best_quality)
                    history_for_n_repeats.append(np.array(current_run_history))
                    i += 1
                history_for_n_repeats = np.array(history_for_n_repeats)
                history_results[crossing_type.name][function.__name__][str(
                    prob)] = list(np.mean(history_for_n_repeats, axis=0))
    print(f'{i}/{all_runs} elapsed_time: {time.time()- start_time}')
    with open('results/base_results.json', 'w') as file:
        json.dump(results, file)
    with open('results/base_history_results.json', 'w') as file:
        json.dump(history_results, file)


def q_learning_results():
    """
    Zwraca wyniki uruchomień algorytmu ewolucyjnego, którego parametry są 
    wybierane na podstawie tablicy Q nauczonej na różnych kombinacjach funkcji nagrody
    """
    results = defaultdict(lambda: defaultdict(list))
    history_results = defaultdict(lambda: defaultdict(list))
    all_runs = len(FUNCTIONS) * len(FUNCTIONS_COMBINATIONS) * N_REPEATS
    i = 1
    start_time = time.time()
    for function in FUNCTIONS:
        best_q_table = None
        best_quality = float(np.inf)
        best_env = None
        for c, combination in enumerate(FUNCTIONS_COMBINATIONS):
            history_for_n_repeats = list()
            for run in range(N_REPEATS):
                print(f'{i}/{all_runs} elapsed_time: {time.time() - start_time}')
                model = EvolutionaryAlgorithm(
                    function, DIMENSION, POPULATION_SIZE, ELITE_SIZE)
                env = EvolutionaryEnv(
                    ENV_MAX_STEPS, ENV_STEP_SIZE, model, combination)
                Q = q_learning_greedy(
                    env, LEARNING_RATE, DISCOUNT_FACTOR, Q_LEARNING_ITERATIONS, EPSILON, 10)

                current_run_history = use_trained_q_table(env, Q)
                history_for_n_repeats.append(current_run_history)
                quality = env._model._best_quality

                if quality < best_quality:
                    best_quality = quality
                    best_q_table = Q
                    best_env = env
                results[function.__name__][f"comb_{c}"].append(quality)
                i += 1
            history_for_n_repeats = np.array(history_for_n_repeats)
            history_results[function.__name__][f"comb_{c}"] = list(np.mean(
                history_for_n_repeats, axis=0))
        best_env.export_Q_to_csv(
            best_q_table, f"results/best_q_{function.__name__}.csv")

    print(f'{i}/{all_runs} elapsed_time: {time.time()- start_time}')
    with open('results/q_results.json', 'w') as file:
        json.dump(results, file)

    with open('results/q_history_results.json', 'w') as file:
        json.dump(history_results, file)


def cross_q_table(q_path, function, reward, output):
    Q = import_Q_from_csv(q_path)
    results = []
    history = []
    for i in range(N_REPEATS):
        model = EvolutionaryAlgorithm(
            function, DIMENSION, POPULATION_SIZE, ELITE_SIZE)
        env = EvolutionaryEnv(ENV_MAX_STEPS, ENV_STEP_SIZE, model, reward)
        current_run_history = use_trained_q_table(env, Q)
        quality = env._model._best_quality
        results.append(quality)
        history.append(current_run_history)
    history = list(np.mean(history, axis=0))
    with open(f'results/{output}_history.json', 'w') as file:
        json.dump(history, file)
    with open(f'results/{output}_results.json', 'w') as file:
        json.dump(results, file)


def cross_q_table_use_results():
    best_combination = FUNCTIONS_COMBINATIONS[0]
    function = f4
    q_path = 'results/best_q_f9.csv'
    cross_q_table(q_path, function, best_combination, 'f4_q9')

    function = f9
    q_path = 'results/best_q_f4.csv'
    cross_q_table(q_path, function, best_combination, 'f9_q4')
