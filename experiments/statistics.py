import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_base_results_statistics():
    """
    Oblicza dane statystyczne i zapisuje je do pliku csv
    """
    with open('results/base_results.json', 'r') as file:
        results = json.load(file)
    data = []

    for crossing_type, functions in results.items():
        for function_name, prob_values in functions.items():
            for crossing_prob, values in prob_values.items():
                for repeat, value in enumerate(values, start=1):
                    data.append((
                        function_name,
                        crossing_type,
                        float(crossing_prob),
                        repeat,
                        value
                    ))

    df = pd.DataFrame(
        data, columns=['Function', 'CrossingType', 'CrossingProb', 'Repeats', 'Value'])

    # Sortowanie ramki danych według funkcji, typu krzyżowania i prawdopodobieństwa krzyżowania
    sorted_df = df.sort_values(by=['Function', 'CrossingType', 'CrossingProb'])

    # Grupowanie i obliczanie statystyk
    grouped_df = sorted_df.groupby(['Function', 'CrossingType', 'CrossingProb']).agg({
        'Value': ['mean', 'min', 'max', 'std']
    }).reset_index()

    # Zaokrąglenie wartości do 2 miejsc po przecinku
    grouped_df.columns = [' '.join(col).strip()
                          for col in grouped_df.columns.values]
    grouped_df['Value mean'] = grouped_df['Value mean'].round(2)
    grouped_df['Value min'] = grouped_df['Value min'].round(2)
    grouped_df['Value max'] = grouped_df['Value max'].round(2)
    grouped_df['Value std'] = grouped_df['Value std'].round(2)

    # Zapisanie do pliku csv
    grouped_df.to_csv("results/base_data.csv", index=False)


def calculate_q_results_statistics():
    """
    Oblicza dane statystyczne i zapisuje je do pliku csv
    """
    with open('results/q_results.json', 'r') as file:
        results = json.load(file)
    data = []

    for function_name, function_values in results.items():
        for reward_combination, values in function_values.items():
            for repeat, value in enumerate(values, start=1):
                data.append((
                    function_name,
                    reward_combination,
                    repeat,
                    value
                ))
    df = pd.DataFrame(
        data, columns=['Function', 'RewardCombination', 'Repeats', 'Value'])

    sorted_df = df.sort_values(by=['Function', 'RewardCombination', 'Repeats'])

    # Grupowanie i obliczanie statystyk
    grouped_df = sorted_df.groupby(['Function', 'RewardCombination']).agg({
        'Value': ['mean', 'min', 'max', 'std']
    }).reset_index()

    # Zaokrąglenie wartości do 2 miejsc po przecinku
    grouped_df.columns = [' '.join(col).strip()
                          for col in grouped_df.columns.values]
    grouped_df['Value mean'] = grouped_df['Value mean'].round(2)
    grouped_df['Value min'] = grouped_df['Value min'].round(2)
    grouped_df['Value max'] = grouped_df['Value max'].round(2)
    grouped_df['Value std'] = grouped_df['Value std'].round(2)

    # Zapisanie do pliku csv
    grouped_df.to_csv("results/q_data.csv", index=False)


def plot_history_data_of_base_results():
    with open('results/base_history_results.json', 'r') as file:
        data = json.load(file)
    for crossing_type, crossing_type_data in data.items():
        plt.figure(figsize=(15, 6))

        for idx, (function, function_data) in enumerate(crossing_type_data.items(), 1):
            plt.subplot(1, 2, idx)
            for crossing_prob, crossing_prob_data in function_data.items():
                plt.plot(range(0, len(crossing_prob_data)*20, 20),
                         crossing_prob_data, label=f'P-swo krzyżowania: {crossing_prob}')
            plt.title(
                f"Funkcja: {function}\nTyp krzyżowania: {crossing_type}")
            plt.legend()
            plt.xlabel("Ilość epok")
            plt.ylabel("Wartość funkcji")
        plt.tight_layout(pad=0)
        plt.savefig(f'plots/wykresy_{crossing_type}.png')


def plot_history_data_of_q_results():
    with open('results/q_history_results.json', 'r') as file:
        data = json.load(file)
    for function, function_data in data.items():
        plt.figure(figsize=(15, 6))
        for combination, combination_data in function_data.items():
            plt.plot(range(0, len(combination_data)*20, 20),
                     combination_data, label=f'Kombinacja: {combination[-1]}')
        plt.title(f"Funkcja: {function}")
        plt.legend()
        plt.xlabel("Ilość epok")
        plt.ylabel("Wartość funkcji")
        plt.tight_layout(pad=0)
        plt.savefig(f'plots/wykresy_q_{function}')


def plot_cross_q_history():
    def plot_one_function(file_path, output_name):
        with open(file_path, 'r') as file:
            data = json.load(file)

        plt.figure(figsize=(15, 6))
        plt.plot(range(0, len(data)*20, 20), data)
        plt.xlabel("Ilość epok")
        plt.ylabel("Wartość funkcji")
        plt.tight_layout(pad=0)
        plt.savefig(f'plots/{output_name}.png')

    plot_one_function('results/f4_q9_history.json', 'f4_q9_history_plot')
    plot_one_function('results/f9_q4_history.json', 'f9_q4_history_plot')


def calculate_cross_q_stats():
    def calculate_stats(file_path, output_name):
        with open(file_path, 'r') as file:
            data = json.load(file)
        values = np.array(data)

        # Średnia
        mean_value = np.mean(values)

        # Minimum
        min_value = np.min(values)

        # Maksimum
        max_value = np.max(values)

        # Odchylenie standardowe
        std_value = np.std(values)

        # Utwórz DataFrame z wynikami
        df = pd.DataFrame({
            'Statystyka': ['Value mean', 'Value min', 'Value max', 'Value std'],
            'Wartość': [mean_value, min_value, max_value, std_value]
        })

        # Zapisz do pliku CSV
        output_csv_path = f"results/{output_name}_stats.csv"
        df.to_csv(output_csv_path, index=False)

    calculate_stats('results/f4_q9_results.json', 'f4_q9')
    calculate_stats('results/f4_q9_results.json', 'f9_q4')
