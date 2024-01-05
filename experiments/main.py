import json
import pandas as pd
import numpy as np


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
    
if __name__ == "__main__":
    calculate_base_results_statistics()
    calculate_q_results_statistics()
    