from experiments.statistics import (calculate_q_results_statistics,
                                    calculate_base_results_statistics,
                                    plot_history_data_of_base_results,
                                    plot_history_data_of_q_results,
                                    plot_cross_q_history,
                                    calculate_cross_q_stats)
from experiments.experiments import base_results, q_learning_results, cross_q_table_use_results

if __name__ == '__main__':
    # EKSPERYMENT BAZOWY
    base_results()
    calculate_base_results_statistics()
    plot_history_data_of_base_results()

    # EKSPERYMENT PORÓWNANIA FUNKCJI NAGRODY
    q_learning_results()
    calculate_q_results_statistics()
    plot_history_data_of_q_results()

    # EKSPERYMENT UŻYCIA TABLICY Q NA INNEJ FUNKCJI
    cross_q_table_use_results()
    calculate_cross_q_stats()
    plot_cross_q_history()
