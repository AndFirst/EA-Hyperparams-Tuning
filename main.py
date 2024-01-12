from experiments.statistics import (calculate_q_results_statistics,
                                    calculate_base_results_statistics,
                                    plot_history_data_of_base_results,
                                    plot_history_data_of_q_results)
from experiments.experiments import base_results, q_learning_results

if __name__ == '__main__':

    # q_learning_results()
    calculate_q_results_statistics()
    plot_history_data_of_q_results()
    base_results()
    calculate_base_results_statistics()
    plot_history_data_of_base_results()
