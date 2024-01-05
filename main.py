from experiments.statistics import calculate_q_results_statistics, calculate_base_results_statistics
from experiments.experiments import base_results, q_learning_results

if __name__ == '__main__':
    # q_learning_results()
    # base_results()
    
    calculate_base_results_statistics()
    calculate_q_results_statistics()
    