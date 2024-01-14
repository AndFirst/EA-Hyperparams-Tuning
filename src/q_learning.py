import gymnasium as gym
import numpy as np
import random


def q_learning_greedy(env: gym.Env, learning_rate, discount_factor, iterations, epsilon, interval):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    rewards = []
    episodes = []
    avg_reward = 0

    # Loop through episodes
    for episode in range(iterations):
        state, _ = env.reset()

        episode_reward = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Update Q-table using Q-learning rule
            Q[state, action] = Q[state, action] + learning_rate * \
                (reward + discount_factor *
                 np.max(Q[next_state, :]) - Q[state, action])
            state = next_state
            episode_reward += reward

        avg_reward += episode_reward
        if episode % interval == 0:
            avg_reward /= interval
            rewards.append(avg_reward)
            episodes.append(episode)
            avg_reward = 0
    return Q


def use_trained_q_table(env: gym.Env, Q: np.ndarray) -> np.ndarray:
    """
    Uruchamia algorytm ewolucyjny i aktualizuje jego parametry zgodnie z otrzymaną tablicą Q.
    Zwraca historię najlepszych punktów.
    """
    results_history = list()
    state, _ = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = np.argmax(Q[state, :])
        next_state, _, terminated, truncated, _ = env.step(action)
        results_history.append(env._model._best_quality)
        state = next_state
    return np.array(results_history)
