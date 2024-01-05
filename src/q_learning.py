import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
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
            # print(f"{episode}: {avg_reward}")
            avg_reward = 0

    return Q


def q_learning_boltzmann(env: gym.Env, learning_rate, discount_factor, iterations, temperature, interval):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    exp_Q = np.exp(Q / temperature)

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

            probs = exp_Q[state, :] / np.sum(exp_Q[state, :])
            action = np.random.choice(np.arange(env.action_space.n), p=probs)

            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Update Q-table using Q-learning rule
            Q[state, action] = Q[state, action] + learning_rate * \
                (reward + discount_factor *
                 np.max(Q[next_state, :]) - Q[state, action])
            exp_Q[state, action] = np.exp(Q[state, action] / temperature)
            state = next_state
            episode_reward += reward

        avg_reward += episode_reward
        if episode % interval == 0:
            avg_reward /= interval
            rewards.append(avg_reward)
            episodes.append(episode)
            print(f"{episode}: {avg_reward}")
            avg_reward = 0

    return Q


def use_trained_q_table(env: gym.Env, Q: np.ndarray):
    """
    Uruchamia algorytm ewolucyjny i aktualizuje jego parametry zgodnie z otrzymaną tablicą Q.
    """
    state, _ = env.reset()

    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = np.argmax(Q[state, :])
        next_state, _, terminated, truncated, _ = env.step(action)

        state = next_state
