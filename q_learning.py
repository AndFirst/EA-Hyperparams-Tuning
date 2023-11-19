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
                # action = env.action_space[np.argmax(Q[state, :])]
                action = np.argmax(Q[state, :])

            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Update Q-table using Q-learning rule
            Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
            state = next_state
            episode_reward += reward

        avg_reward += episode_reward
        if episode % interval == 0:
            avg_reward /= interval
            rewards.append(avg_reward)
            episodes.append(episode)
            print(f"{episode}: {avg_reward}")
            avg_reward = 0

    plt.plot(episodes, rewards, label=f"greedy [e {epsilon} lr {learning_rate} df {discount_factor}")
    plt.xlabel('Episode')
    plt.ylabel('Avarage reward')
    plt.legend()

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
            Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
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

    plt.plot(episodes, rewards, label=f"boltzmann [t {temperature} lr {learning_rate} df {discount_factor}")
    plt.xlabel('Episode')
    plt.ylabel('Avarage reward')
    plt.legend()

    return Q

def show_model(env: gym.Env, Q: np.ndarray):
    state, _ = env.reset()
    print(env.render())

    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = np.argmax(Q[state, :])

        next_state, _, terminated, truncated, _ = env.step(action)
        print(env.render())

        state = next_state


if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode = 'ansi')

    q_learning_greedy(env, 0.8, 0.9, 500, 0.1, 10)
    q_learning_boltzmann(env, 0.8, 0.9, 500, 0.9, 10)
    plt.title("Greedy Vs Boltzmann")
    plt.show()