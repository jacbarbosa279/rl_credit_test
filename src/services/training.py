from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.credit import CreditLimitEnv, QNetwork
from src.utils.plotting import plot_results


def train_agent(episodes=300):
    env = CreditLimitEnv()
    q_net = QNetwork()
    optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    memory = deque(maxlen=2000)

    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.985
    batch_size = 32

    rewards_history = []
    defaults_history = []

    print("Starting RL Agent training...\n")

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        defaulted = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    state_t = torch.tensor(
                        state, dtype=torch.float32
                    ).unsqueeze(0)
                    action = torch.argmax(q_net(state_t)).item()

            next_state, reward, done, is_default = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if is_default:
                defaulted = True

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_t = torch.tensor(
                    np.array(states), dtype=torch.float32
                )
                actions_t = torch.tensor(
                    actions, dtype=torch.int64
                ).unsqueeze(1)
                rewards_t = torch.tensor(
                    rewards, dtype=torch.float32
                ).unsqueeze(1)
                next_states_t = torch.tensor(
                    np.array(next_states), dtype=torch.float32
                )
                dones_t = torch.tensor(
                    dones, dtype=torch.float32
                ).unsqueeze(1)

                q_values = q_net(states_t).gather(1, actions_t)
                with torch.no_grad():
                    max_next_q = q_net(next_states_t).max(1)[0].unsqueeze(1)
                    target_q = (
                        rewards_t + (1 - dones_t) * gamma * max_next_q
                    )

                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(0.05, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        defaults_history.append(1 if defaulted else 0)

        if (ep + 1) % 50 == 0 or ep == 0:
            print(
                f"Episode {ep+1:03d}/{episodes} | Total Reward: "
                f"{total_reward:6.2f} | Default: {defaulted} | "
                f"Epsilon: {epsilon:.2f}"
            )

    print("\n[OK] Training completed successfully! Plotting results...")

    plot_results(rewards_history, defaults_history)
    return q_net
