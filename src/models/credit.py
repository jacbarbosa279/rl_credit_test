import numpy as np
import torch.nn as nn


class CreditLimitEnv:
    """Simulates retail credit limit management over a 12-month horizon."""

    def __init__(self):
        self.state = None
        self.max_steps = 12
        self.current_step = 0

    def reset(self):
        utilization = np.random.uniform(0.1, 0.7)
        delinquency = np.random.randint(0, 2)
        score = np.random.uniform(500, 850)
        self.state = np.array(
            [utilization, delinquency, score], dtype=np.float32
        )
        self.current_step = 0
        return self._get_normalized_state()

    def _get_normalized_state(self):
        utilization, delinquency, score = self.state
        return np.array(
            [utilization, delinquency / 5.0, (score - 300.0) / 550.0],
            dtype=np.float32,
        )

    def step(self, action):
        self.current_step += 1
        utilization, delinquency, score = self.state

        if action == 2:
            utilization *= 0.85
        elif action == 0:
            utilization = min(1.0, utilization * 1.15)

        default_probability = (utilization * 0.4) + (
            (850 - score) / 850 * 0.6
        )
        is_default = np.random.rand() < default_probability

        if is_default:
            reward = -100.0
            done = True
        else:
            reward = (utilization * 15.0) - (delinquency * 3.0)
            score = max(300, min(850, score + (5 if action != 2 else -2)))
            delinquency = max(
                0, delinquency + (1 if utilization > 0.85 else -1)
            )
            done = self.current_step >= self.max_steps

        self.state = np.array(
            [utilization, delinquency, score], dtype=np.float32
        )
        return self._get_normalized_state(), reward, done, is_default


class QNetwork(nn.Module):
    def __init__(self, state_dim=3, action_dim=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)
