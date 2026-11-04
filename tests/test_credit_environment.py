import numpy as np

from src.models.credit import CreditLimitEnv


def test_reset_returns_normalized_state():
    env = CreditLimitEnv()

    state = env.reset()

    assert state.shape == (3,)
    assert state.dtype == np.float32
    assert 0.1 <= state[0] <= 0.7
    assert 0.0 <= state[1] <= 0.2
    assert 0.0 <= state[2] <= 1.0
    assert env.current_step == 0


def test_step_reduces_utilization_for_action_two(monkeypatch):
    env = CreditLimitEnv()
    env.state = np.array([0.5, 0.0, 850.0], dtype=np.float32)
    monkeypatch.setattr(np.random, "rand", lambda: 1.0)

    next_state, reward, done, is_default = env.step(2)

    assert next_state[0] == np.float32(0.425)
    assert reward == np.float32(0.425) * 15.0
    assert not done
    assert not is_default
    assert env.current_step == 1
