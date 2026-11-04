import torch

from src.models.credit import QNetwork


def test_q_network_returns_one_value_per_action():
    network = QNetwork()

    values = network(torch.zeros((4, 3)))

    assert values.shape == (4, 3)
