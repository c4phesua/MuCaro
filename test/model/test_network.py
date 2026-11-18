import pytest
import torch
import torch.nn as nn

from src.model.network import MuCaroNetwork


# ==========================================
# 1. MOCK NETWORKS FOR TESTING
# ==========================================

class MockRepresentation(nn.Module):
    """Mocks the h_network: obs -> hidden_state"""

    def __init__(self, obs_channels, hidden_channels):
        super().__init__()
        self.conv = nn.Conv2d(obs_channels, hidden_channels, kernel_size=1)

    def forward(self, obs, training=False):
        return self.conv(obs)


class MockPrediction(nn.Module):
    """Mocks the f_network: hidden_state -> policy, value"""

    def __init__(self, hidden_channels, action_space_size):
        super().__init__()
        self.policy_head = nn.Linear(hidden_channels, action_space_size)
        self.value_head = nn.Linear(hidden_channels, 1)

    def forward(self, hidden_state, training=False):
        # Flatten spatial dimensions (H, W) using global average pooling
        pooled = hidden_state.mean(dim=(2, 3))  # Shape: (Batch, hidden_channels)
        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        return policy_logits, value


class MockDynamics(nn.Module):
    """Mocks the g_network: [hidden_state, action] -> next_hidden, reward"""

    def __init__(self, hidden_channels):
        super().__init__()
        # Simple mock: ignore action, just pass hidden_state through a conv layer
        self.conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.reward_head = nn.Linear(hidden_channels, 1)

    def forward(self, inputs, training=False):
        hidden_state, action = inputs
        next_hidden = self.conv(hidden_state)
        pooled = next_hidden.mean(dim=(2, 3))
        reward = self.reward_head(pooled)
        return next_hidden, reward


# ==========================================
# 2. PYTEST UNIT TESTS
# ==========================================

@pytest.fixture
def setup_network():
    """Fixture to initialize MuCaroNetwork with mock modules"""
    obs_shape = (3, 15, 15)  # (Channels, Height, Width) for a 15x15 board
    action_space_size = 15 * 15  # 225 possible moves
    hidden_channels = 64

    net = MuCaroNetwork()
    net.h_network = MockRepresentation(obs_shape[0], hidden_channels)
    net.f_network = MockPrediction(hidden_channels, action_space_size)
    net.g_network = MockDynamics(hidden_channels)

    return net, obs_shape, action_space_size, hidden_channels


def test_initial_inference(setup_network):
    net, obs_shape, action_space_size, hidden_channels = setup_network
    batch_size = 4
    dummy_obs = torch.rand((batch_size, *obs_shape))

    hidden, policy, value = net.initial_inference(dummy_obs)

    assert hidden.shape == (batch_size, hidden_channels, obs_shape[1], obs_shape[2])
    assert policy.shape == (batch_size, action_space_size)
    assert value.shape == (batch_size, 1)


def test_recurrent_inference(setup_network):
    net, obs_shape, action_space_size, hidden_channels = setup_network
    batch_size = 4

    # Create dummy hidden state and action
    dummy_hidden = torch.rand((batch_size, hidden_channels, obs_shape[1], obs_shape[2]))
    dummy_action = torch.randint(0, action_space_size, (batch_size,))

    next_hidden, reward, policy, value = net.recurrent_inference(dummy_hidden, dummy_action)

    assert next_hidden.shape == dummy_hidden.shape
    assert reward.shape == (batch_size, 1)
    assert policy.shape == (batch_size, action_space_size)
    assert value.shape == (batch_size, 1)


def test_verify_contract_success(setup_network):
    """Tests if the built-in verify_contract passes successfully with correct shapes"""
    net, obs_shape, action_space_size, _ = setup_network

    try:
        net.verify_contract(obs_shape, action_space_size)
    except AssertionError as e:
        pytest.fail(f"verify_contract failed unexpectedly: {e}")


def test_verify_contract_failure(setup_network):
    """Overrides the f_network output shape to ensure the assert statements catch the error"""
    net, obs_shape, action_space_size, _ = setup_network

    # Override f_network's forward to return a wrong value shape: (B, 2) instead of (B, 1)
    original_forward = net.f_network.forward
    net.f_network.forward = lambda h, training=False: (original_forward(h, training)[0], torch.zeros(h.shape[0], 2))

    with pytest.raises(AssertionError, match="Value shape expected"):
        net.verify_contract(obs_shape, action_space_size)
