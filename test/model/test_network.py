import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from keras import Sequential
from keras.layers import Dense

from src.model.network import MuCaroNetwork


# ==========================================
# 1. MOCK NETWORKS FOR TESTING
# ==========================================

class MockRepresentation(nn.Module):
    """Mocks the h_network: obs -> hidden_state (Batch, hidden_size)"""

    def __init__(self, obs_channels: int, hidden_size: int):
        super().__init__()
        self.conv = nn.Conv2d(obs_channels, hidden_size, kernel_size=1)

    def forward(self, obs, training: bool = False):
        features = self.conv(obs)
        return features.mean(dim=(2, 3))  # Shape: (Batch, hidden_size)


class MockPrediction(nn.Module):
    """Mocks the f_network: hidden_state -> policy, value"""

    def __init__(self, hidden_size: int, action_space_size: int):
        super().__init__()
        self.policy_head = nn.Linear(hidden_size, action_space_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state, training: bool = False):
        policy_logits = self.policy_head(hidden_state)
        value = self.value_head(hidden_state)
        return policy_logits, value


class MockDynamics(nn.Module):
    """Mocks the g_network: [hidden_state, action] -> next_hidden, reward"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, inputs, training: bool = False):
        hidden_state, _ = inputs
        next_hidden = self.fc(hidden_state)
        reward = self.reward_head(next_hidden)
        return next_hidden, reward


# ==========================================
# 2. PYTEST UNIT TESTS
# ==========================================

@pytest.fixture
def setup_network():
    """Fixture to initialize MuCaroNetwork with mock modules and consistency networks."""
    obs_shape = (3, 15, 15)  # (Channels, Height, Width) for a 15x15 board
    action_space_size = 15 * 15  # 225 possible moves
    hidden_size = 64

    net = MuCaroNetwork()
    net.h_network = MockRepresentation(obs_shape[0], hidden_size)
    net.f_network = MockPrediction(hidden_size, action_space_size)
    net.g_network = MockDynamics(hidden_size)
    net.build_consistency_networks(hidden_size)

    return net, obs_shape, action_space_size, hidden_size


def test_initial_inference(setup_network):
    net, obs_shape, action_space_size, hidden_size = setup_network
    batch_size = 4
    dummy_obs = torch.rand((batch_size, *obs_shape))

    hidden, policy, value = net.initial_inference(dummy_obs)

    assert hidden.shape == (batch_size, hidden_size)
    assert policy.shape == (batch_size, action_space_size)
    assert value.shape == (batch_size, 1)


def test_recurrent_inference(setup_network):
    net, _, action_space_size, hidden_size = setup_network
    batch_size = 4

    dummy_hidden = torch.rand((batch_size, hidden_size))
    dummy_action = torch.randint(0, action_space_size, (batch_size,))

    next_hidden, reward, policy, value = net.recurrent_inference(dummy_hidden, dummy_action)

    assert next_hidden.shape == (batch_size, hidden_size)
    assert reward.shape == (batch_size, 1)
    assert policy.shape == (batch_size, action_space_size)
    assert value.shape == (batch_size, 1)


def test_build_consistency_networks_architecture():
    """Verifies layer dimensions, bottleneck ratios, and bias configurations."""
    # Case 1: hidden_size <= 64 -> bottleneck should be hidden_size // 2
    net_small = MuCaroNetwork()
    net_small.build_consistency_networks(hidden_size=64)

    assert isinstance(net_small.projector_network, Sequential)
    assert isinstance(net_small.predictor_network, Sequential)

    proj_dense_layers = [l for l in net_small.projector_network.layers if isinstance(l, Dense)]
    pred_dense_layers = [l for l in net_small.predictor_network.layers if isinstance(l, Dense)]

    assert len(proj_dense_layers) == 2
    assert proj_dense_layers[0].units == 64 and proj_dense_layers[0].use_bias is False
    assert proj_dense_layers[1].units == 64 and proj_dense_layers[1].use_bias is False

    assert len(pred_dense_layers) == 2
    assert pred_dense_layers[0].units == 32 and pred_dense_layers[0].use_bias is False
    assert pred_dense_layers[1].units == 64 and pred_dense_layers[1].use_bias is True

    # Case 2: hidden_size > 64 -> bottleneck should be hidden_size // 4
    net_large = MuCaroNetwork()
    net_large.build_consistency_networks(hidden_size=128)
    pred_large_dense = [l for l in net_large.predictor_network.layers if isinstance(l, Dense)]
    assert pred_large_dense[0].units == 32
    assert pred_large_dense[1].units == 128


def test_consistency_networks_forward(setup_network):
    """Verifies projector and predictor output shapes in both train and eval modes."""
    net, _, _, hidden_size = setup_network
    batch_size = 4
    dummy_hidden = torch.rand((batch_size, hidden_size))

    for is_training in (True, False):
        projected = net.projector_network(dummy_hidden, training=is_training)
        predicted = net.predictor_network(projected, training=is_training)

        assert projected.shape == (batch_size, hidden_size)
        assert predicted.shape == (batch_size, hidden_size)


def test_consistency_asymmetric_gradient_flow(setup_network):
    """Verifies SimSiam gradient flow: online branch updates g/projector/predictor, target branch stops grad."""
    net, obs_shape, action_space_size, _ = setup_network
    batch_size = 4
    obs_0 = torch.rand((batch_size, *obs_shape))
    obs_1 = torch.rand((batch_size, *obs_shape))
    action_0 = torch.randint(0, action_space_size, (batch_size,))

    # Clear existing gradients
    for p in net.params():
        p.grad = None

    # Online branch (with gradient)
    h_0, _, _ = net.initial_inference(obs_0, training=True)
    # Detach h_0 here only to isolate g_network gradient from h_network in this unit test
    g_hidden_1, _, _, _ = net.recurrent_inference(h_0.detach(), action_0, training=True)
    z_pred = net.projector_network(g_hidden_1, training=True)
    p_pred = net.predictor_network(z_pred, training=True)

    # Target branch (stop-gradient)
    with torch.no_grad():
        h_target_1 = net.h_network(obs_1, training=False)
        z_target = net.projector_network(h_target_1, training=False)

    assert p_pred.requires_grad is True
    assert z_target.requires_grad is False

    loss = (1.0 - F.cosine_similarity(p_pred, z_target, dim=-1)).mean()
    loss.backward()

    assert any(p.grad is not None for p in net.g_network.parameters()), "g_network should receive gradients."
    assert any(p.grad is not None for p in net.projector_network.parameters()), "projector_network should receive gradients."
    assert any(p.grad is not None for p in net.predictor_network.parameters()), "predictor_network should receive gradients."
    assert all(p.grad is None for p in net.h_network.parameters()), "h_network target branch must not receive gradients."


def test_verify_contract_success(setup_network):
    """Tests if the built-in verify_contract passes successfully with correct shapes."""
    net, obs_shape, action_space_size, _ = setup_network

    try:
        net.verify_contract(obs_shape, action_space_size)
    except AssertionError as e:
        pytest.fail(f"verify_contract failed unexpectedly: {e}")


def test_verify_contract_failure(setup_network):
    """Overrides the f_network output shape to ensure the assert statements catch the error."""
    net, obs_shape, action_space_size, _ = setup_network

    original_forward = net.f_network.forward
    net.f_network.forward = lambda h, training=False: (original_forward(h, training)[0], torch.zeros(h.shape[0], 2))

    with pytest.raises(AssertionError, match="Value shape expected"):
        net.verify_contract(obs_shape, action_space_size)


def test_verify_contract_consistency_mismatch(setup_network):
    """Ensures verify_contract catches dimension mismatches between hidden_state and consistency networks."""
    net, obs_shape, action_space_size, hidden_size = setup_network
    net.build_consistency_networks(hidden_size + 16)

    with pytest.raises((AssertionError, ValueError)):
        net.verify_contract(obs_shape, action_space_size)


def test_params_method(setup_network):
    """Tests if the params method correctly aggregates parameters from all 5 sub-networks."""
    net, _, _, _ = setup_network

    params_list = net.params()

    expected_params = (
        list(net.h_network.parameters())
        + list(net.f_network.parameters())
        + list(net.g_network.parameters())
        + list(net.projector_network.parameters())
        + list(net.predictor_network.parameters())
    )

    assert isinstance(params_list, list), "The params() method should return a list."
    assert len(params_list) > 0, "The parameter list should not be empty."
    assert len(params_list) == len(expected_params), "Mismatch in total number of parameters."

    for actual_param, expected_param in zip(params_list, expected_params):
        assert actual_param is expected_param, "A parameter in the returned list does not match the expected parameter."