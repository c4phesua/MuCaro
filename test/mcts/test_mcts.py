import os

os.environ['KERAS_BACKEND'] = 'torch'

import math

import pytest
import torch

from src.mcts.minmax import MinMaxStats
# --- Adjust these imports to match your project structure ---
from src.mcts.node import Node
from src.mcts.tree import MCTS


# ==========================================
# 1. MOCK NETWORK
# ==========================================
class MockMuCaroNetwork:
    """Mock for the combined MuCaroNetwork wrapper."""

    def __init__(self, action_space_size):
        self.action_space_size = action_space_size

    def initial_inference(self, current_state, training=False):
        """Mocks initial_inference: Returns (hidden_state, policy_logits, value)"""
        batch_size = current_state.shape[0]
        hidden_state = torch.zeros((batch_size, 64))
        policy_logits = torch.zeros((batch_size, self.action_space_size))
        value = torch.tensor([[0.5]] * batch_size)

        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state, action, training=False):
        """Mocks recurrent_inference: Returns (next_hidden, reward, policy_logits, value)"""
        batch_size = hidden_state.shape[0]
        next_hidden = torch.ones((batch_size, 64))
        reward = torch.tensor([[1.0]] * batch_size)
        policy_logits = torch.zeros((batch_size, self.action_space_size))
        value = torch.tensor([[0.5]] * batch_size)

        return next_hidden, reward, policy_logits, value


# ==========================================
# 2. MCTS FIXTURE (Runs before each test)
# ==========================================
@pytest.fixture
def mcts_env():
    """Initialize a basic test environment with action_space_size = 9"""
    action_space_size = 9
    mock_net = MockMuCaroNetwork(action_space_size)

    mcts = MCTS(
        network=mock_net,
        action_space_size=action_space_size
    )
    return mcts, action_space_size


# ==========================================
# 3. TEST CASES
# ==========================================

def test_initialization(mcts_env):
    """Test MCTS initialization parameters."""
    mcts, action_space_size = mcts_env
    assert mcts.action_space_size == action_space_size
    assert math.isclose(mcts.root_dirichlet_alpha, 0.03, abs_tol=1e-3)
    assert mcts.gamma == 1.0


def test_cal_distribution(mcts_env):
    """Test the visit count distribution calculation."""
    mcts, action_space_size = mcts_env
    root = Node(action=-1, action_space_size=action_space_size)

    # Mock the visit counts of the child nodes
    # Total visit_count = 10 + 20 + 70 = 100
    root.children[0] = Node(action=0, action_space_size=action_space_size)
    root.children[0].visit_count = 10

    root.children[3] = Node(action=3, action_space_size=action_space_size)
    root.children[3].visit_count = 20

    root.children[8] = Node(action=8, action_space_size=action_space_size)
    root.children[8].visit_count = 70

    dist = mcts.cal_distribution(root, num_simulations=100)

    assert len(dist) == action_space_size
    assert math.isclose(sum(dist), 1.0, abs_tol=1e-6)  # The distribution must sum to 1.0
    assert dist[0] == 0.1
    assert dist[3] == 0.2
    assert dist[8] == 0.7
    assert dist[1] == 0.0  # Unvisited node


def test_add_exploration_noise(mcts_env):
    """Test if Dirichlet noise is added correctly to the root node's priors."""
    mcts, action_space_size = mcts_env
    root = Node(action=-1, action_space_size=action_space_size)

    exact_prior = 1.0 / action_space_size
    for i in range(action_space_size):
        root.children[i] = Node(action=i, action_space_size=action_space_size, prior=exact_prior)

    old_priors = [child.prior for child in root.children]
    mcts.add_exploration_noise(root)
    new_priors = [child.prior for child in root.children]

    assert old_priors != new_priors
    assert math.isclose(sum(new_priors), 1.0, abs_tol=1e-5)


def test_select_child_and_ucb(mcts_env):
    """Test the UCB logic and child selection."""
    mcts, action_space_size = mcts_env
    root = Node(action=-1, action_space_size=action_space_size)
    root.visit_count = 10
    minmax_norm = MinMaxStats(-1, 1)

    # Create 2 mock child nodes
    child1 = Node(action=0, action_space_size=action_space_size, prior=0.5)
    child1.visit_count = 5
    child1.reward = 1.0
    child1.value_sum = 2.5  # node.value() will be 2.5 / 5 = 0.5

    child2 = Node(action=1, action_space_size=action_space_size, prior=0.5)
    child2.visit_count = 1
    child2.reward = 0.0
    child2.value_sum = -0.5  # node.value() will be -0.5 / 1 = -0.5

    root.children[0] = child1
    root.children[1] = child2

    selected_child = mcts.select_child(root, minmax_norm)

    assert selected_child is not None
    assert selected_child.action in [0, 1]


def test_full_simulate_workflow(mcts_env):
    """Test the full Simulation workflow (Integration)."""
    mcts, action_space_size = mcts_env

    # Dummy input board (Caro 15x15 format).
    # Shape is (C, H, W) because your MCTS applies unsqueeze(0) internally
    dummy_state = torch.zeros((3, 15, 15))

    # Run MCTS with 50 simulations
    pi_distribution = mcts.simulate(
        current_state=dummy_state,
        num_simulations=50,
        is_training=True
    )

    # 1. The output distribution length must match action_space_size
    assert len(pi_distribution) == action_space_size

    # 2. The distribution must sum to 1.0
    assert math.isclose(sum(pi_distribution), 1.0, abs_tol=1e-5)

    # 3. Ensure all probabilities are >= 0
    assert all(prob >= 0.0 for prob in pi_distribution)


def test_simulate_validation(mcts_env):
    """Test exception raising in the simulate function."""
    mcts, _ = mcts_env

    # Shape is (C, H, W)
    dummy_state = torch.zeros((3, 15, 15))

    # Must raise ValueError if num_simulations <= 0
    with pytest.raises(ValueError, match="Number of simulation must greater than 0"):
        mcts.simulate(dummy_state, num_simulations=0)
