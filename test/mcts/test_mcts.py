import os

os.environ["KERAS_BACKEND"] = "torch"

import math

import numpy as np
import pytest
import torch

from src.mcts.minmax import MinMaxStats
from src.mcts.node import Node
from src.mcts.tree import MCTS


# ==========================================
# 1. MOCK NETWORK
# ==========================================
class MockMuCaroNetwork:
    """Mock wrapper for MuCaroNetwork used to isolate MCTS logic during testing."""

    def __init__(self, action_space_size: int, custom_logits: torch.Tensor = None):
        self.action_space_size = action_space_size
        self.custom_logits = custom_logits

    def initial_inference(self, current_state, training=False):
        """Mock initial inference returning (hidden_state, policy_logits, value)."""
        batch_size = current_state.shape[0]
        hidden_state = torch.zeros((batch_size, 64))
        if self.custom_logits is not None:
            policy_logits = self.custom_logits.expand(batch_size, -1).clone()
        else:
            policy_logits = torch.zeros((batch_size, self.action_space_size))
        value = torch.tensor([[0.5]] * batch_size)

        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state, action, training=False):
        """Mock recurrent inference returning (next_hidden, reward, policy_logits, value)."""
        batch_size = hidden_state.shape[0]
        next_hidden = torch.ones((batch_size, 64))
        reward = torch.tensor([[1.0]] * batch_size)
        policy_logits = torch.zeros((batch_size, self.action_space_size))
        value = torch.tensor([[0.5]] * batch_size)

        return next_hidden, reward, policy_logits, value


# ==========================================
# 2. MCTS FIXTURE
# ==========================================
@pytest.fixture
def mcts_env():
    """Initialize a default MCTS test environment with action_space_size = 9."""
    action_space_size = 9
    mock_net = MockMuCaroNetwork(action_space_size)

    mcts = MCTS(
        network=mock_net,
        action_space_size=action_space_size,
    )
    return mcts, action_space_size


# ==========================================
# 3. TEST CASES
# ==========================================
def test_initialization(mcts_env):
    """Verify MCTS default hyperparameters and action space size."""
    mcts, action_space_size = mcts_env
    assert mcts.action_space_size == action_space_size
    assert math.isclose(mcts.root_dirichlet_alpha, 0.03, abs_tol=1e-3)
    assert mcts.gamma == 1.0


def test_cal_distribution(mcts_env):
    """Verify visit count normalization across root children."""
    mcts, action_space_size = mcts_env
    root = Node(action=-1, action_space_size=action_space_size)

    # Populate visit counts (total = 10 + 20 + 70 = 100)
    root.children[0] = Node(action=0, action_space_size=action_space_size)
    root.children[0].visit_count = 10

    root.children[3] = Node(action=3, action_space_size=action_space_size)
    root.children[3].visit_count = 20

    root.children[8] = Node(action=8, action_space_size=action_space_size)
    root.children[8].visit_count = 70

    dist = mcts.cal_distribution(root, num_simulations=100)

    assert len(dist) == action_space_size
    assert math.isclose(sum(dist), 1.0, abs_tol=1e-6)
    assert dist[0] == 0.1
    assert dist[3] == 0.2
    assert dist[8] == 0.7
    assert dist[1] == 0.0  # Unexpanded/unvisited node must have 0.0 probability


def test_add_exploration_noise_with_masked_children(mcts_env):
    """Ensure Dirichlet noise is applied strictly to valid (non-None) children."""
    mcts, action_space_size = mcts_env
    root = Node(action=-1, action_space_size=action_space_size)

    valid_indices = [0, 2, 5, 8]
    exact_prior = 1.0 / len(valid_indices)
    for i in valid_indices:
        root.children[i] = Node(action=i, action_space_size=action_space_size, prior=exact_prior)

    old_priors = [root.children[i].prior for i in valid_indices]
    mcts.add_exploration_noise(root)
    new_priors = [root.children[i].prior for i in valid_indices]

    # Invalid children must remain None
    for i in range(action_space_size):
        if i not in valid_indices:
            assert root.children[i] is None

    # Valid priors must be perturbed and still sum to 1.0
    assert old_priors != new_priors
    assert math.isclose(sum(new_priors), 1.0, abs_tol=1e-5)


def test_select_child_and_ucb(mcts_env):
    """Verify UCB scoring assigns -inf to None children and selects valid ones."""
    mcts, action_space_size = mcts_env
    root = Node(action=-1, action_space_size=action_space_size)
    root.visit_count = 10
    minmax_norm = MinMaxStats(-1, 1)

    # Create two valid child nodes
    child1 = Node(action=2, action_space_size=action_space_size, prior=0.5)
    child1.visit_count = 5
    child1.reward = 1.0
    child1.value_sum = 2.5

    child2 = Node(action=6, action_space_size=action_space_size, prior=0.5)
    child2.visit_count = 1
    child2.reward = 0.0
    child2.value_sum = -0.5

    root.children[2] = child1
    root.children[6] = child2

    # Unexpanded actions must score -inf
    scores = mcts.ucb_scores(root, minmax_norm)
    assert scores[0] == float("-inf")
    assert scores[1] == float("-inf")

    selected_child = mcts.select_child(root, minmax_norm)
    assert selected_child is not None
    assert selected_child.action in [2, 6]


def test_full_simulate_workflow_without_mask(mcts_env):
    """Verify full MCTS simulation when valid_mask is omitted (None)."""
    mcts, action_space_size = mcts_env
    dummy_state = torch.zeros((2, 3, 3))

    pi_distribution = mcts.simulate(
        current_state=dummy_state,
        valid_mask=None,
        num_simulations=50,
        is_training=True,
    )

    assert len(pi_distribution) == action_space_size
    assert math.isclose(sum(pi_distribution), 1.0, abs_tol=1e-5)
    assert all(prob >= 0.0 for prob in pi_distribution)


@pytest.mark.parametrize("is_training", [True, False])
def test_simulate_with_valid_mask_blocks_invalid_actions(mcts_env, is_training):
    """
    Verify masked actions receive 0.0 probability in both training
    (with Dirichlet noise) and evaluation modes.
    """
    mcts, action_space_size = mcts_env
    dummy_state = torch.zeros((2, 3, 3))

    # Only actions 0, 4, and 8 are legal
    valid_mask = [True, False, False, False, True, False, False, False, True]

    pi_distribution = mcts.simulate(
        current_state=dummy_state,
        valid_mask=valid_mask,
        num_simulations=60,
        is_training=is_training,
    )

    assert len(pi_distribution) == action_space_size
    assert math.isclose(sum(pi_distribution), 1.0, abs_tol=1e-5)

    for action_idx, is_valid in enumerate(valid_mask):
        if not is_valid:
            assert pi_distribution[action_idx] == 0.0, (
                f"Invalid action {action_idx} received non-zero visits: {pi_distribution[action_idx]}"
            )


def test_simulate_mask_overrides_overwhelming_prior_logits():
    """
    Ensure valid_mask suppresses occupied cells even when the network outputs
    extreme logits (+100.0) for those invalid actions.
    """
    action_space_size = 9
    # Network strongly favors occupied actions 1 and 7
    biased_logits = torch.tensor([[0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0]])
    mock_net = MockMuCaroNetwork(action_space_size, custom_logits=biased_logits)
    mcts = MCTS(network=mock_net, action_space_size=action_space_size)

    valid_mask = [True, False, True, True, True, True, True, False, True]
    dummy_state = torch.zeros((2, 3, 3))

    pi_distribution = mcts.simulate(
        current_state=dummy_state,
        valid_mask=valid_mask,
        num_simulations=50,
        is_training=False,
    )

    assert pi_distribution[1] == 0.0
    assert pi_distribution[7] == 0.0
    assert math.isclose(sum(pi_distribution), 1.0, abs_tol=1e-5)


def test_simulate_single_valid_action_forced_move(mcts_env):
    """Ensure 100% of simulations go to the sole remaining legal action without NaNs."""
    mcts, action_space_size = mcts_env
    dummy_state = torch.zeros((2, 3, 3))

    valid_mask = [False] * action_space_size
    forced_action = 5
    valid_mask[forced_action] = True

    pi_distribution = mcts.simulate(
        current_state=dummy_state,
        valid_mask=valid_mask,
        num_simulations=30,
        is_training=True,
    )

    assert not any(math.isnan(p) for p in pi_distribution)
    assert pi_distribution[forced_action] == 1.0
    assert sum(pi_distribution) == 1.0


def test_simulate_accepts_int_and_numpy_masks(mcts_env):
    """Verify valid_mask handles 0/1 integer lists and NumPy boolean arrays."""
    mcts, _ = mcts_env
    dummy_state = torch.zeros((2, 3, 3))

    # 1. Integer list mask (0/1)
    int_mask = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    pi_int = mcts.simulate(dummy_state, valid_mask=int_mask, num_simulations=20, is_training=False)
    assert math.isclose(sum(pi_int), 1.0, abs_tol=1e-5)
    assert all(pi_int[i] == 0.0 for i, m in enumerate(int_mask) if m == 0)

    # 2. NumPy boolean array mask
    np_mask = np.array([False, True, False, False, True, False, False, True, False], dtype=bool)
    pi_np = mcts.simulate(dummy_state, valid_mask=np_mask, num_simulations=20, is_training=False)
    assert math.isclose(sum(pi_np), 1.0, abs_tol=1e-5)
    assert all(pi_np[i] == 0.0 for i, m in enumerate(np_mask) if not m)


def test_simulate_validation(mcts_env):
    """Verify ValueError is raised when num_simulations <= 0."""
    mcts, _ = mcts_env
    dummy_state = torch.zeros((2, 3, 3))

    with pytest.raises(ValueError, match="Number of simulation must greater than 0"):
        mcts.simulate(dummy_state, num_simulations=0)


@pytest.mark.parametrize(
    "valid_mask",
    [
        # Early game: 2 occupied cells, 7 empty cells
        [True, True, False, True, True, True, False, True, True],
        # Mid game: 5 occupied cells, 4 empty cells
        [False, True, False, True, False, False, True, False, True],
        # Endgame: 7 occupied cells, 2 empty cells
        [False, False, True, False, False, False, False, True, False],
    ],
)
def test_policy_strictly_zero_on_false_mask_across_game_phases(valid_mask):
    """
    Verify that across early, mid, and endgame board states with random network logits,
    every False position in valid_mask yields an exact 0.0 probability in the returned policy.
    """
    torch.manual_seed(42)
    action_space_size = 9
    random_logits = torch.randn((1, action_space_size))
    mock_net = MockMuCaroNetwork(action_space_size, custom_logits=random_logits)
    mcts = MCTS(network=mock_net, action_space_size=action_space_size)

    dummy_state = torch.zeros((2, 3, 3))
    policy = mcts.simulate(
        current_state=dummy_state,
        valid_mask=valid_mask,
        num_simulations=80,
        is_training=True,
    )

    assert len(policy) == action_space_size
    assert math.isclose(sum(policy), 1.0, abs_tol=1e-5)

    for idx, is_legal in enumerate(valid_mask):
        if not is_legal:
            assert policy[idx] == 0.0, f"Expected 0.0 at masked index {idx}, got {policy[idx]}"
        else:
            assert policy[idx] >= 0.0


def test_policy_zero_on_false_mask_under_full_exploration_noise(mcts_env):
    """
    Ensure Dirichlet exploration noise never leaks into masked positions,
    even when root_exploration_fraction is set to 1.0 (100% noise).
    """
    mcts, action_space_size = mcts_env
    mcts.root_exploration_fraction = 1.0
    mcts.root_dirichlet_alpha = 1.0

    valid_mask = [False, True, False, True, False, True, False, True, False]
    dummy_state = torch.zeros((2, 3, 3))

    policy = mcts.simulate(
        current_state=dummy_state,
        valid_mask=valid_mask,
        num_simulations=60,
        is_training=True,
    )

    assert math.isclose(sum(policy), 1.0, abs_tol=1e-5)
    for idx, is_legal in enumerate(valid_mask):
        if not is_legal:
            assert policy[idx] == 0.0


def test_policy_zero_on_false_mask_when_valid_logits_are_strongly_negative():
    """
    Verify masking works cleanly without NaN or probability leakage even when
    legal actions have strongly negative logits (-50.0) and illegal actions have +10.0.
    """
    action_space_size = 9
    valid_mask = [True, False, False, False, True, False, False, False, False]

    # Legal actions (0, 4) have -50.0 logits; all illegal actions have +10.0 logits
    logits = torch.full((1, action_space_size), 10.0)
    logits[0, 0] = -50.0
    logits[0, 4] = -50.0

    mock_net = MockMuCaroNetwork(action_space_size, custom_logits=logits)
    mcts = MCTS(network=mock_net, action_space_size=action_space_size)
    dummy_state = torch.zeros((2, 3, 3))

    policy = mcts.simulate(
        current_state=dummy_state,
        valid_mask=valid_mask,
        num_simulations=40,
        is_training=False,
    )

    assert not any(math.isnan(p) for p in policy)
    assert math.isclose(policy[0] + policy[4], 1.0, abs_tol=1e-5)
    for idx, is_legal in enumerate(valid_mask):
        if not is_legal:
            assert policy[idx] == 0.0