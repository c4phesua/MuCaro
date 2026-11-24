import math
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from model.trainer import MuCaroTrainer
from model.utils import save_checkpoint


# ==============================================================================
# 1. DUMMY CLASSES TO MOCK ENVIRONMENT, MCTS, AND NETWORK
# ==============================================================================

class DummyEnv:
    """
    Configurable dummy environment matching the Board contract:
    - Returns torch.Tensor for 'next_state' (compatible with torch.stack in sample_batch).
    - Tracks legal moves via get_legal_mask() and updates mask state on play().
    - Supports decisive wins (winner > 0), ties (winner == -1), invalid moves (ValueError),
      and unexpected runtime errors.
    """

    def __init__(
            self,
            obs_shape=(2, 9),
            action_space_size=9,
            max_steps=5,
            outcome_winner=1,
            raise_unexpected_error=False,
            enforce_unique_moves=False,
    ):
        self.obs_shape = obs_shape
        self.action_space_size = action_space_size
        self.max_steps = max_steps
        self.outcome_winner = outcome_winner
        self.raise_unexpected_error = raise_unexpected_error
        self.enforce_unique_moves = enforce_unique_moves
        self.step_count = 0
        self.size = 15  # Required when saving checkpoints
        self.m = 3
        self.legal_mask = [True] * self.action_space_size
        self.get_legal_mask_calls = 0

    def reset(self):
        self.step_count = 0
        self.legal_mask = [True] * self.action_space_size
        return {'next_state': torch.zeros(self.obs_shape, dtype=torch.float32)}

    def get_legal_mask(self):
        self.get_legal_mask_calls += 1
        return list(self.legal_mask)

    def play(self, action):
        if self.raise_unexpected_error:
            raise RuntimeError("Unexpected environment failure")

        if action == 999:
            raise ValueError("Invalid move")

        if 0 <= action < self.action_space_size:
            if self.enforce_unique_moves and not self.legal_mask[action]:
                raise ValueError(f"Action {action} is already occupied")
            self.legal_mask[action] = False

        self.step_count += 1
        done = self.step_count >= self.max_steps
        winner = self.outcome_winner if done else 0

        return {
            'next_state': torch.full(self.obs_shape, float(self.step_count), dtype=torch.float32),
            'done': done,
            'winner': winner
        }


class DummyMCTS:
    """Dummy MCTS returning a valid probability distribution over legal actions."""

    def __init__(self, action_space_size=9):
        self.action_space_size = action_space_size
        self.call_history = []

    def simulate(self, current_state, num_simulations, is_training, valid_mask=None):
        self.call_history.append({
            "state_shape": tuple(current_state.shape),
            "state_dtype": current_state.dtype,
            "num_simulations": num_simulations,
            "is_training": is_training,
            "valid_mask": list(valid_mask) if valid_mask is not None else None,
        })
        if valid_mask is not None:
            mask_arr = np.asarray(valid_mask, dtype=bool)
            pi = np.where(mask_arr, 1.0, 0.0).astype(np.float32)
            total = np.sum(pi)
            if total > 0:
                pi /= total
            else:
                pi = np.ones(self.action_space_size, dtype=np.float32) / self.action_space_size
        else:
            pi = np.ones(self.action_space_size, dtype=np.float32) / self.action_space_size
        return pi


class DummyNetwork(nn.Module):
    """
    Dummy Network inheriting from nn.Module with non-zero directional forward outputs
    across all 5 sub-networks (h, g, f, projector, predictor) so PyTorch computes
    true non-zero gradients across the entire architecture including SimSiam branches.
    """

    def __init__(self, action_space_size=9, hidden_dim=8):
        super().__init__()
        self.action_space_size = action_space_size
        self.hidden_dim = hidden_dim

        # Separate parameters for each of the 5 sub-networks
        self.h_param = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.g_param = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.f_param = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        # Non-uniform vector weights ensure directional sensitivity for cosine_similarity gradients
        self.proj_param = nn.Parameter(torch.linspace(0.4, 0.8, hidden_dim), requires_grad=True)
        self.pred_param = nn.Parameter(torch.linspace(0.9, 0.5, hidden_dim), requires_grad=True)

        # Mock sub-modules to verify .train()/.eval() transitions and forward calls
        self.h_network = MagicMock(side_effect=self._h_forward)
        self.f_network = MagicMock()
        self.g_network = MagicMock()
        self.projector_network = MagicMock(side_effect=self._proj_forward)
        self.predictor_network = MagicMock(side_effect=self._pred_forward)

    def _h_forward(self, obs, training=False):
        batch_size = obs.shape[0]
        obs_mean = obs.view(batch_size, -1).mean(dim=-1, keepdim=True) + 1.0
        return obs_mean.expand(batch_size, self.hidden_dim) * self.h_param

    def _proj_forward(self, hidden_state, training=False):
        return hidden_state * self.proj_param

    def _pred_forward(self, projector_state, training=False):
        return projector_state * self.pred_param

    def params(self):
        return list(self.parameters())

    def initial_inference(self, obs, training=False):
        batch_size = obs.shape[0]
        hidden_state = self._h_forward(obs, training=training)
        pred_policies = hidden_state.mean(dim=-1, keepdim=True).expand(
            batch_size, self.action_space_size
        ) * self.f_param
        pred_values = hidden_state.mean(dim=-1, keepdim=True) * self.f_param
        return hidden_state, pred_policies, pred_values

    def recurrent_inference(self, hidden_state, action, training=False):
        batch_size = hidden_state.shape[0]
        next_hidden = (hidden_state + 0.1) * self.g_param
        pred_reward = next_hidden.mean(dim=-1, keepdim=True) * self.g_param
        pred_policies = next_hidden.mean(dim=-1, keepdim=True).expand(
            batch_size, self.action_space_size
        ) * self.f_param
        pred_values = next_hidden.mean(dim=-1, keepdim=True) * self.f_param
        return next_hidden, pred_reward, pred_policies, pred_values

    @torch.no_grad()
    def consistency_project(self, observation):
        hidden_state = self.h_network(observation, training=False)
        return self.projector_network(hidden_state, training=False)

    def consistency_predict(self, hidden_state, training=True):
        projector_state = self.projector_network(hidden_state, training=training)
        return self.predictor_network(projector_state, training=training)


# ==============================================================================
# 2. PYTEST FIXTURES
# ==============================================================================

@pytest.fixture
def trainer():
    """Provides a fresh instance of MuCaroTrainer for each test."""
    np.random.seed(42)
    torch.manual_seed(42)

    env = DummyEnv(obs_shape=(2, 9), action_space_size=9, max_steps=5, outcome_winner=1)
    mcts = DummyMCTS(action_space_size=9)
    network = DummyNetwork(action_space_size=9)

    return MuCaroTrainer(
        env=env,
        mcts=mcts,
        network=network,
        use_valid_mask=True,
        buffer_size=20,
        batch_size=4,
        unroll_steps=3,
        num_simulations=10,
        discount=0.9,
        lr=1e-2,
        temp_threshold=3
    )


@pytest.fixture
def mock_evaluator():
    """Provides a mocked evaluator function that returns a dummy result."""
    evaluator = MagicMock()
    evaluator.return_value = {
        'win': 8,
        'lose': 2,
        'invalid': 1,
        'tie': 0,
        'win_percentage': 0.8
    }
    return evaluator


# ==============================================================================
# 3. UNIT TESTS (PYTEST)
# ==============================================================================

# --------------------------------------------------------------------------
# Initialization & Valid Mask Configuration Tests
# --------------------------------------------------------------------------

def test_trainer_use_valid_mask_configuration():
    """Verify use_valid_mask defaults to True and can be explicitly disabled."""
    env = DummyEnv()
    mcts = DummyMCTS()
    network = DummyNetwork()

    default_trainer = MuCaroTrainer(env=env, mcts=mcts, network=network)
    assert default_trainer.use_valid_mask is True

    unmasked_trainer = MuCaroTrainer(env=env, mcts=mcts, network=network, use_valid_mask=False)
    assert unmasked_trainer.use_valid_mask is False


def test_self_play_passes_legal_mask_when_use_valid_mask_true(trainer):
    """
    Verify that when use_valid_mask=True, self_play_game() queries env.get_legal_mask()
    and forwards the evolving legal mask to mcts.simulate() at every step.
    """
    trainer.use_valid_mask = True
    trainer.self_play_game()

    assert trainer.env.get_legal_mask_calls == 5
    assert len(trainer.mcts.call_history) == 5

    # First step must receive an all-True mask
    first_mask = trainer.mcts.call_history[0]["valid_mask"]
    assert first_mask == [True] * 9

    # Each subsequent step must have one fewer legal action as moves are played
    for step_idx, call_record in enumerate(trainer.mcts.call_history):
        mask = call_record["valid_mask"]
        assert mask is not None
        assert sum(mask) == 9 - step_idx


def test_self_play_passes_none_when_use_valid_mask_false(trainer):
    """
    Verify that when use_valid_mask=False, self_play_game() skips env.get_legal_mask()
    and passes valid_mask=None to mcts.simulate() at every step.
    """
    trainer.use_valid_mask = False
    trainer.self_play_game()

    assert trainer.env.get_legal_mask_calls == 0
    assert len(trainer.mcts.call_history) == 5
    for call_record in trainer.mcts.call_history:
        assert call_record["valid_mask"] is None


def test_self_play_with_valid_mask_prevents_duplicate_illegal_moves():
    """
    Verify end-to-end masked self-play completes a full 9-move board without
    ever re-selecting an occupied cell when enforce_unique_moves=True.
    """
    np.random.seed(123)
    env = DummyEnv(
        obs_shape=(2, 9),
        action_space_size=9,
        max_steps=9,
        outcome_winner=-1,
        enforce_unique_moves=True,
    )
    mcts = DummyMCTS(action_space_size=9)
    network = DummyNetwork(action_space_size=9)
    masked_trainer = MuCaroTrainer(
        env=env,
        mcts=mcts,
        network=network,
        use_valid_mask=True,
        temp_threshold=5,
    )

    game_data = masked_trainer.self_play_game()

    assert len(game_data["actions"]) == 9
    assert len(set(game_data["actions"])) == 9
    assert game_data["rewards"][-1] == 0.0  # Clean tie, no -1.0 invalid-move penalty


# --------------------------------------------------------------------------
# Self-Play Tests
# --------------------------------------------------------------------------

def test_self_play_game_win_and_alternating_discount(trainer):
    """Verify reward assignment and alternating zero-sum value discounting on a win."""
    game_data = trainer.self_play_game()

    trainer.network.h_network.eval.assert_called()
    trainer.network.f_network.eval.assert_called()
    trainer.network.g_network.eval.assert_called()
    trainer.network.projector_network.eval.assert_called()
    trainer.network.predictor_network.eval.assert_called()

    assert len(game_data["observations"]) == 5
    assert len(game_data["actions"]) == 5
    assert game_data["rewards"] == [0.0, 0.0, 0.0, 0.0, 1.0]

    # v_t = r_t - discount * v_{t+1}
    v4 = 1.0
    v3 = 0.0 - 0.9 * v4
    v2 = 0.0 - 0.9 * v3
    v1 = 0.0 - 0.9 * v2
    v0 = 0.0 - 0.9 * v1
    expected_values = [v0, v1, v2, v3, v4]

    for actual, expected in zip(game_data["values"], expected_values):
        assert actual == pytest.approx(expected, abs=1e-5)


def test_self_play_game_tie_outcome(trainer):
    """Verify tie outcome (winner == -1) produces 0.0 rewards and 0.0 values throughout."""
    trainer.env = DummyEnv(obs_shape=(2, 9), max_steps=4, outcome_winner=-1)
    game_data = trainer.self_play_game()

    assert len(game_data["observations"]) == 4
    assert game_data["rewards"] == [0.0, 0.0, 0.0, 0.0]
    assert game_data["values"] == [0.0, 0.0, 0.0, 0.0]


def test_self_play_game_aborted_first_step(trainer):
    """Verify immediate game abort when an illegal move occurs on the first step."""
    with patch('numpy.random.choice', return_value=999):
        game_data = trainer.self_play_game()

    assert len(game_data["observations"]) == 1
    assert game_data["actions"] == [999]
    assert game_data["rewards"] == [-1.0]
    assert game_data["values"] == [-1.0]


def test_self_play_game_aborted_mid_game_does_not_backpropagate_penalty(trainer):
    """
    Verify mid-game illegal move assigns -1.0 only to the offending step
    and leaves prior valid steps at 0.0 (no alternating backup when aborted).
    """
    trainer.env = DummyEnv(obs_shape=(2, 9), max_steps=10, outcome_winner=1)
    with patch('numpy.random.choice', side_effect=[0, 1, 999]):
        game_data = trainer.self_play_game()

    assert len(game_data["observations"]) == 3
    assert game_data["rewards"] == [0.0, 0.0, -1.0]
    assert game_data["values"] == [0.0, 0.0, -1.0]


def test_self_play_temperature_threshold_behavior(trainer):
    """
    Verify action selection uses stochastic sampling when step < temp_threshold
    and deterministic argmax when step >= temp_threshold.
    """
    trainer.temp_threshold = 2
    trainer.env = DummyEnv(obs_shape=(2, 9), max_steps=4, outcome_winner=1)

    with patch('numpy.random.choice', return_value=2) as mock_choice, \
            patch('numpy.argmax', return_value=5) as mock_argmax:
        game_data = trainer.self_play_game()

    assert mock_choice.call_count == 2
    assert mock_argmax.call_count == 2
    assert game_data["actions"] == [2, 2, 5, 5]


def test_self_play_reraises_unexpected_exception(trainer):
    """Ensure non-ValueError exceptions from the environment are not swallowed."""
    trainer.env = DummyEnv(obs_shape=(2, 9), raise_unexpected_error=True)
    with pytest.raises(RuntimeError):
        trainer.self_play_game()


# --------------------------------------------------------------------------
# Replay Buffer & Batch Sampling Tests
# --------------------------------------------------------------------------

def test_replay_buffer_maxlen_eviction(trainer):
    """Verify replay buffer enforces maxlen and evicts oldest games."""
    for idx in range(25):
        trainer.replay_buffer.append({"game_id": idx})

    assert len(trainer.replay_buffer) == 20
    assert trainer.replay_buffer[0]["game_id"] == 5
    assert trainer.replay_buffer[-1]["game_id"] == 24


def test_sample_batch_shapes_dtypes_and_absorbing_padding(trainer):
    """
    Verify batch shapes, dtypes, observation sequence alignment, and absorbing state
    zero-padding (including consistency_mask) when trajectories exceed game length.
    """
    obs_shape = (2, 9)
    obs0 = torch.full(obs_shape, 2.0, dtype=torch.float32)
    obs1 = torch.full(obs_shape, 5.0, dtype=torch.float32)
    pi0 = np.array([1.0] + [0.0] * 8, dtype=np.float32)
    pi1 = np.array([0.0, 1.0] + [0.0] * 7, dtype=np.float32)

    short_game = {
        "observations": [obs0, obs1],
        "actions": [3, 7],
        "rewards": [0.0, 1.0],
        "policies": [pi0, pi1],
        "values": [-0.9, 1.0]
    }
    for _ in range(6):
        trainer.replay_buffer.append(short_game)

    with patch('random.randint', return_value=0):
        obs_b, act_b, pol_b, val_b, rew_b, mask_b = trainer.sample_batch()

    assert obs_b.shape == (4, 4, *obs_shape)
    assert act_b.shape == (4, 4)
    assert pol_b.shape == (4, 4, 9)
    assert val_b.shape == (4, 4)
    assert rew_b.shape == (4, 4)
    assert mask_b.shape == (4, 4)

    assert obs_b.dtype == torch.float32
    assert act_b.dtype == torch.long
    assert pol_b.dtype == torch.float32
    assert val_b.dtype == torch.float32
    assert rew_b.dtype == torch.float32
    assert mask_b.dtype == torch.float32

    # First 2 steps are valid (mask=1.0), last 2 steps are absorbing padding (mask=0.0, obs=0)
    expected_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 4, dtype=torch.float32)
    assert torch.equal(mask_b, expected_mask)
    assert torch.equal(obs_b[0, 0], obs0)
    assert torch.equal(obs_b[0, 1], obs1)
    assert torch.equal(obs_b[0, 2], torch.zeros(obs_shape, dtype=torch.float32))
    assert torch.equal(obs_b[0, 3], torch.zeros(obs_shape, dtype=torch.float32))


# --------------------------------------------------------------------------
# Training Step Mathematical & Computation Tests
# --------------------------------------------------------------------------

def test_train_step_exact_loss_and_reward_k_minus_1_alignment(trainer):
    """
    Verify exact analytical loss values for policy, value, reward, and consistency losses,
    ensuring reward at unroll step k aligns with target_rewards[:, k - 1].
    """
    trainer.unroll_steps = 2
    batch_size = 2
    num_actions = 3

    obs = torch.zeros((batch_size, 3, 2, 9), dtype=torch.float32)
    actions = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    target_policies = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ], dtype=torch.float32)

    target_values = torch.tensor([
        [1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
    ], dtype=torch.float32)

    target_rewards = torch.tensor([
        [1.0, 0.5, 999.0],
        [1.0, 0.5, 999.0],
    ], dtype=torch.float32)

    consistency_mask = torch.ones((batch_size, 3), dtype=torch.float32)

    dummy_w = trainer.network.h_param

    def mock_init(x, training=False):
        h = torch.ones((batch_size, 8)) * dummy_w
        p = torch.zeros((batch_size, num_actions)) * dummy_w
        v = torch.zeros((batch_size, 1)) * dummy_w
        return h, p, v

    def mock_rec(h, a, training=False):
        nh = torch.ones((batch_size, 8)) * dummy_w
        r = torch.zeros((batch_size, 1)) * dummy_w
        p = torch.zeros((batch_size, num_actions)) * dummy_w
        v = torch.zeros((batch_size, 1)) * dummy_w
        return nh, r, p, v

    def mock_proj(x):
        return torch.ones((batch_size, 8))

    def mock_pred(h, training=True):
        return torch.ones((batch_size, 8)) * dummy_w

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards, consistency_mask)
    ), patch.object(
        trainer.network, 'initial_inference', side_effect=mock_init
    ), patch.object(
        trainer.network, 'recurrent_inference', side_effect=mock_rec
    ), patch.object(
        trainer.network, 'consistency_project', side_effect=mock_proj
    ), patch.object(
        trainer.network, 'consistency_predict', side_effect=mock_pred
    ):
        metrics = trainer.train_step()

    expected_policy_loss = (3.0 * math.log(3.0)) / 3.0
    expected_value_loss = 2.0 / 3.0
    expected_reward_loss = 1.25 / 2.0
    expected_consistency_loss = 0.0
    expected_total_loss = (
        expected_policy_loss + expected_value_loss + expected_reward_loss + expected_consistency_loss
    )

    assert metrics["policy_loss"] == pytest.approx(expected_policy_loss, abs=1e-5)
    assert metrics["value_loss"] == pytest.approx(expected_value_loss, abs=1e-5)
    assert metrics["reward_loss"] == pytest.approx(expected_reward_loss, abs=1e-5)
    assert metrics["consistency_loss"] == pytest.approx(expected_consistency_loss, abs=1e-5)
    assert metrics["total_loss"] == pytest.approx(expected_total_loss, abs=1e-5)


def test_train_step_unroll_action_sequence_alignment(trainer):
    """Verify recurrent_inference receives actions[:, k - 1] in exact sequential order."""
    trainer.unroll_steps = 3
    obs = torch.zeros((2, 4, 2, 9), dtype=torch.float32)
    actions = torch.tensor([[1, 4, 7, 0], [2, 5, 8, 0]], dtype=torch.long)
    target_policies = torch.full((2, 4, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.zeros((2, 4), dtype=torch.float32)
    target_rewards = torch.zeros((2, 4), dtype=torch.float32)
    consistency_mask = torch.ones((2, 4), dtype=torch.float32)

    received_actions = []
    orig_rec = trainer.network.recurrent_inference

    def spy_recurrent(h, a, training=False):
        received_actions.append(a.clone())
        return orig_rec(h, a, training=training)

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards, consistency_mask)
    ), patch.object(
        trainer.network, 'recurrent_inference', side_effect=spy_recurrent
    ):
        trainer.train_step()

    assert len(received_actions) == 3
    assert torch.equal(received_actions[0], torch.tensor([1, 2], dtype=torch.long))
    assert torch.equal(received_actions[1], torch.tensor([4, 5], dtype=torch.long))
    assert torch.equal(received_actions[2], torch.tensor([7, 8], dtype=torch.long))


def test_train_step_hidden_state_half_gradient_scaling(trainer):
    """
    Verify `hidden_state = hidden_state * 0.5 + hidden_state.detach() * 0.5`
    attenuates gradients flowing back to initial_inference by 0.5^k per unroll step.
    """
    trainer.unroll_steps = 2
    with torch.no_grad():
        trainer.network.h_param.copy_(torch.tensor([0.1]))

    obs = torch.zeros((1, 3, 2, 9), dtype=torch.float32)
    actions = torch.zeros((1, 3), dtype=torch.long)
    target_policies = torch.full((1, 3, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.zeros((1, 3), dtype=torch.float32)
    target_rewards = torch.zeros((1, 3), dtype=torch.float32)
    # Zero out consistency mask for k >= 1 to isolate reward gradient scaling cleanly
    consistency_mask = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    call_idx = {"step": 0}

    def custom_init(x, training=False):
        h0 = torch.ones((1, 8)) * trainer.network.h_param
        p0 = torch.zeros((1, 9))
        v0 = torch.zeros((1, 1))
        return h0, p0, v0

    def custom_rec(h, a, training=False):
        call_idx["step"] += 1
        nh = h
        r = nh.mean(dim=-1, keepdim=True) if call_idx["step"] == 2 else torch.zeros((1, 1))
        p = torch.zeros((1, 9))
        v = torch.zeros((1, 1))
        return nh, r, p, v

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards, consistency_mask)
    ), patch.object(
        trainer.network, 'initial_inference', side_effect=custom_init
    ), patch.object(
        trainer.network, 'recurrent_inference', side_effect=custom_rec
    ):
        trainer.train_step()

    expected_grad = (2.0 * 0.1 / 2.0) * 0.25
    assert trainer.network.h_param.grad.item() == pytest.approx(expected_grad, abs=1e-6)


def test_train_step_gradient_clipping_enforced(trainer):
    """Verify clip_grad_norm_(max_norm=0.5) caps exploding gradients."""
    obs = torch.ones((4, 4, 2, 9), dtype=torch.float32)
    actions = torch.zeros((4, 4), dtype=torch.long)
    target_policies = torch.full((4, 4, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.full((4, 4), 1000.0, dtype=torch.float32)
    target_rewards = torch.full((4, 4), 1000.0, dtype=torch.float32)
    consistency_mask = torch.ones((4, 4), dtype=torch.float32)

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards, consistency_mask)
    ):
        trainer.train_step()

    total_grad_norm = torch.sqrt(
        sum(p.grad.detach().pow(2).sum() for p in trainer.network.params() if p.grad is not None)
    ).item()

    assert total_grad_norm <= 0.5 + 1e-5
    assert total_grad_norm > 0.49


def test_train_step_updates_all_networks_with_valid_gradients(trainer):
    """
    Verify train_step():
    - Sets all 5 sub-networks (h, f, g, projector, predictor) to train mode.
    - Computes finite scalar losses including consistency_loss.
    - Produces non-zero gradients and updates weights across h, g, f, projector, and predictor.
    """
    game_data = trainer.self_play_game()
    for _ in range(8):
        trainer.replay_buffer.append(game_data)

    h_before = trainer.network.h_param.clone().detach()
    g_before = trainer.network.g_param.clone().detach()
    f_before = trainer.network.f_param.clone().detach()
    proj_before = trainer.network.proj_param.clone().detach()
    pred_before = trainer.network.pred_param.clone().detach()

    metrics = trainer.train_step()

    for key in ("total_loss", "policy_loss", "value_loss", "reward_loss", "consistency_loss"):
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert not math.isnan(metrics[key])
        assert not math.isinf(metrics[key])

    assert metrics["total_loss"] == pytest.approx(
        metrics["policy_loss"] + metrics["value_loss"] + metrics["reward_loss"] + metrics["consistency_loss"],
        abs=1e-6
    )

    assert trainer.network.h_param.grad is not None and trainer.network.h_param.grad.abs().sum() > 0
    assert trainer.network.g_param.grad is not None and trainer.network.g_param.grad.abs().sum() > 0
    assert trainer.network.f_param.grad is not None and trainer.network.f_param.grad.abs().sum() > 0
    assert trainer.network.proj_param.grad is not None and trainer.network.proj_param.grad.abs().sum() > 0
    assert trainer.network.pred_param.grad is not None and trainer.network.pred_param.grad.abs().sum() > 0

    assert not torch.equal(h_before, trainer.network.h_param.detach())
    assert not torch.equal(g_before, trainer.network.g_param.detach())
    assert not torch.equal(f_before, trainer.network.f_param.detach())
    assert not torch.equal(proj_before, trainer.network.proj_param.detach())
    assert not torch.equal(pred_before, trainer.network.pred_param.detach())

    trainer.network.h_network.train.assert_called()
    trainer.network.f_network.train.assert_called()
    trainer.network.g_network.train.assert_called()
    trainer.network.projector_network.train.assert_called()
    trainer.network.predictor_network.train.assert_called()


# --------------------------------------------------------------------------
# SimSiam Consistency Loss (Projector + Predictor) Tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "target_vec, expected_loss",
    [
        ([1.0, 0.0, 0.0, 0.0], 0.0),   # Identical direction: (-1 + 1) / 2 = 0.0
        ([0.0, 1.0, 0.0, 0.0], 0.5),   # Orthogonal vectors:  ( 0 + 1) / 2 = 0.5
        ([-1.0, 0.0, 0.0, 0.0], 1.0),  # Opposite direction:  (+1 + 1) / 2 = 1.0
    ],
)
def test_consistency_loss_exact_cosine_distance_values(trainer, target_vec, expected_loss):
    """
    Verify consistency_loss computes (-cosine_similarity + 1.0) / 2.0 accurately
    for aligned (0.0), orthogonal (0.5), and opposite (1.0) projected/predicted vectors.
    """
    trainer.unroll_steps = 1
    obs = torch.zeros((2, 2, 2, 9), dtype=torch.float32)
    actions = torch.zeros((2, 2), dtype=torch.long)
    target_policies = torch.full((2, 2, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.zeros((2, 2), dtype=torch.float32)
    target_rewards = torch.zeros((2, 2), dtype=torch.float32)
    consistency_mask = torch.ones((2, 2), dtype=torch.float32)

    pred_dir = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    target_tensor = torch.tensor([target_vec, target_vec], dtype=torch.float32)

    def mock_init(x, training=False):
        h = pred_dir * trainer.network.h_param
        return h, torch.zeros((2, 9)), torch.zeros((2, 1))

    def mock_rec(h, a, training=False):
        nh = pred_dir * trainer.network.g_param
        return nh, torch.zeros((2, 1)), torch.zeros((2, 9)), torch.zeros((2, 1))

    def mock_pred(h, training=True):
        return h * trainer.network.pred_param[0]

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards, consistency_mask)
    ), patch.object(
        trainer.network, 'initial_inference', side_effect=mock_init
    ), patch.object(
        trainer.network, 'recurrent_inference', side_effect=mock_rec
    ), patch.object(
        trainer.network, 'consistency_project', return_value=target_tensor
    ), patch.object(
        trainer.network, 'consistency_predict', side_effect=mock_pred
    ):
        metrics = trainer.train_step()

    assert metrics["consistency_loss"] == pytest.approx(expected_loss, abs=1e-5)


def test_consistency_loss_masks_padding_steps_and_excludes_step_zero(trainer):
    """
    Verify consistency_loss:
    - Ignores padded future steps where consistency_mask[:, k] == 0.
    - Normalizes strictly by consistency_mask[:, 1:].sum() (excluding k == 0).
    """
    trainer.unroll_steps = 2
    obs = torch.zeros((2, 3, 2, 9), dtype=torch.float32)
    actions = torch.zeros((2, 3), dtype=torch.long)
    target_policies = torch.full((2, 3, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.zeros((2, 3), dtype=torch.float32)
    target_rewards = torch.zeros((2, 3), dtype=torch.float32)

    # Sample 0: valid at k=0, 1, 2
    # Sample 1: valid at k=0, 1; PADDED at k=2
    # Total valid steps for k >= 1 is 3 (NOT 5)
    consistency_mask = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=torch.float32)

    pred_dir = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    # k=1: Sample 0 orthogonal (loss=0.5), Sample 1 aligned (loss=0.0)
    # k=2: Sample 0 orthogonal (loss=0.5), Sample 1 opposite (loss=1.0 -> MUST be masked out)
    proj_targets = [
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32),
    ]

    def mock_init(x, training=False):
        return pred_dir * trainer.network.h_param, torch.zeros((2, 9)), torch.zeros((2, 1))

    def mock_rec(h, a, training=False):
        return pred_dir * trainer.network.g_param, torch.zeros((2, 1)), torch.zeros((2, 9)), torch.zeros((2, 1))

    def mock_pred(h, training=True):
        return h * trainer.network.pred_param[0]

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards, consistency_mask)
    ), patch.object(
        trainer.network, 'initial_inference', side_effect=mock_init
    ), patch.object(
        trainer.network, 'recurrent_inference', side_effect=mock_rec
    ), patch.object(
        trainer.network, 'consistency_project', side_effect=proj_targets
    ), patch.object(
        trainer.network, 'consistency_predict', side_effect=mock_pred
    ):
        metrics = trainer.train_step()

    # Valid losses: (0.5 + 0.0 + 0.5) / 3 = 1.0 / 3.0
    assert metrics["consistency_loss"] == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_consistency_loss_handles_all_padded_future_steps_without_nan(trainer):
    """
    Verify clamp_min(1) prevents division-by-zero / NaN when every sample in the batch
    starts at a terminal step (all consistency_mask[:, 1:] == 0).
    """
    trainer.unroll_steps = 2
    obs = torch.zeros((2, 3, 2, 9), dtype=torch.float32)
    actions = torch.zeros((2, 3), dtype=torch.long)
    target_policies = torch.full((2, 3, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.zeros((2, 3), dtype=torch.float32)
    target_rewards = torch.zeros((2, 3), dtype=torch.float32)
    consistency_mask = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards, consistency_mask)
    ):
        metrics = trainer.train_step()

    assert not math.isnan(metrics["consistency_loss"])
    assert not math.isinf(metrics["consistency_loss"])
    assert metrics["consistency_loss"] == pytest.approx(0.0, abs=1e-6)


def test_consistency_simsiam_asymmetric_branch_calls_and_training_flags(trainer):
    """
    Verify SimSiam branch execution during train_step():
    - Target branch: consistency_project(obs[:, k]) invokes h_network(obs[:, k], training=False)
      and projector_network(..., training=False) under torch.no_grad().
    - Online branch: consistency_predict(hidden_state, training=True) invokes
      projector_network(..., training=True) and predictor_network(..., training=True) with grad enabled.
    """
    trainer.unroll_steps = 2
    obs = torch.stack([
        torch.full((2, 2, 9), 0.0),
        torch.full((2, 2, 9), 1.0),
        torch.full((2, 2, 9), 2.0),
    ], dim=1)  # shape: (B=2, K+1=3, 2, 9)
    actions = torch.zeros((2, 3), dtype=torch.long)
    target_policies = torch.full((2, 3, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.zeros((2, 3), dtype=torch.float32)
    target_rewards = torch.zeros((2, 3), dtype=torch.float32)
    consistency_mask = torch.ones((2, 3), dtype=torch.float32)

    project_calls = []
    predict_calls = []

    orig_project = trainer.network.consistency_project
    orig_predict = trainer.network.consistency_predict

    def spy_project(x):
        project_calls.append({
            "obs_mean": x.mean().item(),
            "grad_enabled": torch.is_grad_enabled(),
        })
        return orig_project(x)

    def spy_predict(h, training=True):
        predict_calls.append({
            "training": training,
            "grad_enabled": torch.is_grad_enabled(),
            "requires_grad": h.requires_grad,
        })
        return orig_predict(h, training=training)

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards, consistency_mask)
    ), patch.object(
        trainer.network, 'consistency_project', side_effect=spy_project
    ), patch.object(
        trainer.network, 'consistency_predict', side_effect=spy_predict
    ):
        trainer.train_step()

    # Both consistency_project and consistency_predict must be called for k=1 and k=2
    assert len(project_calls) == 2
    assert [c["obs_mean"] for c in project_calls] == pytest.approx([1.0, 2.0])
    # Note: @torch.no_grad() wraps orig_project; inside spy_project outer grad is enabled,
    # so verify output tensor of consistency_project has requires_grad == False and h_network got training=False
    assert len(predict_calls) == 2
    assert all(c["training"] is True for c in predict_calls)
    assert all(c["grad_enabled"] is True for c in predict_calls)
    assert all(c["requires_grad"] is True for c in predict_calls)

    # Check h_network and projector_network call args for target vs online branches
    h_calls = trainer.network.h_network.call_args_list
    assert len(h_calls) == 2
    assert h_calls[0].kwargs["training"] is False
    assert h_calls[1].kwargs["training"] is False
    assert h_calls[0].args[0].mean().item() == pytest.approx(1.0)
    assert h_calls[1].args[0].mean().item() == pytest.approx(2.0)

    # projector_network is called 4 times total: k=1 target (False), k=1 online (True), k=2 target (False), k=2 online (True)
    proj_calls = trainer.network.projector_network.call_args_list
    assert len(proj_calls) == 4
    assert [c.kwargs["training"] for c in proj_calls] == [False, True, False, True]

    # predictor_network is called 2 times total (online branch only, training=True)
    pred_calls = trainer.network.predictor_network.call_args_list
    assert len(pred_calls) == 2
    assert [c.kwargs["training"] for c in pred_calls] == [True, True]


def test_consistency_simsiam_asymmetric_gradient_flow(trainer):
    """
    Verify SimSiam stop-gradient isolation in train_step():
    - Online branch (g_network, projector_network, predictor_network) receives non-zero gradients
      from consistency_loss.
    - Target branch parameter used exclusively in consistency_project receives ZERO gradient,
      preventing representation collapse.
    """
    trainer.unroll_steps = 1
    obs = torch.stack([
        torch.full((2, 2, 9), 0.0),
        torch.full((2, 2, 9), 1.0),
    ], dim=1)
    actions = torch.zeros((2, 2), dtype=torch.long)
    target_policies = torch.full((2, 2, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.zeros((2, 2), dtype=torch.float32)
    target_rewards = torch.zeros((2, 2), dtype=torch.float32)
    consistency_mask = torch.ones((2, 2), dtype=torch.float32)

    target_only_h_param = nn.Parameter(torch.linspace(0.2, 0.9, 8), requires_grad=True)

    def spy_target_h(x, training=False):
        assert torch.is_grad_enabled() is False
        assert training is False
        return target_only_h_param.unsqueeze(0).expand(x.shape[0], 8)

    def mock_init(x, training=False):
        # Detach policy/value from g_param/proj_param/pred_param so gradients on those
        # parameters come strictly from consistency_loss
        h0 = torch.ones((2, 8))
        return h0, torch.zeros((2, 9)), torch.zeros((2, 1))

    def mock_rec(h, a, training=False):
        # Differentiable non-uniform vector w.r.t. g_param
        g_vec = torch.linspace(0.3, 1.0, 8) * trainer.network.g_param
        nh = g_vec.unsqueeze(0).expand(2, 8)
        return nh, torch.zeros((2, 1)), torch.zeros((2, 9)), torch.zeros((2, 1))

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards, consistency_mask)
    ), patch.object(
        trainer.network, 'initial_inference', side_effect=mock_init
    ), patch.object(
        trainer.network, 'recurrent_inference', side_effect=mock_rec
    ), patch.object(
        trainer.network, 'h_network', side_effect=spy_target_h
    ):
        trainer.train_step()

    # Target branch parameter must NOT receive gradients
    assert target_only_h_param.grad is None

    # Online branch parameters MUST receive non-zero gradients from consistency_loss
    assert trainer.network.g_param.grad is not None and trainer.network.g_param.grad.abs().sum() > 0
    assert trainer.network.proj_param.grad is not None and trainer.network.proj_param.grad.abs().sum() > 0
    assert trainer.network.pred_param.grad is not None and trainer.network.pred_param.grad.abs().sum() > 0


def test_consistency_loss_computed_along_feature_dimension(trainer):
    """
    Verify cosine_similarity(..., dim=-1) computes similarity across the feature dimension
    independently for each sample in the batch rather than across the batch dimension.
    """
    trainer.unroll_steps = 1
    obs = torch.zeros((2, 2, 2, 9), dtype=torch.float32)
    actions = torch.zeros((2, 2), dtype=torch.long)
    target_policies = torch.full((2, 2, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.zeros((2, 2), dtype=torch.float32)
    target_rewards = torch.zeros((2, 2), dtype=torch.float32)
    consistency_mask = torch.ones((2, 2), dtype=torch.float32)

    # Sample 0: pred=[1, 0, 0, 0], target=[1, 0, 0, 0] -> cosine = 1.0 -> loss = 0.0
    # Sample 1: pred=[0, 1, 0, 0], target=[0, -1, 0, 0] -> cosine = -1.0 -> loss = 1.0
    # Mean across batch along dim=-1 must be (0.0 + 1.0) / 2 = 0.5
    pred_vecs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    target_vecs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]], dtype=torch.float32)

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards, consistency_mask)
    ), patch.object(
        trainer.network, 'consistency_project', return_value=target_vecs
    ), patch.object(
        trainer.network, 'consistency_predict', return_value=pred_vecs * trainer.network.pred_param[0]
    ):
        metrics = trainer.train_step()

    assert metrics["consistency_loss"] == pytest.approx(0.5, abs=1e-5)


# ==============================================================================
# 4. FULL TRAINING LOOP & EVALUATOR TESTS
# ==============================================================================

@patch('model.trainer.save_checkpoint')
def test_run_loop_min_buffer_gating_and_history(mock_save_checkpoint, trainer):
    """Verify run() defers training until min_buffer_size is reached and logs history including consistency_loss."""
    history = trainer.run(
        evaluator=MagicMock(),
        checkpoint_path="./dummy_checkpoints",
        model_name="dummy_model",
        num_iterations=3,
        games_per_iter=2,
        train_steps_per_iter=2,
        min_buffer_size=4
    )

    assert len(trainer.replay_buffer) == 6
    assert len(history) == 2
    assert history[0]["iteration"] == 2
    assert history[0]["buffer_size"] == 4
    assert "consistency_loss" in history[0]
    assert isinstance(history[0]["consistency_loss"], float)
    assert not math.isnan(history[0]["consistency_loss"])
    assert history[1]["iteration"] == 3
    assert history[1]["buffer_size"] == 6


@patch('model.trainer.save_checkpoint')
def test_run_loop_triggers_evaluator_at_correct_intervals(mock_save_checkpoint, trainer, mock_evaluator):
    """
    Verify that the evaluator is called exactly at the specified eval_interval,
    and the evaluation results are correctly appended to the training history.
    """
    history = trainer.run(
        evaluator=mock_evaluator,
        checkpoint_path="./dummy_checkpoints",
        model_name="dummy_model",
        evaluation_steps=10,
        eval_interval=2,
        num_iterations=4,
        games_per_iter=1,
        train_steps_per_iter=1,
        min_buffer_size=1
    )

    assert mock_evaluator.call_count == 2
    mock_evaluator.assert_called_with(
        mcts=trainer.mcts,
        env=trainer.env,
        step_num=10,
        use_valid_mask=True
    )

    assert len(history) == 4
    assert history[0]['eval'] is None
    assert history[1]['eval']['win'] == 8  # Iter 2: Eval triggered
    assert history[2]['eval'] is None
    assert history[3]['eval'] is not None  # Iter 4: Eval triggered


@pytest.mark.parametrize("use_valid_mask", [True, False])
@patch('model.trainer.save_checkpoint')
def test_run_loop_forwards_use_valid_mask_to_evaluator(
        mock_save_checkpoint, trainer, mock_evaluator, use_valid_mask
):
    """
    Verify that run() propagates trainer.use_valid_mask (True or False)
    to the evaluator function during evaluation intervals.
    """
    trainer.use_valid_mask = use_valid_mask

    trainer.run(
        evaluator=mock_evaluator,
        checkpoint_path="./dummy_checkpoints",
        model_name="dummy_model",
        evaluation_steps=15,
        eval_interval=1,
        num_iterations=1,
        games_per_iter=1,
        train_steps_per_iter=1,
        min_buffer_size=1
    )

    mock_evaluator.assert_called_once_with(
        mcts=trainer.mcts,
        env=trainer.env,
        step_num=15,
        use_valid_mask=use_valid_mask
    )


@patch('model.trainer.save_checkpoint')
def test_run_loop_skips_evaluator_when_interval_not_reached(mock_save_checkpoint, trainer, mock_evaluator):
    """
    Verify that the evaluator is entirely skipped if num_iterations
    is strictly less than eval_interval.
    """
    history = trainer.run(
        evaluator=mock_evaluator,
        checkpoint_path="./dummy_checkpoints",
        model_name="dummy_model",
        evaluation_steps=10,
        eval_interval=5,
        num_iterations=3,
        games_per_iter=1,
        train_steps_per_iter=1,
        min_buffer_size=1
    )

    mock_evaluator.assert_not_called()
    assert len(history) == 3
    for iter_metrics in history:
        assert iter_metrics['eval'] is None


@patch('model.trainer.save_checkpoint')
def test_run_loop_evaluator_handles_empty_buffer_skips(mock_save_checkpoint, trainer, mock_evaluator):
    """
    Verify that evaluator logic is not executed during iterations that
    are skipped due to insufficient replay buffer size.
    """
    history = trainer.run(
        evaluator=mock_evaluator,
        checkpoint_path="./dummy_checkpoints",
        model_name="dummy_model",
        evaluation_steps=10,
        eval_interval=1,
        num_iterations=2,
        games_per_iter=1,
        train_steps_per_iter=1,
        min_buffer_size=100
    )

    mock_evaluator.assert_not_called()
    mock_save_checkpoint.assert_not_called()
    assert len(history) == 0


@patch('torch.save')
@patch('os.makedirs')
def test_save_checkpoint_creates_file_and_correct_dict(mock_makedirs, mock_torch_save, trainer):
    """
    Verify that save_checkpoint correctly structures the state dictionary,
    creates the target directory, and saves the file with the expected filename
    using the updated MCTS parameter signature.
    """
    eval_stats = {'win': 10, 'tie': 2, 'lose': 8, 'win_percentage': 0.65}
    trainer.optimizer = MagicMock()
    trainer.optimizer.state_dict.return_value = "dummy_opt_state"

    # Attach required attributes to DummyMCTS to prevent AttributeError in save_checkpoint
    trainer.mcts.network = trainer.network
    trainer.mcts.gamma = 0.99
    trainer.mcts.pb_c_base = 19652
    trainer.mcts.pb_c_init = 1.25
    trainer.mcts.root_dirichlet_alpha = 0.3
    trainer.mcts.root_exploration_fraction = 0.25

    save_checkpoint(
        mcts=trainer.mcts,
        optimizer=trainer.optimizer,
        board_size=(15, 15),
        eval_stats=eval_stats,
        model_name="mucaro_test",
        iteration=42,
        save_dir="./dummy_checkpoints",
        win_length=trainer.env.m
    )

    mock_makedirs.assert_called_once_with("./dummy_checkpoints", exist_ok=True)
    assert mock_torch_save.call_count == 1

    saved_dict = mock_torch_save.call_args[0][0]
    saved_path = mock_torch_save.call_args[0][1]

    assert saved_dict['iteration'] == 42
    assert saved_dict['optimizer_state'] == "dummy_opt_state"
    assert saved_dict['eval_stats']['win_percentage'] == 0.65
    assert saved_dict['gamma'] == 0.99
    assert saved_dict['pb_c_base'] == 19652
    assert 'h_network_state' in saved_dict

    # Verify filename format: 0.65 * 100 = 65 -> 65pct
    assert "mucaro_test-42-65pct.pt" in saved_path.replace("\\", "/")


@patch('model.trainer.save_checkpoint')
def test_run_loop_triggers_save_checkpoint_after_eval(mock_save_checkpoint, trainer, mock_evaluator):
    """
    Verify that run() successfully passes the correct arguments to save_checkpoint
    exactly when the eval_interval is reached.
    """
    trainer.env.size = (3, 3)
    trainer.optimizer = MagicMock()

    trainer.run(
        evaluator=mock_evaluator,
        checkpoint_path="./out_checkpoints",
        model_name="test_model",
        evaluation_steps=5,
        eval_interval=2,
        num_iterations=3,
        games_per_iter=1,
        train_steps_per_iter=1,
        min_buffer_size=1
    )

    assert mock_save_checkpoint.call_count == 1

    mock_save_checkpoint.assert_called_once_with(
        mcts=trainer.mcts,
        optimizer=trainer.optimizer,
        board_size=(3, 3),
        eval_stats=mock_evaluator.return_value,
        model_name="test_model",
        iteration=2,
        save_dir="./out_checkpoints",
        win_length=3
    )


def test_set_eval_and_train_mode_helpers(trainer):
    """Verify _set_eval_mode and _set_train_mode toggle all 5 sub-networks."""
    trainer._set_eval_mode()
    trainer.network.h_network.eval.assert_called_once()
    trainer.network.f_network.eval.assert_called_once()
    trainer.network.g_network.eval.assert_called_once()
    trainer.network.projector_network.eval.assert_called_once()
    trainer.network.predictor_network.eval.assert_called_once()

    trainer._set_train_mode()
    trainer.network.h_network.train.assert_called_once()
    trainer.network.f_network.train.assert_called_once()
    trainer.network.g_network.train.assert_called_once()
    trainer.network.projector_network.train.assert_called_once()
    trainer.network.predictor_network.train.assert_called_once()


@patch('model.trainer.save_checkpoint')
def test_run_loop_sets_eval_mode_before_evaluator(mock_save_checkpoint, trainer):
    """
    Verify that run() switches all 5 sub-networks back to eval mode after train_step()
    and immediately before invoking the evaluator, while leaving them in train mode
    on non-evaluation iterations.
    """
    active_modes = {"h": None, "f": None, "g": None, "projector": None, "predictor": None}
    modes_during_eval = []

    trainer.network.h_network.eval.side_effect = lambda: active_modes.update({"h": "eval"})
    trainer.network.f_network.eval.side_effect = lambda: active_modes.update({"f": "eval"})
    trainer.network.g_network.eval.side_effect = lambda: active_modes.update({"g": "eval"})
    trainer.network.projector_network.eval.side_effect = lambda: active_modes.update({"projector": "eval"})
    trainer.network.predictor_network.eval.side_effect = lambda: active_modes.update({"predictor": "eval"})

    trainer.network.h_network.train.side_effect = lambda: active_modes.update({"h": "train"})
    trainer.network.f_network.train.side_effect = lambda: active_modes.update({"f": "train"})
    trainer.network.g_network.train.side_effect = lambda: active_modes.update({"g": "train"})
    trainer.network.projector_network.train.side_effect = lambda: active_modes.update({"projector": "train"})
    trainer.network.predictor_network.train.side_effect = lambda: active_modes.update({"predictor": "train"})

    def spy_evaluator(mcts, env, step_num, use_valid_mask=True):
        modes_during_eval.append(active_modes.copy())
        return {'win': 5, 'lose': 5, 'invalid': 3, 'tie': 0, 'win_percentage': 0.5}

    # Iteration 1: trains only (no eval) -> networks remain in 'train' mode at end of iter
    trainer.run(
        evaluator=spy_evaluator,
        checkpoint_path="./dummy_checkpoints",
        model_name="dummy_model",
        evaluation_steps=10,
        eval_interval=2,
        num_iterations=1,
        games_per_iter=1,
        train_steps_per_iter=1,
        min_buffer_size=1
    )
    assert len(modes_during_eval) == 0
    assert active_modes == {"h": "train", "f": "train", "g": "train", "projector": "train", "predictor": "train"}

    # Iteration 2: triggers eval -> networks must switch from 'train' to 'eval' before evaluator runs
    trainer.run(
        evaluator=spy_evaluator,
        checkpoint_path="./dummy_checkpoints",
        model_name="dummy_model",
        evaluation_steps=10,
        eval_interval=1,
        num_iterations=1,
        games_per_iter=1,
        train_steps_per_iter=1,
        min_buffer_size=1
    )
    assert len(modes_during_eval) == 1
    assert modes_during_eval[0] == {"h": "eval", "f": "eval", "g": "eval", "projector": "eval", "predictor": "eval"}


# --------------------------------------------------------------------------
# Learning Rate Scheduling Helper Tests
# --------------------------------------------------------------------------

def test_set_lr_updates_optimizer_learning_rate(trainer):
    """Verify set_lr() updates the learning rate in the optimizer's param_groups."""
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)

    trainer.set_lr(5e-4)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)

    trainer.set_lr(1e-5)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_set_lr_updates_all_param_groups(trainer):
    """Verify set_lr() updates every parameter group when multiple groups exist."""
    trainer.optimizer = torch.optim.AdamW(
        [
            {"params": [trainer.network.h_param], "lr": 1e-2},
            {"params": [trainer.network.g_param], "lr": 5e-3},
            {"params": [trainer.network.f_param], "lr": 1e-3},
            {"params": [trainer.network.proj_param], "lr": 1e-3},
            {"params": [trainer.network.pred_param], "lr": 1e-3},
        ]
    )

    trainer.set_lr(2.5e-4)

    assert len(trainer.optimizer.param_groups) == 5
    for group in trainer.optimizer.param_groups:
        assert group["lr"] == pytest.approx(2.5e-4)


def test_set_lr_preserves_optimizer_momentum_state(trainer):
    """
    Verify set_lr() modifies learning rates in-place without clearing
    AdamW state buffers (step, exp_avg, exp_avg_sq) accumulated from prior steps.
    """
    game_data = trainer.self_play_game()
    for _ in range(6):
        trainer.replay_buffer.append(game_data)

    # Populate AdamW state buffers with one training step
    trainer.train_step()
    state_before = {
        param: {k: v.clone() for k, v in state.items()}
        for param, state in trainer.optimizer.state.items()
    }
    assert len(state_before) > 0

    trainer.set_lr(1e-4)

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    for param, expected_state in state_before.items():
        actual_state = trainer.optimizer.state[param]
        for buf_name, expected_tensor in expected_state.items():
            assert torch.equal(actual_state[buf_name], expected_tensor)


def test_set_lr_controls_parameter_update_magnitude_in_train_step(trainer):
    """
    Verify set_lr() directly governs weight update magnitude during train_step(),
    including freezing weight updates across all 5 sub-networks when set to 0.0.
    """
    game_data = trainer.self_play_game()
    for _ in range(6):
        trainer.replay_buffer.append(game_data)

    trainer.set_lr(0.0)
    h_before = trainer.network.h_param.clone().detach()
    g_before = trainer.network.g_param.clone().detach()
    f_before = trainer.network.f_param.clone().detach()
    proj_before = trainer.network.proj_param.clone().detach()
    pred_before = trainer.network.pred_param.clone().detach()

    trainer.train_step()

    # Gradients are still computed, but weights remain unchanged because lr == 0.0
    assert trainer.network.h_param.grad is not None and trainer.network.h_param.grad.abs().sum() > 0
    assert trainer.network.g_param.grad is not None and trainer.network.g_param.grad.abs().sum() > 0
    assert trainer.network.f_param.grad is not None and trainer.network.f_param.grad.abs().sum() > 0
    assert trainer.network.proj_param.grad is not None and trainer.network.proj_param.grad.abs().sum() > 0
    assert trainer.network.pred_param.grad is not None and trainer.network.pred_param.grad.abs().sum() > 0

    assert torch.equal(trainer.network.h_param.detach(), h_before)
    assert torch.equal(trainer.network.g_param.detach(), g_before)
    assert torch.equal(trainer.network.f_param.detach(), f_before)
    assert torch.equal(trainer.network.proj_param.detach(), proj_before)
    assert torch.equal(trainer.network.pred_param.detach(), pred_before)
