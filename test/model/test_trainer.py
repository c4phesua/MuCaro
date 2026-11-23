import math
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import torch.nn as nn
import os

from model.trainer import MuCaroTrainer
from model.utils import save_checkpoint


# ==============================================================================
# 1. DUMMY CLASSES TO MOCK ENVIRONMENT, MCTS, AND NETWORK
# ==============================================================================

class DummyEnv:
    """
    Configurable dummy environment matching Board contract:
    - Returns torch.Tensor for 'next_state' (compatible with torch.stack in sample_batch).
    - Supports decisive wins (winner > 0), ties (winner == -1), invalid moves (ValueError),
      and unexpected runtime errors.
    """

    def __init__(
            self,
            obs_shape=(2, 9),
            max_steps=5,
            outcome_winner=1,
            raise_unexpected_error=False
    ):
        self.obs_shape = obs_shape
        self.max_steps = max_steps
        self.outcome_winner = outcome_winner
        self.raise_unexpected_error = raise_unexpected_error
        self.step_count = 0
        self.size = 15  # Added to prevent AttributeError when saving checkpoints
        self.m = 3

    def reset(self):
        self.step_count = 0
        return {'next_state': torch.zeros(self.obs_shape, dtype=torch.float32)}

    def play(self, action):
        if self.raise_unexpected_error:
            raise RuntimeError("Unexpected environment failure")

        if action == 999:
            raise ValueError("Invalid move")

        self.step_count += 1
        done = self.step_count >= self.max_steps
        winner = self.outcome_winner if done else 0

        return {
            'next_state': torch.full(self.obs_shape, float(self.step_count), dtype=torch.float32),
            'done': done,
            'winner': winner
        }


class DummyMCTS:
    """Dummy MCTS returning a valid probability distribution over the action space."""

    def __init__(self, action_space_size=9):
        self.action_space_size = action_space_size
        self.call_history = []

    def simulate(self, current_state, num_simulations, is_training):
        self.call_history.append({
            "state_shape": tuple(current_state.shape),
            "state_dtype": current_state.dtype,
            "num_simulations": num_simulations,
            "is_training": is_training,
        })
        pi = np.ones(self.action_space_size, dtype=np.float32)
        pi /= np.sum(pi)
        return pi


class DummyNetwork(nn.Module):
    """
    Dummy Network inheriting from nn.Module with non-zero forward outputs
    so PyTorch computes true non-zero gradients across h, g, and f networks.
    """

    def __init__(self, action_space_size=9, hidden_dim=8):
        super().__init__()
        self.action_space_size = action_space_size
        self.hidden_dim = hidden_dim

        # Separate parameters for each sub-network to verify full BPTT gradient flow
        self.h_param = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.g_param = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.f_param = nn.Parameter(torch.tensor([0.5], requires_grad=True))

        # Mock sub-modules to verify .train() and .eval() mode transitions
        self.h_network = MagicMock()
        self.f_network = MagicMock()
        self.g_network = MagicMock()

    def params(self):
        return self.parameters()

    def initial_inference(self, obs, training=False):
        batch_size = obs.shape[0]
        obs_mean = obs.view(batch_size, -1).mean(dim=-1, keepdim=True) + 1.0
        hidden_state = obs_mean.expand(batch_size, self.hidden_dim) * self.h_param
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


# ==============================================================================
# 2. PYTEST FIXTURES
# ==============================================================================

@pytest.fixture
def trainer():
    """Provides a fresh instance of MuCaroTrainer for each test."""
    np.random.seed(42)
    torch.manual_seed(42)

    env = DummyEnv(obs_shape=(2, 9), max_steps=5, outcome_winner=1)
    mcts = DummyMCTS(action_space_size=9)
    network = DummyNetwork(action_space_size=9)

    return MuCaroTrainer(
        env=env,
        mcts=mcts,
        network=network,
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
        'tie': 0,
        'win_percentage': 0.8
    }
    return evaluator


# ==============================================================================
# 3. UNIT TESTS (PYTEST)
# ==============================================================================

# --------------------------------------------------------------------------
# Self-Play Tests
# --------------------------------------------------------------------------

def test_self_play_game_win_and_alternating_discount(trainer):
    """Verify reward assignment and alternating zero-sum value discounting on a win."""
    game_data = trainer.self_play_game()

    trainer.network.h_network.eval.assert_called()
    trainer.network.f_network.eval.assert_called()
    trainer.network.g_network.eval.assert_called()

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
    Verify batch shapes, dtypes, and absorbing state zero-padding
    when sampled trajectories exceed the end of a game.
    """
    obs_shape = (2, 9)
    obs0 = torch.zeros(obs_shape, dtype=torch.float32)
    obs1 = torch.ones(obs_shape, dtype=torch.float32)
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
        obs_b, act_b, pol_b, val_b, rew_b = trainer.sample_batch()

    assert obs_b.shape == (4, *obs_shape)
    assert act_b.shape == (4, 4)
    assert pol_b.shape == (4, 4, 9)
    assert val_b.shape == (4, 4)
    assert rew_b.shape == (4, 4)

    assert obs_b.dtype == torch.float32
    assert act_b.dtype == torch.long
    assert pol_b.dtype == torch.float32
    assert val_b.dtype == torch.float32
    assert rew_b.dtype == torch.float32


# --------------------------------------------------------------------------
# Training Step Mathematical & Computation Tests
# --------------------------------------------------------------------------

def test_train_step_exact_loss_and_reward_k_minus_1_alignment(trainer):
    """
    Verify exact analytical loss values scaled by 1 / (unroll_steps + 1),
    specifically ensuring reward at unroll step k aligns with target_rewards[:, k - 1].
    """
    trainer.unroll_steps = 2
    batch_size = 2
    num_actions = 3

    obs = torch.zeros((batch_size, 2, 9), dtype=torch.float32)
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

    dummy_w = trainer.network.h_param

    def mock_init(x, training=False):
        h = torch.zeros((batch_size, 8)) * dummy_w
        p = torch.zeros((batch_size, num_actions)) * dummy_w
        v = torch.zeros((batch_size, 1)) * dummy_w
        return h, p, v

    def mock_rec(h, a, training=False):
        nh = h * dummy_w
        r = torch.zeros((batch_size, 1)) * dummy_w
        p = torch.zeros((batch_size, num_actions)) * dummy_w
        v = torch.zeros((batch_size, 1)) * dummy_w
        return nh, r, p, v

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards)
    ), patch.object(
        trainer.network, 'initial_inference', side_effect=mock_init
    ), patch.object(
        trainer.network, 'recurrent_inference', side_effect=mock_rec
    ):
        metrics = trainer.train_step()

    expected_policy_loss = (3.0 * math.log(3.0)) / 3.0
    expected_value_loss = 2.0 / 3.0
    expected_reward_loss = 1.25 / 2.0
    expected_total_loss = expected_policy_loss + expected_value_loss + expected_reward_loss

    assert metrics["policy_loss"] == pytest.approx(expected_policy_loss, abs=1e-5)
    assert metrics["value_loss"] == pytest.approx(expected_value_loss, abs=1e-5)
    assert metrics["reward_loss"] == pytest.approx(expected_reward_loss, abs=1e-5)
    assert metrics["total_loss"] == pytest.approx(expected_total_loss, abs=1e-5)


def test_train_step_unroll_action_sequence_alignment(trainer):
    """Verify recurrent_inference receives actions[:, k - 1] in exact sequential order."""
    trainer.unroll_steps = 3
    obs = torch.zeros((2, 2, 9), dtype=torch.float32)
    actions = torch.tensor([[1, 4, 7, 0], [2, 5, 8, 0]], dtype=torch.long)
    target_policies = torch.full((2, 4, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.zeros((2, 4), dtype=torch.float32)
    target_rewards = torch.zeros((2, 4), dtype=torch.float32)

    received_actions = []
    orig_rec = trainer.network.recurrent_inference

    def spy_recurrent(h, a, training=False):
        received_actions.append(a.clone())
        return orig_rec(h, a, training=training)

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards)
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

    obs = torch.zeros((1, 2, 9), dtype=torch.float32)
    actions = torch.zeros((1, 3), dtype=torch.long)
    target_policies = torch.full((1, 3, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.zeros((1, 3), dtype=torch.float32)
    target_rewards = torch.zeros((1, 3), dtype=torch.float32)

    call_idx = {"step": 0}

    def custom_init(x, training=False):
        h0 = torch.ones((1, 1)) * trainer.network.h_param
        p0 = torch.zeros((1, 9))
        v0 = torch.zeros((1, 1))
        return h0, p0, v0

    def custom_rec(h, a, training=False):
        call_idx["step"] += 1
        nh = h
        r = nh if call_idx["step"] == 2 else torch.zeros((1, 1))
        p = torch.zeros((1, 9))
        v = torch.zeros((1, 1))
        return nh, r, p, v

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards)
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
    obs = torch.ones((4, 2, 9), dtype=torch.float32)
    actions = torch.zeros((4, 4), dtype=torch.long)
    target_policies = torch.full((4, 4, 9), 1.0 / 9.0, dtype=torch.float32)
    target_values = torch.full((4, 4), 1000.0, dtype=torch.float32)
    target_rewards = torch.full((4, 4), 1000.0, dtype=torch.float32)

    with patch.object(
            trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards)
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
    - Sets all sub-networks to train mode.
    - Computes finite scalar losses.
    - Produces non-zero gradients and updates weights across h, g, and f networks.
    """
    game_data = trainer.self_play_game()
    for _ in range(8):
        trainer.replay_buffer.append(game_data)

    h_before = trainer.network.h_param.item()
    g_before = trainer.network.g_param.item()
    f_before = trainer.network.f_param.item()

    metrics = trainer.train_step()

    for key in ("total_loss", "policy_loss", "value_loss", "reward_loss"):
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert not math.isnan(metrics[key])
        assert not math.isinf(metrics[key])

    assert metrics["total_loss"] == pytest.approx(
        metrics["policy_loss"] + metrics["value_loss"] + metrics["reward_loss"], abs=1e-6
    )

    assert trainer.network.h_param.grad is not None
    assert trainer.network.g_param.grad is not None
    assert trainer.network.f_param.grad is not None
    assert trainer.network.h_param.grad.item() != 0.0
    assert trainer.network.g_param.grad.item() != 0.0
    assert trainer.network.f_param.grad.item() != 0.0

    assert h_before != trainer.network.h_param.item()
    assert g_before != trainer.network.g_param.item()
    assert f_before != trainer.network.f_param.item()

    trainer.network.h_network.train.assert_called()
    trainer.network.f_network.train.assert_called()
    trainer.network.g_network.train.assert_called()


# ==============================================================================
# 4. Full Training Loop & Evaluator Tests
# ==============================================================================

@patch('model.trainer.save_checkpoint')
def test_run_loop_min_buffer_gating_and_history(mock_save_checkpoint, trainer):
    """Verify run() defers training until min_buffer_size is reached and logs history."""
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
        step_num=10
    )

    assert len(history) == 4
    assert history[0]['eval'] is None
    assert history[1]['eval']['win'] == 8  # Iter 2: Eval triggered
    assert history[2]['eval'] is None
    assert history[3]['eval'] is not None  # Iter 4: Eval triggered


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

    # Call save_checkpoint using the updated signature
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

    # Verify newly saved dictionary keys
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

    # Verify trainer.run() passes the expected arguments matching the new signature
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
    """Verify _set_eval_mode and _set_train_mode toggle all three sub-networks."""
    trainer._set_eval_mode()
    trainer.network.h_network.eval.assert_called_once()
    trainer.network.f_network.eval.assert_called_once()
    trainer.network.g_network.eval.assert_called_once()

    trainer._set_train_mode()
    trainer.network.h_network.train.assert_called_once()
    trainer.network.f_network.train.assert_called_once()
    trainer.network.g_network.train.assert_called_once()


@patch('model.trainer.save_checkpoint')
def test_run_loop_sets_eval_mode_before_evaluator(mock_save_checkpoint, trainer):
    """
    Verify that run() switches all sub-networks back to eval mode after train_step()
    and immediately before invoking the evaluator, while leaving them in train mode
    on non-evaluation iterations.
    """
    active_modes = {"h": None, "f": None, "g": None}
    modes_during_eval = []

    trainer.network.h_network.eval.side_effect = lambda: active_modes.update({"h": "eval"})
    trainer.network.f_network.eval.side_effect = lambda: active_modes.update({"f": "eval"})
    trainer.network.g_network.eval.side_effect = lambda: active_modes.update({"g": "eval"})

    trainer.network.h_network.train.side_effect = lambda: active_modes.update({"h": "train"})
    trainer.network.f_network.train.side_effect = lambda: active_modes.update({"f": "train"})
    trainer.network.g_network.train.side_effect = lambda: active_modes.update({"g": "train"})

    def spy_evaluator(mcts, env, step_num):
        # Capture the exact network modes active when evaluator executes
        modes_during_eval.append(active_modes.copy())
        return {'win': 5, 'lose': 5, 'tie': 0, 'win_percentage': 0.5}

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
    assert active_modes == {"h": "train", "f": "train", "g": "train"}

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
    assert modes_during_eval[0] == {"h": "eval", "f": "eval", "g": "eval"}
