import math
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import torch.nn as nn

from model.trainer import MuCaroTrainer


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
# 2. UNIT TESTS
# ==============================================================================

class TestMuCaroTrainer(unittest.TestCase):
    def setUp(self):
        """Initialize trainer and dummy components before each test case."""
        np.random.seed(42)
        torch.manual_seed(42)

        self.obs_shape = (2, 9)
        self.action_space_size = 9

        self.env = DummyEnv(obs_shape=self.obs_shape, max_steps=5, outcome_winner=1)
        self.mcts = DummyMCTS(action_space_size=self.action_space_size)
        self.network = DummyNetwork(action_space_size=self.action_space_size)

        self.trainer = MuCaroTrainer(
            env=self.env,
            mcts=self.mcts,
            network=self.network,
            buffer_size=20,
            batch_size=4,
            unroll_steps=3,
            num_simulations=10,
            discount=0.9,
            lr=1e-2,
            temp_threshold=3
        )

    # --------------------------------------------------------------------------
    # Self-Play Tests
    # --------------------------------------------------------------------------

    def test_self_play_game_win_and_alternating_discount(self):
        """Verify reward assignment and alternating zero-sum value discounting on a win."""
        game_data = self.trainer.self_play_game()

        self.network.h_network.eval.assert_called()
        self.network.f_network.eval.assert_called()
        self.network.g_network.eval.assert_called()

        self.assertEqual(len(game_data["observations"]), 5)
        self.assertEqual(len(game_data["actions"]), 5)
        self.assertEqual(game_data["rewards"], [0.0, 0.0, 0.0, 0.0, 1.0])

        # v_t = r_t - discount * v_{t+1}
        v4 = 1.0
        v3 = 0.0 - 0.9 * v4
        v2 = 0.0 - 0.9 * v3
        v1 = 0.0 - 0.9 * v2
        v0 = 0.0 - 0.9 * v1
        expected_values = [v0, v1, v2, v3, v4]

        for actual, expected in zip(game_data["values"], expected_values):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_self_play_game_tie_outcome(self):
        """Verify tie outcome (winner == -1) produces 0.0 rewards and 0.0 values throughout."""
        self.trainer.env = DummyEnv(obs_shape=self.obs_shape, max_steps=4, outcome_winner=-1)
        game_data = self.trainer.self_play_game()

        self.assertEqual(len(game_data["observations"]), 4)
        self.assertEqual(game_data["rewards"], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(game_data["values"], [0.0, 0.0, 0.0, 0.0])

    def test_self_play_game_aborted_first_step(self):
        """Verify immediate game abort when an illegal move occurs on the first step."""
        with patch('numpy.random.choice', return_value=999):
            game_data = self.trainer.self_play_game()

        self.assertEqual(len(game_data["observations"]), 1)
        self.assertEqual(game_data["actions"], [999])
        self.assertEqual(game_data["rewards"], [-1.0])
        self.assertEqual(game_data["values"], [-1.0])

    def test_self_play_game_aborted_mid_game_does_not_backpropagate_penalty(self):
        """
        Verify mid-game illegal move assigns -1.0 only to the offending step
        and leaves prior valid steps at 0.0 (no alternating backup when aborted).
        """
        self.trainer.env = DummyEnv(obs_shape=self.obs_shape, max_steps=10, outcome_winner=1)
        with patch('numpy.random.choice', side_effect=[0, 1, 999]):
            game_data = self.trainer.self_play_game()

        self.assertEqual(len(game_data["observations"]), 3)
        self.assertEqual(game_data["rewards"], [0.0, 0.0, -1.0])
        self.assertEqual(game_data["values"], [0.0, 0.0, -1.0])

    def test_self_play_temperature_threshold_behavior(self):
        """
        Verify action selection uses stochastic sampling when step < temp_threshold
        and deterministic argmax when step >= temp_threshold.
        """
        self.trainer.temp_threshold = 2
        self.trainer.env = DummyEnv(obs_shape=self.obs_shape, max_steps=4, outcome_winner=1)

        with patch('numpy.random.choice', return_value=2) as mock_choice, \
             patch('numpy.argmax', return_value=5) as mock_argmax:
            game_data = self.trainer.self_play_game()

        self.assertEqual(mock_choice.call_count, 2)
        self.assertEqual(mock_argmax.call_count, 2)
        self.assertEqual(game_data["actions"], [2, 2, 5, 5])

    def test_self_play_reraises_unexpected_exception(self):
        """Ensure non-ValueError exceptions from the environment are not swallowed."""
        self.trainer.env = DummyEnv(obs_shape=self.obs_shape, raise_unexpected_error=True)
        with self.assertRaises(RuntimeError):
            self.trainer.self_play_game()

    # --------------------------------------------------------------------------
    # Replay Buffer & Batch Sampling Tests
    # --------------------------------------------------------------------------

    def test_replay_buffer_maxlen_eviction(self):
        """Verify replay buffer enforces maxlen and evicts oldest games."""
        for idx in range(25):
            self.trainer.replay_buffer.append({"game_id": idx})

        self.assertEqual(len(self.trainer.replay_buffer), 20)
        self.assertEqual(self.trainer.replay_buffer[0]["game_id"], 5)
        self.assertEqual(self.trainer.replay_buffer[-1]["game_id"], 24)

    def test_sample_batch_shapes_dtypes_and_absorbing_padding(self):
        """
        Verify batch shapes, dtypes, and absorbing state zero-padding
        when sampled trajectories exceed the end of a game.
        """
        obs0 = torch.zeros(self.obs_shape, dtype=torch.float32)
        obs1 = torch.ones(self.obs_shape, dtype=torch.float32)
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
            self.trainer.replay_buffer.append(short_game)

        with patch('random.randint', return_value=0):
            obs_b, act_b, pol_b, val_b, rew_b = self.trainer.sample_batch()

        self.assertEqual(obs_b.shape, (4, *self.obs_shape))
        self.assertEqual(act_b.shape, (4, 4))
        self.assertEqual(pol_b.shape, (4, 4, self.action_space_size))
        self.assertEqual(val_b.shape, (4, 4))
        self.assertEqual(rew_b.shape, (4, 4))

        self.assertEqual(obs_b.dtype, torch.float32)
        self.assertEqual(act_b.dtype, torch.long)
        self.assertEqual(pol_b.dtype, torch.float32)
        self.assertEqual(val_b.dtype, torch.float32)
        self.assertEqual(rew_b.dtype, torch.float32)

        self.assertTrue(torch.all(act_b[:, 0] == 3))
        self.assertTrue(torch.all(act_b[:, 1] == 7))
        self.assertTrue(torch.allclose(rew_b[:, 1], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(val_b[:, 1], torch.tensor(1.0)))

        self.assertTrue(torch.all(act_b[:, 2:] == 0))
        self.assertTrue(torch.all(pol_b[:, 2:, :] == 0.0))
        self.assertTrue(torch.all(val_b[:, 2:] == 0.0))
        self.assertTrue(torch.all(rew_b[:, 2:] == 0.0))

    # --------------------------------------------------------------------------
    # Training Step Mathematical & Computation Tests
    # --------------------------------------------------------------------------

    def test_train_step_exact_loss_and_reward_k_minus_1_alignment(self):
        """
        Verify exact analytical loss values scaled by 1 / (unroll_steps + 1),
        specifically ensuring reward at unroll step k aligns with target_rewards[:, k - 1].
        """
        self.trainer.unroll_steps = 2
        batch_size = 2
        num_actions = 3

        obs = torch.zeros((batch_size, 2, 9), dtype=torch.float32)
        actions = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        # One-hot target policies across k = 0, 1, 2
        target_policies = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ], dtype=torch.float32)

        # Target values across k = 0, 1, 2
        target_values = torch.tensor([
            [1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ], dtype=torch.float32)

        # Target rewards: k=0 -> 1.0, k=1 -> 0.5, k=2 -> 999.0 (sentinel that must NOT be used)
        target_rewards = torch.tensor([
            [1.0, 0.5, 999.0],
            [1.0, 0.5, 999.0],
        ], dtype=torch.float32)

        # Controlled network outputs: all zeros (connected to dummy_param so backward() succeeds)
        dummy_w = self.network.h_param

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
                self.trainer,
                'sample_batch',
                return_value=(obs, actions, target_policies, target_values, target_rewards)
        ), patch.object(
            self.network, 'initial_inference', side_effect=mock_init
        ), patch.object(
            self.network, 'recurrent_inference', side_effect=mock_rec
        ):
            metrics = self.trainer.train_step()

        # Analytical expectations with fraction = 1 / (unroll_steps + 1) = 1 / 3:
        # 1. Policy: 3 steps (k=0,1,2) of uniform zero logits vs one-hot -> (3 * ln(3)) / 3
        expected_policy_loss = (3.0 * math.log(3.0)) / 3.0

        # 2. Value: (MSE(0, 1.0) + MSE(0, -1.0) + MSE(0, 0.0)) / 3 = (1.0 + 1.0 + 0.0) / 3
        expected_value_loss = 2.0 / 3.0

        # 3. Reward: (MSE(0, target_rewards[:, 0]=1.0) + MSE(0, target_rewards[:, 1]=0.5)) / 3
        #            = (1.0^2 + 0.5^2) / 3 = 1.25 / 3
        expected_reward_loss = 1.25 / 3.0
        expected_total_loss = expected_policy_loss + expected_value_loss + expected_reward_loss

        self.assertAlmostEqual(metrics["policy_loss"], expected_policy_loss, places=5)
        self.assertAlmostEqual(metrics["value_loss"], expected_value_loss, places=5)
        self.assertAlmostEqual(metrics["reward_loss"], expected_reward_loss, places=5)
        self.assertAlmostEqual(metrics["total_loss"], expected_total_loss, places=5)

    def test_train_step_unroll_action_sequence_alignment(self):
        """Verify recurrent_inference receives actions[:, k - 1] in exact sequential order."""
        self.trainer.unroll_steps = 3
        obs = torch.zeros((2, 2, 9), dtype=torch.float32)
        actions = torch.tensor([[1, 4, 7, 0], [2, 5, 8, 0]], dtype=torch.long)
        target_policies = torch.full((2, 4, 9), 1.0 / 9.0, dtype=torch.float32)
        target_values = torch.zeros((2, 4), dtype=torch.float32)
        target_rewards = torch.zeros((2, 4), dtype=torch.float32)

        received_actions = []
        orig_rec = self.network.recurrent_inference

        def spy_recurrent(h, a, training=False):
            received_actions.append(a.clone())
            return orig_rec(h, a, training=training)

        with patch.object(
                self.trainer,
                'sample_batch',
                return_value=(obs, actions, target_policies, target_values, target_rewards)
        ), patch.object(
            self.network, 'recurrent_inference', side_effect=spy_recurrent
        ):
            self.trainer.train_step()

        self.assertEqual(len(received_actions), 3)
        self.assertTrue(torch.equal(received_actions[0], torch.tensor([1, 2], dtype=torch.long)))
        self.assertTrue(torch.equal(received_actions[1], torch.tensor([4, 5], dtype=torch.long)))
        self.assertTrue(torch.equal(received_actions[2], torch.tensor([7, 8], dtype=torch.long)))

    def test_train_step_hidden_state_half_gradient_scaling(self):
        """
        Verify `hidden_state = hidden_state * 0.5 + hidden_state.detach() * 0.5`
        attenuates gradients flowing back to initial_inference by 0.5^k per unroll step.
        """
        self.trainer.unroll_steps = 2
        with torch.no_grad():
            self.network.h_param.copy_(torch.tensor([0.1]))

        obs = torch.zeros((1, 2, 9), dtype=torch.float32)
        actions = torch.zeros((1, 3), dtype=torch.long)
        target_policies = torch.full((1, 3, 9), 1.0 / 9.0, dtype=torch.float32)
        target_values = torch.zeros((1, 3), dtype=torch.float32)
        target_rewards = torch.zeros((1, 3), dtype=torch.float32)

        call_idx = {"step": 0}

        def custom_init(x, training=False):
            # h0 = 1.0 * h_param = 0.1; policy & value have no dependency on h_param
            h0 = torch.ones((1, 1)) * self.network.h_param
            p0 = torch.zeros((1, 9))
            v0 = torch.zeros((1, 1))
            return h0, p0, v0

        def custom_rec(h, a, training=False):
            call_idx["step"] += 1
            # Identity transition so only the 0.5 scaling affects gradient magnitude
            nh = h
            # Isolate reward loss strictly at unroll step k = 2
            r = nh if call_idx["step"] == 2 else torch.zeros((1, 1))
            p = torch.zeros((1, 9))
            v = torch.zeros((1, 1))
            return nh, r, p, v

        with patch.object(
                self.trainer,
                'sample_batch',
                return_value=(obs, actions, target_policies, target_values, target_rewards)
        ), patch.object(
            self.network, 'initial_inference', side_effect=custom_init
        ), patch.object(
            self.network, 'recurrent_inference', side_effect=custom_rec
        ):
            self.trainer.train_step()

        # Analytical gradient derivation with fraction = 1 / (unroll_steps + 1) = 1 / 3:
        # total_loss = (const_policy + const_value + MSE(h2, 0)) / 3
        # d(total_loss)/d(h2) = (2 * h2) / 3 = (2 * 0.1) / 3
        # Two unroll steps apply 0.5 scaling twice: dh2/dh0 = 0.5 * 0.5 = 0.25
        # Expected grad = (2 * 0.1 / 3) * 0.25 = 0.05 / 3.0 (~0.016667)
        expected_grad = (2.0 * 0.1 / 3.0) * 0.25
        self.assertAlmostEqual(self.network.h_param.grad.item(), expected_grad, places=6)

    def test_train_step_gradient_clipping_enforced(self):
        """Verify clip_grad_norm_(max_norm=0.5) caps exploding gradients."""
        obs = torch.ones((4, 2, 9), dtype=torch.float32)
        actions = torch.zeros((4, 4), dtype=torch.long)
        target_policies = torch.full((4, 4, 9), 1.0 / 9.0, dtype=torch.float32)
        # Large target values to induce huge raw gradients >> 0.5
        target_values = torch.full((4, 4), 1000.0, dtype=torch.float32)
        target_rewards = torch.full((4, 4), 1000.0, dtype=torch.float32)

        with patch.object(
            self.trainer,
            'sample_batch',
            return_value=(obs, actions, target_policies, target_values, target_rewards)
        ):
            self.trainer.train_step()

        total_grad_norm = torch.sqrt(
            sum(p.grad.detach().pow(2).sum() for p in self.network.params() if p.grad is not None)
        ).item()

        self.assertLessEqual(total_grad_norm, 0.5 + 1e-5)
        self.assertGreater(total_grad_norm, 0.49)

    def test_train_step_updates_all_networks_with_valid_gradients(self):
        """
        Verify train_step():
        - Sets all sub-networks to train mode.
        - Computes finite scalar losses.
        - Produces non-zero gradients and updates weights across h, g, and f networks.
        """
        game_data = self.trainer.self_play_game()
        for _ in range(8):
            self.trainer.replay_buffer.append(game_data)

        h_before = self.network.h_param.item()
        g_before = self.network.g_param.item()
        f_before = self.network.f_param.item()

        metrics = self.trainer.train_step()

        for key in ("total_loss", "policy_loss", "value_loss", "reward_loss"):
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)
            self.assertFalse(np.isnan(metrics[key]))
            self.assertFalse(np.isinf(metrics[key]))

        # Verify total_loss equals sum of normalized component losses
        self.assertAlmostEqual(
            metrics["total_loss"],
            metrics["policy_loss"] + metrics["value_loss"] + metrics["reward_loss"],
            places=6
        )

        # Verify non-zero gradients flowed through the entire unrolled computation graph
        self.assertIsNotNone(self.network.h_param.grad)
        self.assertIsNotNone(self.network.g_param.grad)
        self.assertIsNotNone(self.network.f_param.grad)
        self.assertNotEqual(self.network.h_param.grad.item(), 0.0)
        self.assertNotEqual(self.network.g_param.grad.item(), 0.0)
        self.assertNotEqual(self.network.f_param.grad.item(), 0.0)

        # Verify optimizer updated all parameters
        self.assertNotEqual(h_before, self.network.h_param.item())
        self.assertNotEqual(g_before, self.network.g_param.item())
        self.assertNotEqual(f_before, self.network.f_param.item())

        # Ensure train mode was enabled
        self.network.h_network.train.assert_called()
        self.network.f_network.train.assert_called()
        self.network.g_network.train.assert_called()

    # --------------------------------------------------------------------------
    # Full Training Loop Tests
    # --------------------------------------------------------------------------

    def test_run_loop_min_buffer_gating_and_history(self):
        """Verify run() defers training until min_buffer_size is reached and logs history."""
        history = self.trainer.run(
            num_iterations=3,
            games_per_iter=2,
            train_steps_per_iter=2,
            min_buffer_size=4
        )

        self.assertEqual(len(self.trainer.replay_buffer), 6)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["iteration"], 2)
        self.assertEqual(history[0]["buffer_size"], 4)
        self.assertEqual(history[1]["iteration"], 3)
        self.assertEqual(history[1]["buffer_size"], 6)
