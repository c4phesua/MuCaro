import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
import torch.nn as nn

from model.trainer import MuCaroTrainer


# ==============================================================================
# 1. DUMMY CLASSES TO MOCK ENVIRONMENT, MCTS, AND NETWORK
# ==============================================================================

class DummyEnv:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 5

    def reset(self):
        self.step_count = 0
        # obs shape: (Channels, H, W). Assuming Caro 15x15, 3 channels
        return {'next_state': np.zeros((3, 15, 15), dtype=np.float32)}

    def play(self, action):
        if action == 999:
            # Trigger invalid move simulation
            raise ValueError("Invalid move")

        self.step_count += 1
        done = self.step_count >= self.max_steps
        winner = 1 if done else 0

        return {
            'next_state': np.ones((3, 15, 15), dtype=np.float32),
            'done': done,
            'winner': winner
        }


class DummyMCTS:
    def __init__(self, action_space_size=225):
        self.action_space_size = action_space_size

    def simulate(self, current_state, num_simulations, is_training):
        # Return a valid probability distribution (sum = 1)
        pi = np.ones(self.action_space_size, dtype=np.float32)
        pi /= np.sum(pi)
        return pi


class DummyNetwork(nn.Module):
    """
    Dummy Network inherits from nn.Module so PyTorch can compute Gradients.
    """

    def __init__(self, action_space_size=225):
        super().__init__()
        # Create a dummy parameter to build the computation graph and allow backward()
        self.dummy_param = nn.Parameter(torch.tensor([1.0], requires_grad=True))

        self.h_network = MagicMock()
        self.f_network = MagicMock()
        self.g_network = MagicMock()
        self.action_space_size = action_space_size

    def params(self):
        return self.parameters()

    def initial_inference(self, obs, training=False):
        batch_size = obs.shape[0]
        # Multiply by dummy_param to connect to the computation graph
        hidden_state = torch.zeros((batch_size, 64)) * self.dummy_param
        pred_policies = torch.zeros((batch_size, self.action_space_size)) * self.dummy_param
        pred_values = torch.zeros((batch_size, 1)) * self.dummy_param
        return hidden_state, pred_policies, pred_values

    def recurrent_inference(self, hidden_state, action, training=False):
        batch_size = hidden_state.shape[0]
        next_hidden = hidden_state * self.dummy_param
        pred_reward = torch.zeros((batch_size, 1)) * self.dummy_param
        pred_policies = torch.zeros((batch_size, self.action_space_size)) * self.dummy_param
        pred_values = torch.zeros((batch_size, 1)) * self.dummy_param
        return next_hidden, pred_reward, pred_policies, pred_values


# ==============================================================================
# 2. UNIT TESTS
# ==============================================================================

class TestMuCaroTrainer(unittest.TestCase):
    def setUp(self):
        """Initialize trainer and dummy components before each test case."""
        # Fix random seed for deterministic testing
        np.random.seed(42)
        torch.manual_seed(42)

        self.env = DummyEnv()
        self.mcts = DummyMCTS(action_space_size=225)
        self.network = DummyNetwork(action_space_size=225)

        self.trainer = MuCaroTrainer(
            env=self.env,
            mcts=self.mcts,
            network=self.network,
            buffer_size=100,
            batch_size=4,
            unroll_steps=3,
            num_simulations=10,
            discount=0.9,
            lr=1e-3
        )

    def test_self_play_game_normal(self):
        """Test a normal game, determining win/loss/tie."""
        game_data = self.trainer.self_play_game()

        # Dummy environment will finish after 5 steps
        self.assertEqual(len(game_data["observations"]), 5)
        self.assertEqual(len(game_data["actions"]), 5)
        self.assertEqual(len(game_data["rewards"]), 5)
        self.assertEqual(len(game_data["policies"]), 5)
        self.assertEqual(len(game_data["values"]), 5)

        # Check the logic for backward target_values calculation (discounting)
        # Dummy returns reward = 1 at the last step, previous steps are 0
        self.assertEqual(game_data["rewards"][-1], 1.0)
        self.assertEqual(game_data["values"][-1], 1.0)
        self.assertAlmostEqual(game_data["values"][-2], 0.0 - 0.9 * 1.0)  # r - discount * next_v

    def test_self_play_game_aborted(self):
        """Test aborted game due to ValueError (invalid move)."""
        # Force MCTS (or action selection) to always choose action = 999 to trigger ValueError
        with unittest.mock.patch('numpy.argmax', return_value=999), \
                unittest.mock.patch('numpy.random.choice', return_value=999):
            game_data = self.trainer.self_play_game()

        # The game will end immediately at the first step
        self.assertEqual(len(game_data["observations"]), 1)
        self.assertEqual(game_data["rewards"][-1], -1.0)

        # Target value at the end is -1, but because it's aborted, previous steps (if any) remain 0
        self.assertEqual(game_data["values"][-1], -1.0)

    def test_sample_batch_shapes_and_absorbing_states(self):
        """Test data sampling, check padding state when the game is shorter than unroll_steps."""
        # Create a short game_data (length 2) and add it to the buffer
        obs = np.zeros((3, 15, 15))
        pi = np.zeros(225)
        fake_game = {
            "observations": [obs, obs],
            "actions": [10, 20],
            "rewards": [0.0, 1.0],
            "policies": [pi, pi],
            "values": [-0.9, 1.0]
        }
        for _ in range(10):  # Insert 10 samples to ensure enough for batch_size = 4
            self.trainer.replay_buffer.append(fake_game)

        obs_b, act_b, pol_b, val_b, rew_b = self.trainer.sample_batch()

        # Batch_size = 4, unroll_steps = 3 -> seq_len = 4 (step 0 + 3 unroll steps)
        self.assertEqual(obs_b.shape, (4, 3, 15, 15))
        self.assertEqual(act_b.shape, (4, 4))
        self.assertEqual(pol_b.shape, (4, 4, 225))
        self.assertEqual(val_b.shape, (4, 4))
        self.assertEqual(rew_b.shape, (4, 4))

        # Check absorbing state (padding 0) at the last step because fake_game length is only 2
        # Depending on the random start_idx, column k=3 (index 3) will definitely be padded with 0
        self.assertTrue(torch.all(act_b[:, 3] == 0))
        self.assertTrue(torch.all(val_b[:, 3] == 0.0))

    def test_train_step_execution(self):
        """Test BPTT training process (Forward, Loss, Backward, Step)."""
        # Create dummy data
        fake_game = self.trainer.self_play_game()
        for _ in range(10):
            self.trainer.replay_buffer.append(fake_game)

        # Record parameter state before training
        old_param_val = self.trainer.network.dummy_param.item()

        # Run 1 training step
        metrics = self.trainer.train_step()

        # Check returned results
        self.assertIn("total_loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("reward_loss", metrics)

        # Check Gradient Update: dummy parameter must change after optimizer.step()
        new_param_val = self.trainer.network.dummy_param.item()
        self.assertNotEqual(old_param_val, new_param_val)

        # Ensure network is switched to train mode
        self.trainer.network.h_network.train.assert_called()

    def test_run_loop(self):
        """Test the entire Self-Play -> Train pipeline runs smoothly."""
        history = self.trainer.run(
            num_iterations=2,
            games_per_iter=2,
            train_steps_per_iter=2,
            min_buffer_size=2
        )

        self.assertEqual(len(history), 2)
        self.assertIn("total_loss", history[0])
        self.assertEqual(history[0]["iteration"], 1)
        self.assertEqual(history[1]["iteration"], 2)

        # Buffer must accumulate 4 games (2 iter x 2 games)
        self.assertEqual(len(self.trainer.replay_buffer), 4)
