import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from environment.caro import Board
from mcts.tree import MCTS
from model.network import MuCaroNetwork


class MuCaroTrainer:
    def __init__(
            self,
            env,
            mcts,
            network,
            buffer_size: int = 10_000,
            batch_size: int = 64,
            unroll_steps: int = 5,
            num_simulations: int = 40,
            discount: float = 1.0,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            temp_threshold: int = 10,
    ):
        self.env: Board = env
        self.mcts: MCTS = mcts
        self.network: MuCaroNetwork = network

        # Replay buffer storing full game trajectories
        self.replay_buffer: deque = deque(maxlen=buffer_size)

        # Hyperparameters
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.num_simulations = num_simulations
        self.discount = discount
        self.temp_threshold = temp_threshold

        # Collect parameters from all three sub-networks (h, f, g) into a single optimizer
        params = (
                list(self.network.h_network.parameters())
                + list(self.network.f_network.parameters())
                + list(self.network.g_network.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def _set_eval_mode(self):
        self.network.h_network.eval()
        self.network.f_network.eval()
        self.network.g_network.eval()

    def _set_train_mode(self):
        self.network.h_network.train()
        self.network.f_network.train()
        self.network.g_network.train()

    def self_play_game(self) -> Dict[str, list]:
        self._set_eval_mode()
        obs = self.env.reset()
        done = False

        observations, actions, rewards, policies = [], [], [], []
        step = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            pi = self.mcts.simulate(
                current_state=obs_tensor,
                num_simulations=self.num_simulations,
                is_training=True,
            )

            # Mask out illegal moves and re-normalize the policy distribution
            legal_mask = np.asarray(self.env.get_legal_mask(), dtype=np.float32)
            masked_pi = np.asarray(pi, dtype=np.float32) * legal_mask
            if masked_pi.sum() > 0:
                masked_pi /= masked_pi.sum()
            else:
                masked_pi = legal_mask / legal_mask.sum()

            # Sample probabilistically during opening moves; act greedily afterwards
            if step < self.temp_threshold:
                action = int(np.random.choice(len(masked_pi), p=masked_pi))
            else:
                action = int(np.argmax(masked_pi))

            next_obs, reward, done = self.env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            policies.append(masked_pi)

            obs = next_obs
            step += 1

        # Compute target values (z_t) backwards with alternating signs for 2-player zero-sum game
        game_len = len(observations)
        target_values = [0.0] * game_len
        running_value = 0.0
        for t in reversed(range(game_len)):
            running_value = rewards[t] - self.discount * running_value
            target_values[t] = running_value

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "policies": policies,
            "values": target_values,
        }

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a batch of unroll trajectories from the replay buffer."""
        action_space_size = self.mcts.action_space_size
        obs_batch, actions_batch, policies_batch, values_batch, rewards_batch = [], [], [], [], []

        sampled_games = random.choices(self.replay_buffer, k=self.batch_size)
        for game in sampled_games:
            game_len = len(game["observations"])
            start_idx = random.randint(0, game_len - 1)

            obs_batch.append(game["observations"][start_idx])

            act_seq, pol_seq, val_seq, rew_seq = [], [], [], []
            for k in range(self.unroll_steps + 1):
                idx = start_idx + k
                if idx < game_len:
                    act_seq.append(game["actions"][idx])
                    pol_seq.append(game["policies"][idx])
                    val_seq.append(game["values"][idx])
                    rew_seq.append(game["rewards"][idx])
                else:
                    # Absorbing state padding after terminal step
                    act_seq.append(0)
                    pol_seq.append(np.zeros(action_space_size, dtype=np.float32))
                    val_seq.append(0.0)
                    rew_seq.append(0.0)

            actions_batch.append(act_seq)
            policies_batch.append(pol_seq)
            values_batch.append(val_seq)
            rewards_batch.append(rew_seq)

        return (
            torch.tensor(np.array(obs_batch), dtype=torch.float32),
            torch.tensor(np.array(actions_batch), dtype=torch.long),
            torch.tensor(np.array(policies_batch), dtype=torch.float32),
            torch.tensor(np.array(values_batch), dtype=torch.float32),
            torch.tensor(np.array(rewards_batch), dtype=torch.float32),
        )

    def train_step(self) -> Dict[str, float]:
        """Performs one K-step unroll and updates network weights."""
        self._set_train_mode()
        obs, actions, target_policies, target_values, target_rewards = self.sample_batch()

        # Step k = 0: Initial inference from real board observation
        hidden_state, pred_logits, pred_value = self.network.initial_inference(obs, training=True)

        policy_loss = -(target_policies[:, 0] * F.log_softmax(pred_logits, dim=-1)).sum(dim=-1).mean()
        value_loss = F.mse_loss(pred_value.squeeze(-1), target_values[:, 0])
        reward_loss = torch.tensor(0.0)

        # Steps k = 1 ... unroll_steps: Recurrent inference in latent space
        for k in range(1, self.unroll_steps + 1):
            # Scale hidden state gradient by 0.5 at each unroll step
            hidden_state = hidden_state * 0.5 + hidden_state.detach() * 0.5

            action_k = actions[:, k - 1]
            hidden_state, pred_reward, pred_logits, pred_value = self.network.recurrent_inference(
                hidden_state, action_k, training=True
            )

            policy_loss = policy_loss - (
                    target_policies[:, k] * F.log_softmax(pred_logits, dim=-1)
            ).sum(dim=-1).mean()
            value_loss = value_loss + F.mse_loss(pred_value.squeeze(-1), target_values[:, k])
            reward_loss = reward_loss + F.mse_loss(pred_reward.squeeze(-1), target_rewards[:, k - 1])

        scale = 1.0 / self.unroll_steps
        total_loss = (policy_loss + value_loss + reward_loss) * scale

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]], max_norm=5.0
        )
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": (policy_loss * scale).item(),
            "value_loss": (value_loss * scale).item(),
            "reward_loss": (reward_loss * scale).item(),
        }

    def run(
            self,
            num_iterations: int = 100,
            games_per_iter: int = 10,
            train_steps_per_iter: int = 20,
            min_buffer_size: int = 20,
    ) -> List[Dict[str, float]]:
        """Main training loop: Self-Play -> Replay Buffer -> Gradient Updates."""
        history = []

        for iteration in range(1, num_iterations + 1):
            # 1. Self-play data collection
            for _ in range(games_per_iter):
                game_data = self.self_play_game()
                self.replay_buffer.append(game_data)

            if len(self.replay_buffer) < min_buffer_size:
                continue

            # 2. Train networks on sampled trajectories
            iter_metrics = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "reward_loss": 0.0}
            for _ in range(train_steps_per_iter):
                metrics = self.train_step()
                for key in iter_metrics:
                    iter_metrics[key] += metrics[key] / train_steps_per_iter

            iter_metrics["iteration"] = iteration
            iter_metrics["buffer_size"] = len(self.replay_buffer)
            history.append(iter_metrics)

            print(
                f"[Iter {iteration:03d}/{num_iterations:03d}] "
                f"Buffer: {len(self.replay_buffer)} | "
                f"Loss: {iter_metrics['total_loss']:.4f} "
                f"(P: {iter_metrics['policy_loss']:.4f}, "
                f"V: {iter_metrics['value_loss']:.4f}, "
                f"R: {iter_metrics['reward_loss']:.4f})"
            )

        return history
