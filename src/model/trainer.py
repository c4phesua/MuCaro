import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.functional import cross_entropy, mse_loss, cosine_similarity
from torch.nn.utils import clip_grad_norm_

from environment.caro import Board
from mcts.tree import MCTS
from model.network import MuCaroNetwork
from model.utils import save_checkpoint


class MuCaroTrainer:
    def __init__(
            self,
            env,
            mcts,
            network,
            use_valid_mask: bool = True,
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
        self.use_valid_mask = use_valid_mask

        self.replay_buffer: deque = deque(maxlen=buffer_size)

        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.num_simulations = num_simulations
        self.discount = discount
        self.temp_threshold = temp_threshold
        self.optimizer = torch.optim.AdamW(self.network.params(), lr=lr, weight_decay=weight_decay)

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
        obs = self.env.reset()['next_state']
        done: bool = False

        observations, actions, rewards, policies = [], [], [], []
        step = 0
        # flag to check if there is an invalid move was played
        is_game_aborted = False

        while not done:
            pi = self.mcts.simulate(
                current_state=obs,
                num_simulations=self.num_simulations,
                is_training=True,
                valid_mask=self.env.get_legal_mask() if self.use_valid_mask else None
            )
            if step < self.temp_threshold:
                action = int(np.random.choice(len(pi), p=pi))
            else:
                action = int(np.argmax(pi))
            reward = 0
            try:
                result = self.env.play(action)
                next_obs = result['next_state']
                done = result['done']
                if done and result['winner'] == -1:  # tie
                    reward = 0
                elif done:
                    reward = 1
            except ValueError as ex:
                # invalid move penalty
                reward = -1
                done = True
                next_obs = obs
                is_game_aborted = True
            except Exception as ex:
                raise ex
            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            policies.append(pi)

            obs = next_obs
            step += 1
        game_len = len(observations)
        target_values = [0.0] * game_len
        target_values[-1] = rewards[-1]
        if not is_game_aborted:
            for t in reversed(range(game_len - 1)):
                target_values[t] = rewards[t] - self.discount * target_values[t + 1]

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "policies": policies,
            "values": target_values,
        }

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a batch of unroll trajectories from the replay buffer."""
        action_space_size = self.mcts.action_space_size
        obs_batch, actions_batch, policies_batch, values_batch, rewards_batch, con_mask_batch = [], [], [], [], [], []

        sampled_games = random.choices(self.replay_buffer, k=self.batch_size)
        for game in sampled_games:
            game_len = len(game["observations"])
            start_idx = random.randint(0, game_len - 1)

            obs_seq, act_seq, pol_seq, val_seq, rew_seq, mask_seq = [], [], [], [], [], []
            for k in range(self.unroll_steps + 1):
                idx = start_idx + k
                if idx < game_len:
                    obs_seq.append(game["observations"][idx])
                    act_seq.append(game["actions"][idx])
                    pol_seq.append(game["policies"][idx])
                    val_seq.append(game["values"][idx])
                    rew_seq.append(game["rewards"][idx])
                    mask_seq.append(1)
                else:
                    # Absorbing state padding after terminal step
                    obs_seq.append(torch.zeros_like(game["observations"][0]))
                    act_seq.append(0)
                    pol_seq.append(np.zeros(action_space_size, dtype=np.float32))
                    val_seq.append(0.0)
                    rew_seq.append(0.0)
                    mask_seq.append(0)
            obs_batch.append(torch.stack(obs_seq).to(dtype=torch.float32))
            actions_batch.append(act_seq)
            policies_batch.append(pol_seq)
            values_batch.append(val_seq)
            rewards_batch.append(rew_seq)
            con_mask_batch.append(mask_seq)
        return (
            torch.stack(obs_batch).to(dtype=torch.float32),
            torch.tensor(np.array(actions_batch), dtype=torch.long),
            torch.tensor(np.array(policies_batch), dtype=torch.float32),
            torch.tensor(np.array(values_batch), dtype=torch.float32),
            torch.tensor(np.array(rewards_batch), dtype=torch.float32),
            torch.tensor(np.array(con_mask_batch), dtype=torch.float32)
        )

    def train_step(self) -> Dict[str, float]:
        self._set_train_mode()
        obs, actions, target_policies, target_values, target_rewards, consistency_mask = self.sample_batch()
        init_hidden_states, init_pred_policies, init_pred_values = self.network.initial_inference(obs[:, 0],
                                                                                                  training=True)
        policy_loss = cross_entropy(init_pred_policies, target_policies[:, 0])
        value_loss = mse_loss(init_pred_values.squeeze(-1), target_values[:, 0])
        reward_loss = torch.tensor(0.0, dtype=torch.float32)
        consistency_loss = torch.tensor(0.0, dtype=torch.float32)
        hidden_state = init_hidden_states
        for k in range(1, self.unroll_steps + 1):
            hidden_state = hidden_state * 0.5 + hidden_state.detach() * 0.5
            hidden_state, pred_reward, pred_policy, pred_value = self.network.recurrent_inference(
                hidden_state, actions[:, k - 1], training=True
            )
            with torch.no_grad():
                h_hidden_state = self.network.h_network(obs[:, k], training=False)
            cosine_loss = cosine_similarity(hidden_state, h_hidden_state)
            cosine_loss = (-cosine_loss + 1.0) / 2.0
            cosine_loss = cosine_loss * consistency_mask[:, k]
            consistency_loss += cosine_loss.sum()
            policy_loss += cross_entropy(pred_policy, target_policies[:, k])
            value_loss += mse_loss(pred_value.squeeze(-1), target_values[:, k])
            reward_loss += mse_loss(pred_reward.squeeze(-1), target_rewards[:, k - 1])
        policy_loss = policy_loss / (self.unroll_steps + 1)
        value_loss = value_loss / (self.unroll_steps + 1)
        reward_loss = reward_loss / self.unroll_steps
        consistency_loss = consistency_loss / consistency_mask[:, 1:].sum().clamp_min(1)
        total_loss = policy_loss + value_loss + reward_loss + consistency_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(self.network.params(), max_norm=0.5)
        self.optimizer.step()
        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "reward_loss": reward_loss.item(),
            "consistency_loss": consistency_loss.item()
        }

    def run(
            self,
            evaluator,
            checkpoint_path: str,
            model_name: str,
            evaluation_steps: int = 50,
            eval_interval: int = 10,
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
            iter_metrics = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "reward_loss": 0.0,
                            "consistency_loss": 0.0}
            for _ in range(train_steps_per_iter):
                metrics = self.train_step()
                for key in iter_metrics:
                    iter_metrics[key] += metrics[key] / train_steps_per_iter
            iter_metrics['eval'] = None
            if iteration % eval_interval == 0:
                self._set_eval_mode()
                eval_result = evaluator(mcts=self.mcts, env=self.env, step_num=evaluation_steps,
                                        use_valid_mask=self.use_valid_mask)
                iter_metrics['eval'] = eval_result
                save_checkpoint(mcts=self.mcts, optimizer=self.optimizer, board_size=self.env.size,
                                eval_stats=eval_result, model_name=model_name, iteration=iteration,
                                save_dir=checkpoint_path, win_length=self.env.m)

            iter_metrics["iteration"] = iteration
            iter_metrics["buffer_size"] = len(self.replay_buffer)
            history.append(iter_metrics)

            log_str = (
                f"[Iter {iteration:03d}/{num_iterations:03d}] "
                f"Buffer: {len(self.replay_buffer)} | "
                f"Loss: {iter_metrics['total_loss']:.4f} "
                f"(P: {iter_metrics['policy_loss']:.4f}, "
                f"V: {iter_metrics['value_loss']:.4f}, "
                f"R: {iter_metrics['reward_loss']:.4f}, "
                f"C: {iter_metrics['consistency_loss']:.4f})"
            )
            if iter_metrics['eval']:
                e = iter_metrics['eval']
                log_str += f" | Eval [W:{e['win']} T:{e['tie']} L:{e['lose']}] (WinRate: {e['win_percentage']:.1%})"

            print(log_str)

        return history

    def set_lr(self, new_lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
