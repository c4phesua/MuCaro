import math
from typing import List, Optional

import numpy as np
import torch

from model.network import MuCaroNetwork
from src.mcts.minmax import MinMaxStats
from src.mcts.node import Node


class MCTS:
    def __init__(self, network: MuCaroNetwork, action_space_size: int):
        self.network: MuCaroNetwork = network
        self.gamma: float = 1.0
        self.pb_c_base: float = 19652.0
        self.pb_c_init: float = 1.25
        self.root_dirichlet_alpha: float = 0.03
        self.root_exploration_fraction: float = 0.25
        self.action_space_size = action_space_size

    def simulate(self, current_state, valid_mask: Optional[list]=None, num_simulations=40, is_training: bool = True):
        if num_simulations <= 0:
            raise ValueError("Number of simulation must greater than 0")
        minmax_norm = MinMaxStats(-1, 1)
        current_state = current_state.unsqueeze(0)
        with torch.inference_mode():
            hidden_state, policy, _ = self.network.initial_inference(current_state)
            if valid_mask is not None:
                valid_mask_tensor = ~torch.tensor(valid_mask, dtype=torch.bool)
                policy = policy.masked_fill(valid_mask_tensor.unsqueeze(0), -torch.inf)
            priors = torch.softmax(policy, dim=-1).squeeze(0).cpu().tolist()
            root = Node(action=-1, action_space_size=self.action_space_size)
            root.hidden_state = hidden_state
            root.visit_count = 1
            for action in range(self.action_space_size):
                if valid_mask is None or valid_mask[action]:
                    root.children[action] = Node(action, self.action_space_size, prior=priors[action])
            if is_training:
                self.add_exploration_noise(root)
            for _ in range(num_simulations):
                current_child: Node = self.select_child(parent=root, min_max_stats=minmax_norm)
                visited_history: List[Node] = [root, current_child]
                while current_child.visit_count > 0:
                    current_child = self.select_child(parent=current_child, min_max_stats=minmax_norm)
                    visited_history.append(current_child)
                parent_state = visited_history[-2].hidden_state
                action_tensor = torch.tensor([current_child.action], dtype=torch.long)
                hidden_state, reward, policy, leaf_value = self.network.recurrent_inference(parent_state, action_tensor)
                current_child.hidden_state = hidden_state
                current_child.reward = reward.squeeze(-1)[0].item()
                priors = torch.softmax(policy, dim=-1).squeeze(0).cpu().tolist()
                for action in range(self.action_space_size):
                    current_child.children[action] = Node(action, self.action_space_size, prior=priors[action])
                update_value = leaf_value.squeeze(-1)[0].item()
                for visited_node in reversed(visited_history):
                    visited_node.visit_count += 1
                    visited_node.value_sum += update_value
                    minmax_norm.update(update_value)
                    update_value = visited_node.reward - self.gamma * update_value
            return self.cal_distribution(root, num_simulations=num_simulations)

    def cal_distribution(self, root, num_simulations):
        distribution = [0] * self.action_space_size
        for child in root.children:
            if child is not None:
                distribution[child.action] = child.visit_count / num_simulations
        return distribution

    def select_child(self, parent: Node, min_max_stats: MinMaxStats):
        scores = self.ucb_scores(parent, min_max_stats)
        max_child_index = np.argmax(scores)
        return parent.children[max_child_index]

    def ucb_scores(self, parent: Node, min_max_stats: MinMaxStats) -> List[float]:
        scores = [float('-inf')] * self.action_space_size
        c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        for i, child in enumerate(parent.children):
            if child is None:
                continue
            if child.visit_count > 0:
                q_value = child.reward + self.gamma * -child.value()
                q_value = min_max_stats.normalize(q_value)
            else:
                q_value = min_max_stats.normalize(0.0)
            prior_value = c * child.prior * (math.sqrt(parent.visit_count) / (1 + child.visit_count))
            scores[i] = q_value + prior_value
        return scores

    def add_exploration_noise(self, root: Node):
        valid_actions = [a for a, child in enumerate(root.children) if child is not None]
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * len(valid_actions))
        frac = self.root_exploration_fraction

        for action, n in zip(valid_actions, noise):
            child = root.children[action]
            child.prior = child.prior * (1.0 - frac) + n * frac
