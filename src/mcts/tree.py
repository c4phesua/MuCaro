import math
from typing import List, Optional

import numpy as np
import torch
from keras import Model

from src.mcts.minmax import MinMaxStats
from src.mcts.node import Node


class MCTS:
    def __init__(self, h_network, f_network, g_network, action_space_size: int):
        self.h_network: Model = h_network
        self.f_network: Model = f_network
        self.g_network: Model = g_network
        self.gamma: float = 1.0
        self.pb_c_base: float = 19652.0
        self.pb_c_init: float = 1.25
        self.root_dirichlet_alpha: float = 0.03
        self.root_exploration_fraction: float = 0.25
        self.action_space_size = action_space_size

    def simulate(self, current_state, num_simulations=40, is_training: bool = True):
        if num_simulations <= 0:
            raise ValueError("Number of simulation must greater than 0")
        minmax_norm = MinMaxStats(-1, 1)
        with torch.no_grad():
            hidden_state = self.h_network(current_state, training=False)
            policy, _ = self.f_network(hidden_state, training=False)
            priors = torch.softmax(policy, dim=-1).squeeze(0).cpu().tolist()
            root = Node(action=-1, action_space_size=self.action_space_size)
            root.hidden_state = hidden_state
            root.visit_count = 1
            for action in range(self.action_space_size):
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
                hidden_state, reward = self.g_network([parent_state, action_tensor], training=False)
                policy, leaf_value = self.f_network(hidden_state, training=False)
                current_child.hidden_state = hidden_state
                current_child.reward = reward.item()
                priors = torch.softmax(policy, dim=-1).squeeze(0).cpu().tolist()
                for action in range(self.action_space_size):
                    current_child.children[action] = Node(action, self.action_space_size, prior=priors[action])
                update_value = leaf_value.item()
                for visited_node in reversed(visited_history):
                    visited_node.visit_count += 1
                    visited_node.value_sum += update_value
                    minmax_norm.update(update_value)
                    update_value = visited_node.reward - self.gamma * update_value
            return self.cal_distribution(root, num_simulations=num_simulations)

    def cal_distribution(self, root, num_simulations):
        distribution = [0] * self.action_space_size
        for child in root.children:
            distribution[child.action] = child.visit_count / num_simulations
        return distribution

    def select_child(self, parent: Node, min_max_stats: MinMaxStats):
        scores = self.ucb_scores(parent, min_max_stats)
        max_child_index = np.argmax(scores)
        return parent.children[max_child_index]

    def ucb_scores(self, parent: Node, min_max_stats: MinMaxStats) -> List[Optional[float]]:
        scores = [float('-inf')] * self.action_space_size
        c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        for i, child in enumerate(parent.children):
            if child is None:
                continue
            if child.visit_count > 0:
                q_value = child.reward + self.gamma * -child.value()
                q_value = min_max_stats.normalize(q_value)
            else:
                q_value = 0
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
