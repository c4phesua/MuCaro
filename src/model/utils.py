import os

import torch

from mcts.tree import MCTS


def save_checkpoint(mcts: MCTS, optimizer, board_size, eval_stats, model_name, iteration, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'iteration': iteration,
        'h_network_state': mcts.network.h_network.state_dict(),
        'f_network_state': mcts.network.f_network.state_dict(),
        'g_network_state': mcts.network.g_network.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'eval_stats': eval_stats,
        'action_space_size': mcts.action_space_size,
        'board_size': board_size,
        'gamma': mcts.gamma,
        'pb_c_base': mcts.pb_c_base,
        'pb_c_init': mcts.pb_c_init,
        'root_dirichlet_alpha': mcts.root_dirichlet_alpha,
        'root_exploration_fraction': mcts.root_exploration_fraction
    }

    win_pct_formatted = round(eval_stats['win_percentage'] * 100)
    path = os.path.join(save_dir, f"{model_name}-{iteration}-{win_pct_formatted}pct.pt")
    torch.save(checkpoint, path)
