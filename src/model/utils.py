import os

import torch


def save_checkpoint(network, optimizer, action_space_size, board_size, eval_stats, model_name, iteration, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'iteration': iteration,
        'h_network_state': network.h_network.state_dict(),
        'f_network_state': network.f_network.state_dict(),
        'g_network_state': network.g_network.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'eval_stats': eval_stats,
        'action_space_size': action_space_size,
        'board_size': board_size
    }

    win_pct_formatted = int(eval_stats['win_percentage'] * 100)
    path = os.path.join(save_dir, f"{model_name}-{iteration}-{win_pct_formatted}pct.pt")
    torch.save(checkpoint, path)
