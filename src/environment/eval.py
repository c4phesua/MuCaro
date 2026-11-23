import random

import numpy as np
import torch

from environment.caro import Board
from mcts.tree import MCTS


@torch.inference_mode()
def random_evaluator(mcts: MCTS, env: Board, step_num, use_valid_mask: bool = True):
    total_win = 0
    total_tie = 0
    total_lose = 0
    for _ in range(step_num):
        state_info = env.reset()
        current_state = state_info['next_state']
        done = False
        model_play_id = random.randint(0, 1)
        invalid_move_made = False
        while not done:
            if model_play_id == env.current_player:
                policy = mcts.simulate(current_state=current_state, is_training=False,
                                       valid_mask=env.get_legal_mask() if use_valid_mask else None)
                action = np.argmax(policy)
                try:
                    state_info = env.play(action)
                except ValueError as ex:
                    # handle model pick invalid move
                    total_lose += 1
                    invalid_move_made = True
                    break
            else:
                valid_mask = env.get_legal_mask()
                available_actions = [i for i in range(len(valid_mask)) if valid_mask[i]]
                action = random.choice(available_actions)
                state_info = env.play(action)
            current_state = state_info['next_state']
            done = state_info['done']
        if invalid_move_made:
            continue
        if state_info['winner'] == model_play_id:
            total_win += 1
        elif state_info['winner'] == -1:
            total_tie += 1
        else:
            total_lose += 1
    return {'win': total_win, 'lose': total_lose, 'tie': total_tie, 'win_percentage': total_win / step_num}
