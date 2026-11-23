import argparse
import os
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ['KERAS_BACKEND'] = 'torch'

from model.utils import load_checkpoint
from training.network3x3 import get_mucaro_3x3_network
from environment.caro import Board
import torch

torch.set_default_device('cuda')
SYMBOLS = {0: ".", 1: "X", 2: "O"}


def render_board_2d(env: Board) -> None:
    screen = env.render_board(printable=False)
    size = env.size

    col_header = "    " + " ".join(f"{c:>2}" for c in range(size))
    border = "   +" + "-" * (size * 3)

    print(col_header)
    print(border)
    for r in range(size):
        row_cells = " ".join(f"{SYMBOLS[screen[r][c]]:>2}" for c in range(size))
        print(f"{r:>2} |{row_cells}")
    print()


def get_human_move_2d(env: Board) -> int:
    size = env.size
    legal_mask = env.get_legal_mask()
    while True:
        raw = input(f"Enter move 'row col' (0..{size - 1}): ").strip()
        parts = raw.replace(",", " ").split()

        if len(parts) != 2 or not (parts[0].isdigit() and parts[1].isdigit()):
            print("Invalid format! Please enter two numbers, e.g. '1 2'.")
            continue

        r, c = int(parts[0]), int(parts[1])
        if not (0 <= r < size and 0 <= c < size):
            print(f"Out of bounds! Row and column must be between 0 and {size - 1}.")
            continue

        action = r * size + c
        if not legal_mask[action]:
            print(f"Cell ({r}, {c}) is already occupied! Choose an empty '.' cell.")
            continue

        return action


def main():
    parser = argparse.ArgumentParser(description="Load MuCaro checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--simulation", type=int, default=40, help="Number of simulations for MCTS")
    args = parser.parse_args()

    network = get_mucaro_3x3_network()
    mcts, _, model_infos = load_checkpoint(args.checkpoint, network)
    mcts.network.h_network.eval()
    mcts.network.f_network.eval()
    mcts.network.g_network.eval()

    env = Board(size=model_infos["board_size"], win_length=model_infos["win_length"])
    print(model_infos)

    answer = input("Do you want to go first? (Y/N): ").lower().strip()
    if answer not in ["y", "n", "yes", "no"]:
        raise ValueError("The answer must be yes or no")
    human = 0 if answer in ["yes", "y"] else 1

    done = False
    board_info = env.reset()
    render_board_2d(env)
    state = board_info["next_state"]

    with torch.no_grad():
        while not done:
            valid_mask = env.get_legal_mask()
            if env.current_player == human:
                action = get_human_move_2d(env)
            else:
                policy = mcts.simulate(
                    current_state=state,
                    num_simulations=args.simulation,
                    is_training=False,
                )
                for i in range(len(valid_mask)):
                    if not valid_mask[i]:
                        policy[i] = float("-inf")
                action = int(np.argmax(policy))
                ai_r, ai_c = divmod(action, env.size)
                print(f"AI plays: ({ai_r}, {ai_c})")

            board_info = env.play(action)
            state = board_info["next_state"]
            done = board_info["done"]
            render_board_2d(env)

    if board_info["winner"] == human:
        print("Game Over!! You win")
    elif board_info["winner"] == -1:
        print("Game Over!! Tie")
    else:
        print("Game Over!! You lose")


if __name__ == "__main__":
    main()
