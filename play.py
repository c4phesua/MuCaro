import argparse
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ['KERAS_BACKEND'] = 'torch'

from model.utils import load_checkpoint
from training.network3x3 import get_mucaro_3x3_network
from environment.caro import Board
import torch

torch.set_default_device('cuda')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load MuCaro checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path",
    )
    args = parser.parse_args()
    if args.checkpoint is None:
        raise Exception("checkpoint path is not provided")
    network = get_mucaro_3x3_network()
    mcts, _, model_infos = load_checkpoint(args.checkpoint, network)
    env = Board(size=model_infos["board_size"])
    print(model_infos)
