import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.utils import load_checkpoint

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
    mcts, _, model_infos = load_checkpoint(args.checkpoint)
    print(model_infos)
