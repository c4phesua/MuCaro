import torch

from environment.caro import Board
from environment.eval import random_evaluator
from mcts.tree import MCTS
from model.trainer import MuCaroTrainer
from training.network3x3 import get_mucaro_3x3_network

MODEL_NAME = "3x3"
GAMMA = 0.99
torch.set_default_device('cuda')

if __name__ == '__main__':
    mu_network = get_mucaro_3x3_network()
    mu_network.h_network.summary()
    mu_network.f_network.summary()
    mu_network.g_network.summary()
    mu_network.verify_contract((2, 9), 9)
    checkpoint = torch.load('../../checkpoints/3x3-v4-130-76pct.pt')
    mu_network.h_network.load_state_dict(checkpoint["h_network_state"])
    mu_network.f_network.load_state_dict(checkpoint["f_network_state"])
    mu_network.g_network.load_state_dict(checkpoint["g_network_state"])
    print(checkpoint['eval_stats'])
    env = Board()
    mcts = MCTS(network=mu_network, action_space_size=9)
    mcts.gamma = GAMMA
    mcts.root_dirichlet_alpha = 0.8
    trainer = MuCaroTrainer(env=env, network=mu_network, mcts=mcts, use_valid_mask=True, unroll_steps=5,
                            discount=GAMMA, temp_threshold=2, buffer_size=10000, lr=1e-3)
    # trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
    # trainer.set_lr(2e-4)
    trainer.run(evaluator=random_evaluator, num_iterations=500, train_steps_per_iter=10,
                model_name='3x3-v5', checkpoint_path='../../checkpoints')
