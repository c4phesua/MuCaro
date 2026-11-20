import torch
from keras import Model, Input, Sequential
from keras.src.layers import Dense, Flatten, LayerNormalization, Concatenate, Embedding

from environment.caro import Board
from mcts.tree import MCTS
from model.network import MuCaroNetwork
from model.trainer import MuCaroTrainer

MODEL_NAME = "3x3"
GAMMA = 0.99
torch.set_default_device('cuda')

mu_network = MuCaroNetwork()
mu_network.h_network = Sequential()
mu_network.h_network.add(Input(shape=(2, 9)))
mu_network.h_network.add(Flatten())
mu_network.h_network.add(Dense(64, activation='silu'))
mu_network.h_network.add(Dense(32, activation='linear'))
mu_network.h_network.add(LayerNormalization())

f_input = Input(shape=(32,))

f_value_layers = Dense(16, activation='silu')(f_input)
f_value_layers = Dense(4, activation='silu')(f_value_layers)
f_value_layers = Dense(1, activation='tanh')(f_value_layers)

f_policy_layers = Dense(16, 'silu')(f_input)
f_policy_layers = Dense(9, 'linear')(f_policy_layers)
mu_network.f_network = Model(inputs=f_input, outputs=[f_policy_layers, f_value_layers])

g_latent_input = Input(shape=(32,))
g_action_input = Input(shape=(1,), dtype=torch.long)
g_action_embedding = Embedding(input_dim=9, output_dim=16)(g_action_input)
g_action_embedding = Flatten()(g_action_embedding)
g_layers = Concatenate()([g_latent_input, g_action_embedding])

g_latent_layers = Dense(64, activation='silu')(g_layers)
g_latent_layers = Dense(32, activation='linear')(g_latent_layers)
g_latent_layers = LayerNormalization()(g_latent_layers)

g_reward_layers = Dense(16, activation='silu')(g_layers)
g_reward_layers = Dense(4, activation='silu')(g_reward_layers)
g_reward_layers = Dense(1, activation='tanh')(g_reward_layers)
mu_network.g_network = Model(inputs=[g_latent_input, g_action_input], outputs=[g_latent_layers, g_reward_layers])

if __name__ == '__main__':
    mu_network.h_network.summary()
    mu_network.f_network.summary()
    mu_network.g_network.summary()
    mu_network.verify_contract((2, 9), 9)
    env = Board()
    mcts = MCTS(network=mu_network, action_space_size=9)
    mcts.gamma = GAMMA
    trainer = MuCaroTrainer(env=env, network=mu_network, mcts=mcts, discount=GAMMA, temp_threshold=2)
    trainer.run()
