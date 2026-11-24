import torch
from keras import Sequential, Input
from keras.layers import Dense, BatchNormalization, ReLU


class MuCaroNetwork:
    def __init__(self):
        self.h_network = None
        self.f_network = None
        self.g_network = None
        self.projector_network = None
        self.predictor_network = None

    def build_consistency_networks(self, hidden_size: int):
        self.projector_network = Sequential(name="Projector")
        self.projector_network.add(Input(shape=(hidden_size,)))
        self.projector_network.add(Dense(hidden_size, use_bias=False))
        self.projector_network.add(BatchNormalization())
        self.projector_network.add(ReLU())
        self.projector_network.add(Dense(hidden_size, use_bias=False))
        self.projector_network.add(BatchNormalization())
        self.projector_network.summary()

        predictor_bottleneck = hidden_size // 4 if hidden_size > 64 else hidden_size // 2
        self.predictor_network = Sequential(name="Predictor")
        self.predictor_network.add(Input(shape=(hidden_size,)))
        self.predictor_network.add(Dense(predictor_bottleneck, use_bias=False, activation='linear'))
        self.predictor_network.add(BatchNormalization())
        self.predictor_network.add(ReLU())
        self.predictor_network.add(Dense(hidden_size, activation='linear'))
        self.predictor_network.summary()

    def initial_inference(self, observation, training: bool = False):
        hidden_state = self.h_network(observation, training=training)
        policy_logits, value = self.f_network(hidden_state, training=training)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state, action, training: bool = False):
        next_hidden, reward = self.g_network([hidden_state, action], training=training)
        policy_logits, value = self.f_network(next_hidden, training=training)
        return next_hidden, reward, policy_logits, value

    @torch.no_grad()
    def consistency_project(self, observation):
        hidden_state = self.h_network(observation, training=False)
        return self.projector_network(hidden_state, training=False)

    def consistency_predict(self, hidden_state, training: bool = True):
        projector_state = self.projector_network(hidden_state, training=training)
        return self.predictor_network(projector_state, training=training)

    def params(self):
        return (
                list(self.h_network.parameters())
                + list(self.f_network.parameters())
                + list(self.g_network.parameters())
                + list(self.projector_network.parameters())
                + list(self.predictor_network.parameters())
        )

    @torch.no_grad()
    def verify_contract(self, obs_shape: tuple, action_space_size: int, device="cpu"):
        dummy_obs = torch.zeros((2, *obs_shape), dtype=torch.float32, device=device)
        dummy_act = torch.zeros((2,), dtype=torch.long, device=device)

        h0, p0, v0 = self.initial_inference(dummy_obs, training=False)
        h1, r1, p1, v1 = self.recurrent_inference(h0, dummy_act, training=False)

        assert h0.shape == h1.shape, f"Shape hidden_state difference h {h0.shape} and g {h1.shape}"
        assert p0.shape == (2, action_space_size), f"Policy shape expected (B, {action_space_size}), actual {p0.shape}"
        assert v0.shape == (2, 1), f"Value shape expected (B, 1), actual {v0.shape}"
        assert r1.shape == (2, 1), f"Reward shape expected (B, 1), actual {r1.shape}"

        z0 = self.consistency_project(dummy_obs)
        pred0 = self.consistency_predict(z0, training=False)
        assert z0.shape == h0.shape, f"Projector shape expected {h0.shape}, actual {z0.shape}"
        assert pred0.shape == h0.shape, f"Predictor shape expected {h0.shape}, actual {pred0.shape}"
