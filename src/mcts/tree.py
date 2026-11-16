class MCTS:
    def __init__(self, h_network, f_network, g_network, action_space_size: int):
        self.h_network = h_network
        self.f_network = f_network
        self.g_network = g_network
        self.gamma: float = 1.0
        self.pb_c_base: float = 19652.0
        self.pb_c_init: float = 1.25
        self.root_dirichlet_alpha: float | None = None
        self.root_exploration_fraction: float = 0.25
        self.action_space_size = action_space_size

    def simulate(self, current_state, num_simulations=40):
        pass
