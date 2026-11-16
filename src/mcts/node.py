class Node:
    __slots__ = (
        "prior",
        "visit_count",
        "value_sum",
        "reward",
        "hidden_state",
        "children",
    )

    def __init__(self, action_space_size: int, prior: float = 0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.hidden_state = None
        self.children = [None] * action_space_size

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
