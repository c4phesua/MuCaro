import torch
from torch import zeros


class Board:
    def __init__(self, size: int = 3, n: int = 3):
        self.current_player = 0
        self.state = zeros((2, size * size), dtype=torch.float32, requires_grad=False)
        self.size = size
        self.n = n

    def render_board(self, printable=False):
        screen = [[0 for j in range(self.size)] for i in range(self.size)]
        c_i = 0
        for i in range(self.size):
            for j in range(self.size):
                screen[i][j] = int(self.state[0][c_i].item()) + int(self.state[1][c_i].item() * 2)
                c_i += 1
        if printable:
            print('-' * (self.size * 4))
            for i in range(self.size):
                print(screen[i])
            print('-' * (self.size * 4))
        return screen

    def play(self, index: int):
        if self.state[0][index].item() != 0.0 or self.state[1][index].item() != 0.0:
            return -1
        self.state[self.current_player][index] = 1
        self.current_player = (self.current_player + 1) % 2
        return self.state.clone()
