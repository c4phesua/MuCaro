from turtle import done

import torch
from torch import zeros


class Board:
    def __init__(self, size: int = 3, win_length: int = 3):
        self.current_player = 0
        self.state = zeros((2, size * size), dtype=torch.float32, requires_grad=False)
        self.size = size
        self.m = win_length
        self.done = False

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

    def get_canonical_state(self):
        if self.current_player == 1:
            return self.state[[1, 0], :]
        return self.state.clone()

    def reset(self):
        self.current_player = 0
        self.state.zero_()
        self.done = False

    def play(self, index: int):
        # this one will return json like object (next_state, winner, done)
        if self.done:
            raise Exception('Game is done, please reset the board')
        if self.state[0][index].item() != 0.0 or self.state[1][index].item() != 0.0:
            raise ValueError('Invalid move')
        self.state[self.current_player][index] = 1
        win_check = self.check_win(index)
        if win_check == -1:
            self.done = True
            return {'next_state': self.get_canonical_state(), 'winner': -1, 'done': True}
        if win_check == 1:
            self.done = True
            return {'next_state': self.get_canonical_state(), 'winner': self.current_player, 'done': True}
        self.current_player = (self.current_player + 1) % 2
        return {'next_state': self.get_canonical_state(), 'winner': None, 'done': False}

    def check_win(self, current_index: int):
        check_vectors = [[0, 1], [1, 0], [1, 1], [1, -1]]
        current_x = current_index % self.size
        current_y = current_index // self.size
        player_state = self.state[self.current_player]
        for dx, dy in check_vectors:
            counter = 1
            for step in range(1, self.m):
                check_x = current_x + step * dx
                check_y = current_y + step * dy
                if 0 <= check_x < self.size and 0 <= check_y < self.size:
                    if int(player_state[check_y * self.size + check_x].item()) == 1:
                        counter += 1
                    else:
                        break
                else:
                    break

            for step in range(1, self.m):
                check_x = current_x - step * dx
                check_y = current_y - step * dy
                if 0 <= check_x < self.size and 0 <= check_y < self.size:
                    if int(player_state[check_y * self.size + check_x].item()) == 1:
                        counter += 1
                    else:
                        break
                else:
                    break
            if counter == self.m:
                return 1
        # check for tie
        total_move = self.state.sum().item()
        if total_move == self.size * self.size:
            return -1
        return 0
