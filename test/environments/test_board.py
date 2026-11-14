import torch

from src.environment.caro import Board


def test_play_board():
    board = Board(size=3)
    board.play(0)
    board.play(5)
    board.play(2)
    board.play(8)
    expected_state = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0, 1]], dtype=torch.float32)
    assert torch.equal(board.state, expected_state)


def test_render_board():
    board = Board(size=3)
    board.play(0)
    board.play(5)
    board.play(2)
    board.play(8)
    board.render_board(printable=True)
    expected_board = [[1, 0, 1], [0, 0, 2], [0, 0, 2]]
    assert board.render_board() == expected_board


def test_play_board_invalid_move_1():
    board = Board(size=3)
    board.play(0)
    board.play(5)
    board.play(2)
    board.play(8)
    assert board.play(8) == -1


def test_play_board_invalid_move_2():
    board = Board(size=3)
    board.play(0)
    board.play(5)
    board.play(2)
    assert board.play(5) == -1


def test_play_board_invalid_move_3():
    board = Board(size=3)
    board.play(0)
    board.play(5)
    board.play(2)
    assert board.play(0) == -1
