import pytest
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
    with pytest.raises(ValueError) as exc_info:
        board.play(8)

    # Verify the exact error message matches what you wrote in the engine
    assert str(exc_info.value) == 'Invalid move'


def test_play_board_invalid_move_2():
    board = Board(size=3)
    board.play(0)
    board.play(5)
    board.play(2)
    with pytest.raises(ValueError) as exc_info:
        board.play(5)

    # Verify the exact error message matches what you wrote in the engine
    assert str(exc_info.value) == 'Invalid move'


def test_play_board_invalid_move_3():
    board = Board(size=3)
    board.play(0)
    board.play(5)
    board.play(2)
    with pytest.raises(ValueError) as exc_info:
        board.play(0)

    # Verify the exact error message matches what you wrote in the engine
    assert str(exc_info.value) == 'Invalid move'


def test_play_board_canonical():
    board = Board(size=3)
    result = board.play(4)
    assert result['next_state'][1][4].item() == 1.0
    result = board.play(0)
    assert result['next_state'][0][4].item() == 1.0
    assert result['next_state'][1][0].item() == 1.0
    result = board.play(8)
    assert result['next_state'][1][4].item() == 1.0
    assert result['next_state'][0][0].item() == 1.0
    assert result['next_state'][1][8].item() == 1.0


def test_play_board_check_win_and_reset():
    board = Board(size=3)
    result = board.play(0)
    assert result['done'] is False
    assert result['winner'] is None
    result = board.play(2)
    assert result['done'] is False
    assert result['winner'] is None
    result = board.play(4)
    assert result['done'] is False
    assert result['winner'] is None
    result = board.play(3)
    assert result['done'] is False
    assert result['winner'] is None
    result = board.play(8)
    assert result['done'] is True
    assert result['winner'] == 0
    expected_state = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 1],
                                   [0, 0, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    assert torch.equal(result['next_state'], expected_state)
    with pytest.raises(Exception) as exc_info:
        board.play(1)

    # Verify the exact error message matches what you wrote in the engine
    assert str(exc_info.value) == 'Game is done, please reset the board'
    result = board.reset()
    assert result['done'] is False
    assert result['winner'] is None
    zeros_state = torch.zeros(2, 9)
    assert torch.equal(result['next_state'], zeros_state)
    assert torch.equal(board.state, zeros_state)


def test_play_check_win_5x5():
    board = Board(size=5)
    result = board.play(3)
    assert result['done'] is False
    assert result['winner'] is None
    result = board.play(6)
    assert result['done'] is False
    assert result['winner'] is None
    result = board.play(12)
    assert result['done'] is False
    assert result['winner'] is None
    result = board.play(11)
    assert result['done'] is False
    assert result['winner'] is None
    result = board.play(20)
    assert result['done'] is False
    assert result['winner'] is None
    result = board.play(16)
    assert result['done'] is True
    assert result['winner'] == 1
    expected_state = torch.tensor([[0, 0, 0, 0, 0,
                                    0, 1, 0, 0, 0,
                                    0, 1, 0, 0, 0,
                                    0, 1, 0, 0, 0,
                                    0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0,
                                    0, 0, 0, 0, 0,
                                    0, 0, 1, 0, 0,
                                    0, 0, 0, 0, 0,
                                    1, 0, 0, 0, 0]], dtype=torch.float32)
    assert torch.equal(result['next_state'], expected_state)


def test_get_legal_moves():
    board = Board(size=3)
    board.play(0)
    board.play(5)
    board.play(2)
    board.play(8)
    assert board.get_legal_moves() == [False, True, False, True, True, False, True, True, False]