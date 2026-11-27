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
    assert board.get_legal_mask() == [False, True, False, True, True, False, True, True, False]


# --- Helper to quickly construct board states for check_win tests ---
def _make_board(
        size: int,
        m: int,
        p0_coords: list[tuple[int, int]],
        p1_coords: list[tuple[int, int]],
        current_player: int = 0,
) -> Board:
    board = Board(size=size)
    board.m = m
    board.current_player = current_player
    for x, y in p0_coords:
        board.state[0, y * size + x] = 1.0
    for x, y in p1_coords:
        board.state[1, y * size + x] = 1.0
    return board


# ==================== 1. WIN DIRECTIONS & WRAP-AROUND ====================

def test_win_horizontal_middle_move():
    """Win horizontally by placing the finishing stone in the middle (tests both forward/backward)."""
    board = Board(size=3)
    for move in (3, 0, 5, 1):
        res = board.play(move)
        assert res['done'] is False
    res = board.play(4)  # Completes horizontal row 3-4-5
    assert res['done'] is True
    assert res['winner'] == 0


def test_win_anti_diagonal():
    """Win along the anti-diagonal (2-4-6) by placing the finishing stone in the middle."""
    board = Board(size=3)
    for move in (2, 0, 6, 1):
        res = board.play(move)
        assert res['done'] is False
    res = board.play(4)
    assert res['done'] is True
    assert res['winner'] == 0


def test_no_wrap_around_horizontal_3x3():
    """1D array boundary check: indices 1, 2, 3 are consecutive in 1D but span two rows -> NO win."""
    board = Board(size=3)
    for move in (1, 0, 2, 8, 3):
        res = board.play(move)
    assert res['done'] is False
    assert res['winner'] is None


def test_no_wrap_around_diagonal_5x5():
    """Indices 3, 9, 15 step by +6 on a 5x5 1D array but wrap across the right edge -> NO win."""
    board = Board(size=5)
    for move in (3, 0, 9, 1, 15):
        res = board.play(move)
    assert res['done'] is False
    assert res['winner'] is None


# ==================== 2. DRAW & LAST-MOVE WIN ====================

def test_draw_game_3x3():
    """Filling all 9 squares with no winner ends in a draw with no legal moves remaining."""
    board = Board(size=3)
    # Final board:
    # X O X
    # X X O
    # O X O
    moves = [0, 1, 2, 5, 3, 6, 4, 8, 7]
    for move in moves[:-1]:
        res = board.play(move)
        assert res['done'] is False
        assert res['winner'] is None

    res = board.play(moves[-1])
    assert res['done'] is True
    assert res['winner'] in (-1, None)  # Match the draw convention in play()
    assert board.get_legal_mask() == [False] * 9

    with pytest.raises(Exception) as exc_info:
        board.play(0)
    assert str(exc_info.value) == 'Game is done, please reset the board'


def test_win_on_last_move_not_draw():
    """Completing a winning line on the final empty square must register as a WIN, not a DRAW."""
    board = Board(size=3)
    # Before last move:
    # X O X
    # O X O
    # O X .
    moves = [0, 1, 2, 3, 4, 5, 7, 6]
    for move in moves:
        res = board.play(move)
        assert res['done'] is False

    res = board.play(8)  # Completes diagonal 0-4-8 on the last square
    assert res['done'] is True
    assert res['winner'] == 0


# ==================== 3. RESTRICTED CARO RULES (is_restricted) ====================

def test_caro_vn_blocked_both_ends():
    """O X X X X X O: Not a win under restricted Caro rules; win under unrestricted rules."""
    board = _make_board(
        size=9, m=5,
        p0_coords=[(2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
        p1_coords=[(1, 4), (7, 4)],
    )
    for x in (2, 4, 6):  # Test both edge and middle finishing moves
        idx = 4 * 9 + x
        assert board.check_win(idx, is_restricted=True) == 0
        assert board.check_win(idx, is_restricted=False) == 1


def test_caro_vn_blocked_one_end():
    """O X X X X X . : Blocked on only one end -> still a win."""
    board = _make_board(
        size=9, m=5,
        p0_coords=[(2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
        p1_coords=[(1, 4)],
    )
    assert board.check_win(4 * 9 + 6, is_restricted=True) == 1


def test_caro_vn_edge_not_counted_as_block():
    """| X X X X X O (horizontal and corner diagonal): Board edge does NOT count as a block -> win."""
    board_row = _make_board(
        size=9, m=5,
        p0_coords=[(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
        p1_coords=[(5, 4)],
    )
    assert board_row.check_win(4 * 9 + 0, is_restricted=True) == 1
    assert board_row.check_win(4 * 9 + 4, is_restricted=True) == 1

    board_diag = _make_board(
        size=9, m=5,
        p0_coords=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
        p1_coords=[(5, 5)],
    )
    assert board_diag.check_win(4 * 9 + 4, is_restricted=True) == 1


def test_caro_vn_overline_6_stones():
    """6 in a row: Overline does not win under restricted rules (regardless of placement); wins unrestricted."""
    board = _make_board(
        size=9, m=5,
        p0_coords=[(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
        p1_coords=[],
    )
    for x in (1, 3, 6):
        idx = 4 * 9 + x
        assert board.check_win(idx, is_restricted=True) == 0
        assert board.check_win(idx, is_restricted=False) == 1


def test_caro_vn_one_axis_blocked_other_axis_wins():
    """Intersecting move: horizontal line is blocked on both ends, but vertical line is open 5 -> win."""
    p0_horizontal = [(2, 4), (3, 4), (4, 4), (5, 4), (6, 4)]
    p0_vertical = [(4, 2), (4, 3), (4, 5), (4, 6)]
    board = _make_board(
        size=9, m=5,
        p0_coords=p0_horizontal + p0_vertical,
        p1_coords=[(1, 4), (7, 4)],
    )
    assert board.check_win(4 * 9 + 4, is_restricted=True) == 1


# ==================== 4. BOUNDARY ERRORS & STATE RESET ====================

def test_reset_after_player1_wins_restores_player0_turn():
    """After Player 1 wins and reset() is called, Player 0 must move first and all squares must be legal."""
    board = Board(size=5)
    for move in (3, 6, 12, 11, 20, 16):
        res = board.play(move)
    assert res['winner'] == 1

    board.reset()
    assert board.current_player == 0
    assert board.get_legal_mask() == [True] * 25

    board.play(0)
    assert board.state[0][0].item() == 1.0
    assert board.state[1][0].item() == 0.0


@pytest.mark.parametrize("bad_move", [-1, 9, 100])
def test_play_out_of_bounds_raises(bad_move):
    """Playing outside [0, size*size - 1] must raise an error."""
    board = Board(size=3)
    with pytest.raises((ValueError, IndexError)):
        board.play(bad_move)


@pytest.mark.parametrize("bad_move", [-1, 9, 100])
def test_play_out_of_bounds_message(bad_move):
    """Out-of-range moves must raise ValueError with the exact message."""
    board = Board(size=3)
    with pytest.raises(ValueError) as exc_info:
        board.play(bad_move)
    assert str(exc_info.value) == 'Move out of range'


def test_custom_win_length_5x5_win_4():
    """With win_length=4 on a 5x5 board, 3 in a row must not win; 4 in a row must win."""
    board = Board(size=5, win_length=4)
    for move in (0, 5, 1, 6, 2, 7):
        res = board.play(move)
        assert res['done'] is False
        assert res['winner'] is None

    res = board.play(3)  # Completes 0-1-2-3
    assert res['done'] is True
    assert res['winner'] == 0


def test_play_restricted_caro_blocked_both_ends_continues_game():
    """End-to-end play() with is_restricted=True: blocked line does not end the game."""
    board = Board(size=7, win_length=3, is_restricted=True)
    # Build O X X X O on row 3 (indices 22..26): P0 at 23, 24, 25; P1 at 22, 26
    for move in (23, 22, 25, 26):
        res = board.play(move)
        assert res['done'] is False

    res = board.play(24)  # Completes blocked 3-in-a-row -> game must continue
    assert res['done'] is False
    assert res['winner'] is None

    # P0 continues and wins vertically along col 3 (indices 17, 24, 31)
    for move in (0, 17, 1):
        res = board.play(move)
        assert res['done'] is False

    res = board.play(31)
    assert res['done'] is True
    assert res['winner'] == 0


def test_play_restricted_caro_overline_does_not_win():
    """End-to-end play() with is_restricted=True: creating an overline (4 for m=3) must not win."""
    board = Board(size=7, win_length=3, is_restricted=True)
    # P0 places at (1,3)=22, (2,3)=23, (4,3)=25 then fills gap (3,3)=24 -> 4 in a row
    for move in (22, 0, 23, 1, 25, 7):
        res = board.play(move)
        assert res['done'] is False

    res = board.play(24)
    assert res['done'] is False
    assert res['winner'] is None


def test_canonical_state_is_immutable_copy():
    """Returned next_state tensors must not be mutated by later moves or reset() (replay buffer safety)."""
    board = Board(size=3)
    res1 = board.play(0)
    snap1 = res1['next_state'].clone()

    res2 = board.play(4)
    snap2 = res2['next_state'].clone()

    board.play(8)
    board.reset()

    assert torch.equal(res1['next_state'], snap1)
    assert torch.equal(res2['next_state'], snap2)


def test_reset_after_draw_allows_new_game():
    """Resetting after a full-board draw must allow a clean new game starting with Player 0."""
    board = Board(size=3)
    for move in (0, 1, 2, 5, 3, 6, 4, 8):
        board.play(move)
    res = board.play(7)
    assert res['done'] is True
    assert res['winner'] == -1

    reset_res = board.reset()
    assert reset_res['done'] is False
    assert reset_res['winner'] is None
    assert board.current_player == 0
    assert board.get_legal_mask() == [True] * 9

    next_res = board.play(4)
    assert next_res['done'] is False
    assert board.state[0][4].item() == 1.0
