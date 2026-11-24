import os
os.environ['KERAS_BACKEND'] = 'torch'

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from environment.eval import random_evaluator


# =======================================================
# PYTEST FIXTURES
# =======================================================
@pytest.fixture
def mock_mcts():
    """Provides a mocked MCTS object that always prefers the center cell (index 4)."""
    mcts = MagicMock()
    mock_policy = np.zeros(9, dtype=np.float32)
    mock_policy[4] = 1.0
    mcts.simulate.return_value = mock_policy
    return mcts


@pytest.fixture
def mock_env():
    """Provides a mocked Environment object with default starting states."""
    env = MagicMock()
    env.reset.return_value = {'next_state': 'dummy_state_0'}
    env.get_legal_mask.return_value = [True] * 9
    return env


# =======================================================
# TEST CASES
# =======================================================
def test_model_win_fast(mock_mcts, mock_env):
    """Game 0 assigns model_play_id=0 (first player); model wins immediately."""
    mock_env.current_player = 0
    mock_env.play.return_value = {
        'next_state': 'dummy_state_1',
        'done': True,
        'winner': 0
    }

    result = random_evaluator(mock_mcts, mock_env, step_num=1)

    assert result == {
        'win': 1,
        'lose': 0,
        'tie': 0,
        'invalid': 0,
        'win_percentage': 1.0,
    }
    mock_mcts.simulate.assert_called_once_with(
        current_state='dummy_state_0',
        is_training=False,
        valid_mask=[True] * 9
    )


def test_model_tie(mock_mcts, mock_env):
    """Verify tie outcome (winner == -1) increments tie without affecting win/lose/invalid."""
    mock_env.current_player = 0
    mock_env.play.return_value = {
        'next_state': 'dummy_state_1',
        'done': True,
        'winner': -1
    }

    result = random_evaluator(mock_mcts, mock_env, step_num=1)

    assert result == {
        'win': 0,
        'lose': 0,
        'tie': 1,
        'invalid': 0,
        'win_percentage': 0.0,
    }


def test_model_make_invalid_move_on_first_turn(mock_mcts, mock_env):
    """Verify ValueError on model's move increments both 'lose' and 'invalid'."""
    mock_env.current_player = 0
    mock_env.play.side_effect = ValueError("Move overlaps an existing piece!")

    result = random_evaluator(mock_mcts, mock_env, step_num=1)

    assert result == {
        'win': 0,
        'lose': 1,
        'tie': 0,
        'invalid': 1,
        'win_percentage': 0.0,
    }


@patch('random.choice', return_value=0)
def test_model_make_invalid_move_as_second_player_and_continues(mock_choice, mock_mcts, mock_env):
    """
    Verify invalid move handling when model plays second (game_idx=1, model_play_id=1),
    and confirm evaluation proceeds cleanly across subsequent games.
    """
    game_tracker = {"game": -1}

    def mock_reset():
        game_tracker["game"] += 1
        mock_env.current_player = 0
        return {'next_state': f'init_{game_tracker["game"]}'}

    def mock_play(action):
        g = game_tracker["game"]
        if g == 0:
            # Game 0 (model=0): valid win
            return {'next_state': 's0_end', 'done': True, 'winner': 0}
        elif g == 1:
            # Game 1 (model=1): opponent plays turn 1, then model plays invalid move on turn 2
            if mock_env.current_player == 0:
                mock_env.current_player = 1
                return {'next_state': 's1_mid', 'done': False, 'winner': None}
            raise ValueError("Occupied cell!")
        else:
            # Game 2 (model=0): valid win
            return {'next_state': 's2_end', 'done': True, 'winner': 0}

    mock_env.reset.side_effect = mock_reset
    mock_env.play.side_effect = mock_play

    result = random_evaluator(mock_mcts, mock_env, step_num=3)

    assert result == {
        'win': 2,
        'lose': 1,
        'tie': 0,
        'invalid': 1,
        'win_percentage': pytest.approx(2 / 3),
    }


def test_use_valid_mask_true_forwards_legal_mask(mock_mcts, mock_env):
    """Verify custom legal mask from env is forwarded to mcts.simulate when use_valid_mask=True."""
    custom_mask = [False, True, False, True, True, False, False, True, False]
    mock_env.get_legal_mask.return_value = custom_mask
    mock_env.current_player = 0
    mock_env.play.return_value = {
        'next_state': 'dummy_state_1',
        'done': True,
        'winner': 0
    }

    random_evaluator(mock_mcts, mock_env, step_num=1, use_valid_mask=True)

    mock_env.get_legal_mask.assert_called_once()
    mock_mcts.simulate.assert_called_once_with(
        current_state='dummy_state_0',
        is_training=False,
        valid_mask=custom_mask
    )


def test_use_valid_mask_false_passes_none(mock_mcts, mock_env):
    """Verify valid_mask=None is passed and get_legal_mask() is not queried when use_valid_mask=False."""
    mock_env.current_player = 0
    mock_env.play.return_value = {
        'next_state': 'dummy_state_1',
        'done': True,
        'winner': 0
    }

    random_evaluator(mock_mcts, mock_env, step_num=1, use_valid_mask=False)

    mock_env.get_legal_mask.assert_not_called()
    mock_mcts.simulate.assert_called_once_with(
        current_state='dummy_state_0',
        is_training=False,
        valid_mask=None
    )


@patch('random.choice', return_value=2)
def test_opponent_samples_only_from_legal_actions(mock_choice, mock_mcts, mock_env):
    """Verify random opponent selects strictly from indices where legal mask is True."""
    legal_mask = [False, False, True, False, True, False, False, False, True]
    mock_env.get_legal_mask.return_value = legal_mask
    game_tracker = {"game": -1}

    def mock_reset():
        game_tracker["game"] += 1
        mock_env.current_player = 0
        return {'next_state': 's0'}

    def mock_play(action):
        if game_tracker["game"] == 0:
            return {'next_state': 's_end', 'done': True, 'winner': 0}
        # Game 1 (model=1): opponent moves first (player 0) and ends game
        return {'next_state': 's_end', 'done': True, 'winner': 0}

    mock_env.reset.side_effect = mock_reset
    mock_env.play.side_effect = mock_play

    random_evaluator(mock_mcts, mock_env, step_num=2, use_valid_mask=True)

    mock_choice.assert_called_once_with([2, 4, 8])


@patch('random.randint')
@patch('random.choice', return_value=0)
def test_alternating_player_assignment_without_random_randint(mock_choice, mock_randint, mock_mcts, mock_env):
    """
    Verify model_play_id deterministically alternates via game_idx % 2:
    - Even games (0, 2): model plays first (player 0)
    - Odd games (1, 3): opponent plays first (player 0), model plays second (player 1)
    """
    mcts_turns_per_game = []
    game_tracker = {"game": -1}

    def mock_reset():
        game_tracker["game"] += 1
        mock_env.current_player = 0
        return {'next_state': f'init_{game_tracker["game"]}'}

    def mock_play(action):
        if mock_env.current_player == 0:
            mock_env.current_player = 1
            return {'next_state': 'mid', 'done': False, 'winner': None}
        else:
            return {'next_state': 'end', 'done': True, 'winner': 1}

    def spy_simulate(current_state, is_training, valid_mask=None):
        mcts_turns_per_game.append((game_tracker["game"], mock_env.current_player))
        p = np.zeros(9, dtype=np.float32)
        p[4] = 1.0
        return p

    mock_env.reset.side_effect = mock_reset
    mock_env.play.side_effect = mock_play
    mock_mcts.simulate.side_effect = spy_simulate

    result = random_evaluator(mock_mcts, mock_env, step_num=4, use_valid_mask=True)

    mock_randint.assert_not_called()
    assert mcts_turns_per_game == [
        (0, 0),  # Game 0: model is player 0
        (1, 1),  # Game 1: model is player 1
        (2, 0),  # Game 2: model is player 0
        (3, 1),  # Game 3: model is player 1
    ]
    # Games 0 & 2 end with winner=1 (opponent wins -> 2 losses); Games 1 & 3 end with winner=1 (model wins -> 2 wins)
    assert result == {
        'win': 2,
        'lose': 2,
        'tie': 0,
        'invalid': 0,
        'win_percentage': pytest.approx(0.5),
    }


@patch('random.choice', return_value=0)
def test_multi_game_evaluation_summary(mock_choice, mock_mcts, mock_env):
    """Simulate 4 games with alternating sides: 2 wins, 1 tie, 1 loss."""
    game_tracker = {"game": -1}

    def mock_reset():
        game_tracker["game"] += 1
        mock_env.current_player = 0
        return {'next_state': f'init_{game_tracker["game"]}'}

    def mock_play(action):
        g = game_tracker["game"]
        if g == 0:
            # Game 0 (model=0): model wins immediately
            return {'next_state': 's0', 'done': True, 'winner': 0}
        elif g == 1:
            # Game 1 (model=1): opponent moves first, then model moves and ties
            if mock_env.current_player == 0:
                mock_env.current_player = 1
                return {'next_state': 's1_mid', 'done': False, 'winner': None}
            return {'next_state': 's1_end', 'done': True, 'winner': -1}
        elif g == 2:
            # Game 2 (model=0): model wins immediately
            return {'next_state': 's2', 'done': True, 'winner': 0}
        else:
            # Game 3 (model=1): opponent moves first and wins immediately
            return {'next_state': 's3', 'done': True, 'winner': 0}

    mock_env.reset.side_effect = mock_reset
    mock_env.play.side_effect = mock_play

    result = random_evaluator(mock_mcts, mock_env, step_num=4, use_valid_mask=True)

    assert result == {
        'win': 2,
        'lose': 1,
        'tie': 1,
        'invalid': 0,
        'win_percentage': pytest.approx(0.5)
    }