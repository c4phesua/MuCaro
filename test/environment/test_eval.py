import os
os.environ['KERAS_BACKEND'] = 'torch'

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from environment.eval import random_evaluator


# =======================================================
# PYTEST FIXTURES
# =======================================================
@pytest.fixture
def mock_mcts():
    """Provides a mocked MCTS object that always prefers the center cell (index 4)."""
    mcts = MagicMock()
    mock_policy = np.zeros(9)
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
@patch('random.randint', return_value=0)  # Force model to play first (player 0)
def test_model_win_fast(mock_randint, mock_mcts, mock_env):
    # Setup mock behavior: Model plays and wins immediately
    mock_env.current_player = 0
    mock_env.play.return_value = {
        'next_state': 'dummy_state_1',
        'done': True,
        'winner': 0
    }

    result = random_evaluator(mock_mcts, mock_env, step_num=1)

    assert result['win'] == 1
    assert result['win_percentage'] == 1.0
    # Ensure MCTS was called with the legal mask by default (use_valid_mask=True)
    mock_mcts.simulate.assert_called_once_with(
        current_state='dummy_state_0',
        is_training=False,
        valid_mask=[True] * 9
    )


@patch('random.randint', return_value=0)  # Force model to play first (player 0)
def test_model_tie(mock_randint, mock_mcts, mock_env):
    # Setup mock behavior: Game ends in a tie
    mock_env.current_player = 0
    mock_env.play.return_value = {
        'next_state': 'dummy_state_1',
        'done': True,
        'winner': -1
    }

    result = random_evaluator(mock_mcts, mock_env, step_num=1)

    assert result['tie'] == 1
    assert result['win'] == 0
    assert result['win_percentage'] == 0.0


@patch('random.randint', return_value=1)  # Force opponent to play first (player 0)
@patch('random.choice', return_value=0)  # Force opponent to pick cell 0
def test_model_make_invalid_move(mock_choice, mock_randint, mock_mcts, mock_env):
    # Setup mock behavior:
    # Turn 1 (Opponent - player 0): Valid move, game continues
    # Turn 2 (Model - player 1): MCTS picks an invalid move, raising ValueError

    def mock_play(action):
        if mock_env.current_player == 0:
            mock_env.current_player = 1  # Switch turn to the model
            return {'next_state': 'dummy_state_1', 'done': False, 'winner': None}
        else:
            raise ValueError("Move overlaps an existing piece!")

    mock_env.current_player = 0
    mock_env.play.side_effect = mock_play

    result = random_evaluator(mock_mcts, mock_env, step_num=1)

    assert result['lose'] == 1
    assert result['win'] == 0
    assert result['win_percentage'] == 0.0


@patch('random.randint', return_value=0)  # Force model to play first (player 0)
def test_use_valid_mask_true_forwards_legal_mask(mock_randint, mock_mcts, mock_env):
    # Verify custom legal mask from env is forwarded to mcts.simulate when use_valid_mask=True
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


@patch('random.randint', return_value=0)  # Force model to play first (player 0)
def test_use_valid_mask_false_passes_none(mock_randint, mock_mcts, mock_env):
    # Verify valid_mask=None is passed and get_legal_mask() is not queried when use_valid_mask=False
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


@patch('random.randint', side_effect=[0, 0, 1, 0])
@patch('random.choice', return_value=0)
def test_multi_game_evaluation_summary(mock_choice, mock_randint, mock_mcts, mock_env):
    # Simulate 4 games: 2 wins, 1 tie, 1 loss (opponent wins)
    outcomes = [
        {'next_state': 's1', 'done': True, 'winner': 0},   # Game 1: model=0 -> win
        {'next_state': 's2', 'done': True, 'winner': -1},  # Game 2: model=0 -> tie
        {'next_state': 's3', 'done': True, 'winner': 0},   # Game 3: model=1, opponent=0 wins -> lose
        {'next_state': 's4', 'done': True, 'winner': 0},   # Game 4: model=0 -> win
    ]
    mock_env.current_player = 0
    mock_env.play.side_effect = outcomes

    result = random_evaluator(mock_mcts, mock_env, step_num=4, use_valid_mask=True)

    assert result == {
        'win': 2,
        'lose': 1,
        'tie': 1,
        'win_percentage': pytest.approx(0.5)
    }