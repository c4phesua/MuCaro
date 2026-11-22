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
    # Ensure MCTS was actually called to fetch the action
    mock_mcts.simulate.assert_called_once()


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