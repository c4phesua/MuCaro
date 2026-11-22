import copy
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from model.utils import save_checkpoint


def _create_dummy_mcts():
    """Creates a lightweight mock MCTS object with valid PyTorch sub-network state dicts."""
    network = SimpleNamespace(
        h_network=nn.Linear(18, 32),
        f_network=nn.Linear(32, 9),
        g_network=nn.Linear(32, 32),
    )
    return SimpleNamespace(
        network=network,
        action_space_size=9,
        gamma=0.99,
        pb_c_base=19652.0,
        pb_c_init=1.25,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
    )


@patch("torch.save")
@patch("os.makedirs")
def test_save_checkpoint_structure_and_metadata(mock_makedirs, mock_torch_save):
    """
    Verify that save_checkpoint creates the target directory and passes a complete,
    accurately structured state dictionary with all network states and MCTS hyperparameters.
    """
    mcts = _create_dummy_mcts()
    optimizer = MagicMock()
    optimizer.state_dict.return_value = {"state": {}, "param_groups": []}
    eval_stats = {"win": 13, "tie": 2, "lose": 5, "win_percentage": 0.65}

    save_checkpoint(
        mcts=mcts,
        optimizer=optimizer,
        board_size=(3, 3),
        eval_stats=eval_stats,
        model_name="mucaro_v1",
        iteration=50,
        save_dir="./dummy_checkpoints",
    )

    mock_makedirs.assert_called_once_with("./dummy_checkpoints", exist_ok=True)
    assert mock_torch_save.call_count == 1

    saved_dict, saved_path = mock_torch_save.call_args[0]

    expected_keys = {
        "iteration",
        "h_network_state",
        "f_network_state",
        "g_network_state",
        "optimizer_state",
        "eval_stats",
        "action_space_size",
        "board_size",
        "gamma",
        "pb_c_base",
        "pb_c_init",
        "root_dirichlet_alpha",
        "root_exploration_fraction",
    }
    assert set(saved_dict.keys()) == expected_keys

    # Verify scalar metadata and hyperparameters
    assert saved_dict["iteration"] == 50
    assert saved_dict["optimizer_state"] == {"state": {}, "param_groups": []}
    assert saved_dict["eval_stats"] == eval_stats
    assert saved_dict["action_space_size"] == 9
    assert saved_dict["board_size"] == (3, 3)
    assert saved_dict["gamma"] == 0.99
    assert saved_dict["pb_c_base"] == 19652.0
    assert saved_dict["pb_c_init"] == 1.25
    assert saved_dict["root_dirichlet_alpha"] == 0.3
    assert saved_dict["root_exploration_fraction"] == 0.25

    # Verify all three sub-network state_dicts are captured
    assert set(saved_dict["h_network_state"].keys()) == {"weight", "bias"}
    assert set(saved_dict["f_network_state"].keys()) == {"weight", "bias"}
    assert set(saved_dict["g_network_state"].keys()) == {"weight", "bias"}

    assert saved_path.replace("\\", "/") == "./dummy_checkpoints/mucaro_v1-50-65pct.pt"


@pytest.mark.parametrize(
    "win_percentage, expected_suffix",
    [
        (0.0, "0pct.pt"),
        (0.5, "50pct.pt"),
        (1.0, "100pct.pt"),
    ],
)
@patch("torch.save")
@patch("os.makedirs")
def test_save_checkpoint_filename_formatting(
        mock_makedirs, mock_torch_save, win_percentage, expected_suffix
):
    """Verify checkpoint filename formats win percentage accurately across boundary values."""
    mcts = _create_dummy_mcts()
    optimizer = MagicMock()
    optimizer.state_dict.return_value = {}
    eval_stats = {"win": 0, "tie": 0, "lose": 0, "win_percentage": win_percentage}

    save_checkpoint(
        mcts=mcts,
        optimizer=optimizer,
        board_size=(15, 15),
        eval_stats=eval_stats,
        model_name="caro15",
        iteration=10,
        save_dir="./out",
    )

    _, saved_path = mock_torch_save.call_args[0]
    assert saved_path.replace("\\", "/") == f"./out/caro15-10-{expected_suffix}"


def test_save_checkpoint_roundtrip_serialization(tmp_path):
    """
    End-to-end test writing a real .pt file to a temporary directory (auto-cleaned by pytest)
    and reloading it with torch.load to verify weight and optimizer state integrity.
    """
    mcts = _create_dummy_mcts()
    params = (
            list(mcts.network.h_network.parameters())
            + list(mcts.network.f_network.parameters())
            + list(mcts.network.g_network.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    eval_stats = {"win": 8, "tie": 1, "lose": 1, "win_percentage": 0.8}

    save_dir = str(tmp_path / "checkpoints")
    save_checkpoint(
        mcts=mcts,
        optimizer=optimizer,
        board_size=(3, 3),
        eval_stats=eval_stats,
        model_name="roundtrip_test",
        iteration=7,
        save_dir=save_dir,
    )

    expected_file = tmp_path / "checkpoints" / "roundtrip_test-7-80pct.pt"
    assert expected_file.exists()

    loaded = torch.load(expected_file, map_location="cpu", weights_only=False)
    assert loaded["iteration"] == 7
    assert loaded["board_size"] == (3, 3)
    assert torch.equal(
        loaded["h_network_state"]["weight"], mcts.network.h_network.weight.cpu()
    )
    assert torch.equal(
        loaded["f_network_state"]["bias"], mcts.network.f_network.bias.cpu()
    )
    assert torch.equal(
        loaded["g_network_state"]["weight"], mcts.network.g_network.weight.cpu()
    )
    assert "param_groups" in loaded["optimizer_state"]


def test_save_checkpoint_does_not_mutate_mcts():
    """
    Verify that save_checkpoint is strictly read-only and does not mutate
    MCTS attributes, sub-network weights, tensor devices/dtypes, or training modes.
    """
    # Build sub-networks in distinct training/eval modes to ensure modes are preserved
    h_net = nn.Sequential(nn.Linear(18, 32), nn.BatchNorm1d(32))
    f_net = nn.Linear(32, 9)
    g_net = nn.Linear(32, 32)

    h_net.train()
    f_net.eval()
    g_net.train()

    network = SimpleNamespace(
        h_network=h_net,
        f_network=f_net,
        g_network=g_net,
    )

    mcts = SimpleNamespace(
        network=network,
        action_space_size=9,
        gamma=0.99,
        pb_c_base=19652.0,
        pb_c_init=1.25,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
    )

    optimizer = MagicMock()
    optimizer.state_dict.return_value = {"state": {}, "param_groups": []}
    eval_stats = {"win": 7, "tie": 1, "lose": 2, "win_percentage": 0.7}

    # Snapshot all scalar attributes, training modes, and cloned parameter tensors before saving
    attrs_before = {
        k: copy.deepcopy(v) for k, v in vars(mcts).items() if k != "network"
    }
    modes_before = {
        "h": mcts.network.h_network.training,
        "f": mcts.network.f_network.training,
        "g": mcts.network.g_network.training,
    }
    weights_before = {
        "h": {k: v.detach().clone() for k, v in mcts.network.h_network.state_dict().items()},
        "f": {k: v.detach().clone() for k, v in mcts.network.f_network.state_dict().items()},
        "g": {k: v.detach().clone() for k, v in mcts.network.g_network.state_dict().items()},
    }

    with patch("os.makedirs"), patch("torch.save"):
        save_checkpoint(
            mcts=mcts,
            optimizer=optimizer,
            board_size=(3, 3),
            eval_stats=eval_stats,
            model_name="immutability_test",
            iteration=15,
            save_dir="./dummy_checkpoints",
        )

    # 1. Verify no attributes were added, removed, or modified on MCTS
    attrs_after = {k: v for k, v in vars(mcts).items() if k != "network"}
    assert attrs_after == attrs_before

    # 2. Verify sub-network train/eval modes remain unchanged
    assert mcts.network.h_network.training == modes_before["h"]
    assert mcts.network.f_network.training == modes_before["f"]
    assert mcts.network.g_network.training == modes_before["g"]

    # 3. Verify all parameter and buffer tensors remain identical in value, dtype, and device
    for net_key, net_module in [
        ("h", mcts.network.h_network),
        ("f", mcts.network.f_network),
        ("g", mcts.network.g_network),
    ]:
        current_state = net_module.state_dict()
        assert set(current_state.keys()) == set(weights_before[net_key].keys())
        for param_name, expected_tensor in weights_before[net_key].items():
            actual_tensor = current_state[param_name]
            assert actual_tensor.device == expected_tensor.device
            assert actual_tensor.dtype == expected_tensor.dtype
            assert torch.equal(actual_tensor, expected_tensor)
