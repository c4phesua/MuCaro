import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from keras import Input, Model, Sequential
from keras.src.layers import Concatenate, Dense, Embedding, Flatten, LayerNormalization

from mcts.tree import MCTS
from model.network import MuCaroNetwork
from model.utils import save_checkpoint, load_checkpoint


def _create_dummy_mcts():
    """Creates a lightweight mock MCTS object with valid PyTorch sub-network state dicts."""
    network = SimpleNamespace(
        h_network=nn.Linear(18, 32),
        f_network=nn.Linear(32, 9),
        g_network=nn.Linear(32, 32),
        projector_network=nn.Linear(32, 32),
        predictor_network=nn.Linear(32, 32),
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
    accurately structured state dictionary with all 5 sub-network states and MCTS hyperparameters.
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
        win_length=3
    )

    mock_makedirs.assert_called_once_with("./dummy_checkpoints", exist_ok=True)
    assert mock_torch_save.call_count == 1

    saved_dict, saved_path = mock_torch_save.call_args[0]

    expected_keys = {
        "iteration",
        "h_network_state",
        "f_network_state",
        "g_network_state",
        "projector_network_state",
        "predictor_network_state",
        "optimizer_state",
        "eval_stats",
        "action_space_size",
        "board_size",
        "gamma",
        "pb_c_base",
        "pb_c_init",
        "root_dirichlet_alpha",
        "root_exploration_fraction",
        "win_length",
    }
    assert set(saved_dict.keys()) == expected_keys

    # Verify scalar metadata and hyperparameters
    assert saved_dict["iteration"] == 50
    assert saved_dict["optimizer_state"] == {"state": {}, "param_groups": []}
    assert saved_dict["eval_stats"] == eval_stats
    assert saved_dict["action_space_size"] == 9
    assert saved_dict["board_size"] == (3, 3)
    assert saved_dict["win_length"] == 3
    assert saved_dict["gamma"] == 0.99
    assert saved_dict["pb_c_base"] == 19652.0
    assert saved_dict["pb_c_init"] == 1.25
    assert saved_dict["root_dirichlet_alpha"] == 0.3
    assert saved_dict["root_exploration_fraction"] == 0.25

    # Verify all five sub-network state_dicts are captured
    assert set(saved_dict["h_network_state"].keys()) == {"weight", "bias"}
    assert set(saved_dict["f_network_state"].keys()) == {"weight", "bias"}
    assert set(saved_dict["g_network_state"].keys()) == {"weight", "bias"}
    assert set(saved_dict["projector_network_state"].keys()) == {"weight", "bias"}
    assert set(saved_dict["predictor_network_state"].keys()) == {"weight", "bias"}

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
        win_length=3
    )

    _, saved_path = mock_torch_save.call_args[0]
    assert saved_path.replace("\\", "/") == f"./out/caro15-10-{expected_suffix}"


def test_save_checkpoint_roundtrip_serialization(tmp_path):
    """
    End-to-end test writing a real .pt file to a temporary directory (auto-cleaned by pytest)
    and reloading it with torch.load to verify weight, BatchNorm buffer, and optimizer state integrity
    across all 5 sub-networks.
    """
    mcts = _create_dummy_mcts()
    mcts.network.projector_network = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32))
    mcts.network.predictor_network = nn.Sequential(nn.Linear(32, 16), nn.BatchNorm1d(16), nn.Linear(16, 32))

    # Populate BatchNorm running stats with a forward pass
    dummy_x = torch.randn(4, 32)
    _ = mcts.network.predictor_network(mcts.network.projector_network(dummy_x))

    params = (
        list(mcts.network.h_network.parameters())
        + list(mcts.network.f_network.parameters())
        + list(mcts.network.g_network.parameters())
        + list(mcts.network.projector_network.parameters())
        + list(mcts.network.predictor_network.parameters())
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
        win_length=3
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
    assert torch.equal(
        loaded["projector_network_state"]["1.running_mean"],
        mcts.network.projector_network[1].running_mean.cpu(),
    )
    assert torch.equal(
        loaded["predictor_network_state"]["1.running_var"],
        mcts.network.predictor_network[1].running_var.cpu(),
    )
    assert "param_groups" in loaded["optimizer_state"]


def test_save_checkpoint_does_not_mutate_mcts():
    """
    Verify that save_checkpoint is strictly read-only and does not mutate
    MCTS attributes, sub-network weights, tensor devices/dtypes, or training modes
    across all 5 sub-networks (h, f, g, projector, predictor).
    """
    h_net = nn.Sequential(nn.Linear(18, 32), nn.BatchNorm1d(32))
    f_net = nn.Linear(32, 9)
    g_net = nn.Linear(32, 32)
    projector_net = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32))
    predictor_net = nn.Sequential(nn.Linear(32, 16), nn.BatchNorm1d(16), nn.Linear(16, 32))

    h_net.train()
    f_net.eval()
    g_net.train()
    projector_net.eval()
    predictor_net.train()

    network = SimpleNamespace(
        h_network=h_net,
        f_network=f_net,
        g_network=g_net,
        projector_network=projector_net,
        predictor_network=predictor_net,
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
        "projector": mcts.network.projector_network.training,
        "predictor": mcts.network.predictor_network.training,
    }
    weights_before = {
        "h": {k: v.detach().clone() for k, v in mcts.network.h_network.state_dict().items()},
        "f": {k: v.detach().clone() for k, v in mcts.network.f_network.state_dict().items()},
        "g": {k: v.detach().clone() for k, v in mcts.network.g_network.state_dict().items()},
        "projector": {k: v.detach().clone() for k, v in mcts.network.projector_network.state_dict().items()},
        "predictor": {k: v.detach().clone() for k, v in mcts.network.predictor_network.state_dict().items()},
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
            win_length=3
        )

    # 1. Verify no attributes were added, removed, or modified on MCTS
    attrs_after = {k: v for k, v in vars(mcts).items() if k != "network"}
    assert attrs_after == attrs_before

    # 2. Verify sub-network train/eval modes remain unchanged
    assert mcts.network.h_network.training == modes_before["h"]
    assert mcts.network.f_network.training == modes_before["f"]
    assert mcts.network.g_network.training == modes_before["g"]
    assert mcts.network.projector_network.training == modes_before["projector"]
    assert mcts.network.predictor_network.training == modes_before["predictor"]

    # 3. Verify all parameter and buffer tensors remain identical in value, dtype, and device
    for net_key, net_module in [
        ("h", mcts.network.h_network),
        ("f", mcts.network.f_network),
        ("g", mcts.network.g_network),
        ("projector", mcts.network.projector_network),
        ("predictor", mcts.network.predictor_network),
    ]:
        current_state = net_module.state_dict()
        assert set(current_state.keys()) == set(weights_before[net_key].keys())
        for param_name, expected_tensor in weights_before[net_key].items():
            actual_tensor = current_state[param_name]
            assert actual_tensor.device == expected_tensor.device
            assert actual_tensor.dtype == expected_tensor.dtype
            assert torch.equal(actual_tensor, expected_tensor)


def build_test_network(hidden_dim: int = 32) -> MuCaroNetwork:
    """
    Constructs and initializes a 3x3 MuCaroNetwork with explicit layer names
    so state_dict keys remain deterministic across separate instances in Keras 3.
    """
    net = MuCaroNetwork()

    net.h_network = Sequential(
        [
            Input(shape=(2, 9)),
            Flatten(name="h_flat"),
            Dense(64, activation="silu", name="h_dense_1"),
            Dense(hidden_dim, activation="linear", name="h_dense_2"),
            LayerNormalization(name="h_ln"),
        ],
        name="h_network",
    )

    f_input = Input(shape=(hidden_dim,), name="f_in")
    f_val = Dense(16, activation="silu", name="f_val_1")(f_input)
    f_val = Dense(4, activation="silu", name="f_val_2")(f_val)
    f_val = Dense(1, activation="tanh", name="f_val_out")(f_val)

    f_pol = Dense(16, activation="silu", name="f_pol_1")(f_input)
    f_pol = Dense(9, activation="linear", name="f_pol_out")(f_pol)
    net.f_network = Model(inputs=f_input, outputs=[f_pol, f_val], name="f_network")

    g_latent_in = Input(shape=(hidden_dim,), name="g_lat_in")
    g_action_in = Input(shape=(1,), dtype=torch.long, name="g_act_in")
    g_emb = Flatten(name="g_emb_flat")(
        Embedding(input_dim=9, output_dim=16, name="g_emb")(g_action_in)
    )
    g_cat = Concatenate(name="g_cat")([g_latent_in, g_emb])

    g_lat = Dense(64, activation="silu", name="g_lat_1")(g_cat)
    g_lat = Dense(hidden_dim, activation="linear", name="g_lat_2")(g_lat)
    g_lat = LayerNormalization(name="g_ln")(g_lat)

    g_rew = Dense(16, activation="silu", name="g_rew_1")(g_cat)
    g_rew = Dense(4, activation="silu", name="g_rew_2")(g_rew)
    g_rew = Dense(1, activation="tanh", name="g_rew_out")(g_rew)
    net.g_network = Model(
        inputs=[g_latent_in, g_action_in], outputs=[g_lat, g_rew], name="g_network"
    )

    # Trigger Keras 3 lazy weight initialization
    dummy_obs = torch.zeros((1, 2, 9))
    dummy_action = torch.zeros((1, 1), dtype=torch.long)
    latent = net.h_network(dummy_obs)
    net.f_network(latent)
    net.g_network([latent, dummy_action])

    return net


@pytest.fixture
def sample_checkpoint(tmp_path):
    """Creates a serialized checkpoint file with deterministic weights and optimizer state."""
    source_net = build_test_network()

    with torch.no_grad():
        for i, p in enumerate(source_net.h_network.parameters()):
            p.fill_(0.01 * (i + 1))
        for i, p in enumerate(source_net.f_network.parameters()):
            p.fill_(-0.02 * (i + 1))
        for i, p in enumerate(source_net.g_network.parameters()):
            p.fill_(0.03 * (i + 1))

    all_params = (
        list(source_net.h_network.parameters())
        + list(source_net.f_network.parameters())
        + list(source_net.g_network.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=2e-4)

    # Perform one dummy step so optimizer state buffers are populated
    loss = sum(p.sum() for p in all_params)
    loss.backward()
    optimizer.step()

    ckpt_path = tmp_path / "test_ckpt.pt"
    payload = {
        "action_space_size": 9,
        "h_network_state": source_net.h_network.state_dict(),
        "f_network_state": source_net.f_network.state_dict(),
        "g_network_state": source_net.g_network.state_dict(),
        "gamma": 0.99,
        "pb_c_base": 19652.0,
        "pb_c_init": 1.25,
        "root_dirichlet_alpha": 0.8,
        "root_exploration_fraction": 0.25,
        "optimizer_state": optimizer.state_dict(),
        "board_size": (3, 3),
        "iteration": 200,
        "eval_stats": {"win": 36, "tie": 1, "lose": 13, "win_percentage": 0.72},
        "win_length": 3
    }
    torch.save(payload, ckpt_path)
    return str(ckpt_path), source_net, payload


def test_load_checkpoint_restores_mcts_hyperparameters(sample_checkpoint):
    ckpt_path, _, expected = sample_checkpoint
    target_net = build_test_network()

    mcts, _, _ = load_checkpoint(ckpt_path, network=target_net)

    assert isinstance(mcts, MCTS)
    assert mcts.network is target_net
    assert mcts.action_space_size == expected["action_space_size"]
    assert mcts.gamma == pytest.approx(expected["gamma"])
    assert mcts.pb_c_base == expected["pb_c_base"]
    assert mcts.pb_c_init == pytest.approx(expected["pb_c_init"])
    assert mcts.root_dirichlet_alpha == pytest.approx(expected["root_dirichlet_alpha"])
    assert mcts.root_exploration_fraction == pytest.approx(
        expected["root_exploration_fraction"]
    )


def test_load_checkpoint_restores_network_weights_and_outputs(sample_checkpoint):
    ckpt_path, source_net, _ = sample_checkpoint
    target_net = build_test_network()

    mcts, _, _ = load_checkpoint(ckpt_path, network=target_net)
    loaded_net = mcts.network

    # 1. Verify exact state_dict parameter parity across all three sub-networks
    for subnet_name in ("h_network", "f_network", "g_network"):
        src_state = getattr(source_net, subnet_name).state_dict()
        dst_state = getattr(loaded_net, subnet_name).state_dict()

        assert src_state.keys() == dst_state.keys()
        for key in src_state:
            assert torch.equal(
                src_state[key], dst_state[key]
            ), f"Weight mismatch in {subnet_name}.{key}"

    # 2. Verify forward-pass behavioral equivalence
    obs = torch.randn((2, 2, 9))
    action = torch.tensor([[0], [4]], dtype=torch.long)

    with torch.no_grad():
        src_latent = source_net.h_network(obs)
        dst_latent = loaded_net.h_network(obs)
        assert torch.allclose(src_latent, dst_latent, atol=1e-6)

        src_pol, src_val = source_net.f_network(src_latent)
        dst_pol, dst_val = loaded_net.f_network(dst_latent)
        assert torch.allclose(src_pol, dst_pol, atol=1e-6)
        assert torch.allclose(src_val, dst_val, atol=1e-6)

        src_next_lat, src_rew = source_net.g_network([src_latent, action])
        dst_next_lat, dst_rew = loaded_net.g_network([dst_latent, action])
        assert torch.allclose(src_next_lat, dst_next_lat, atol=1e-6)
        assert torch.allclose(src_rew, dst_rew, atol=1e-6)


def test_load_checkpoint_restores_optimizer_and_metadata(sample_checkpoint):
    ckpt_path, _, expected = sample_checkpoint
    target_net = build_test_network()

    mcts, optimizer_state, metadata = load_checkpoint(ckpt_path, network=target_net)

    assert metadata == {
        "board_size": expected["board_size"],
        "iteration": expected["iteration"],
        "eval_stats": expected["eval_stats"],
        "win_length": 3
    }

    all_params = (
        list(mcts.network.h_network.parameters())
        + list(mcts.network.f_network.parameters())
        + list(mcts.network.g_network.parameters())
    )
    restored_opt = torch.optim.AdamW(all_params, lr=1e-3)
    restored_opt.load_state_dict(optimizer_state)

    assert restored_opt.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert len(restored_opt.state) > 0


def test_load_checkpoint_file_not_found():
    target_net = build_test_network()

    with pytest.raises(FileNotFoundError):
        load_checkpoint("non_existent_checkpoint_path.pt", network=target_net)


def test_load_checkpoint_missing_required_key_raises_key_error(tmp_path):
    target_net = build_test_network()
    bad_ckpt_path = tmp_path / "incomplete_ckpt.pt"
    torch.save({"action_space_size": 9}, bad_ckpt_path)

    with pytest.raises(KeyError):
        load_checkpoint(str(bad_ckpt_path), network=target_net)


def test_load_checkpoint_mismatched_architecture_raises_runtime_error(sample_checkpoint):
    ckpt_path, _, _ = sample_checkpoint
    wrong_shape_net = build_test_network(hidden_dim=16)

    with pytest.raises(RuntimeError):
        load_checkpoint(ckpt_path, network=wrong_shape_net)