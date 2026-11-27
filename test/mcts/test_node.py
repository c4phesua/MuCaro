import pytest
import torch

# Adjust the import path to match your project structure
from mcts.node import Node


def test_node_default_initialization():
    """Verify all default attributes and children list size on creation."""
    node = Node(action=4, action_space_size=9)

    assert node.action == 4
    assert node.prior == 0.0
    assert node.visit_count == 0
    assert node.value_sum == 0.0
    assert node.reward == 0.0
    assert node.hidden_state is None
    assert len(node.children) == 9
    assert all(child is None for child in node.children)


def test_node_custom_prior():
    """Verify explicit prior probability is stored accurately."""
    node = Node(action=2, action_space_size=9, prior=0.35)

    assert node.action == 2
    assert node.prior == pytest.approx(0.35)


def test_node_value_when_unvisited():
    """Unvisited node must return 0.0 without raising ZeroDivisionError."""
    node = Node(action=0, action_space_size=9)
    assert node.value() == 0.0

    # Even if value_sum is non-zero while visit_count is 0
    node.value_sum = 5.0
    assert node.value() == 0.0


@pytest.mark.parametrize(
    "visit_count, value_sum, expected",
    [
        (1, 1.0, 1.0),
        (4, 2.0, 0.5),
        (3, -1.5, -0.5),
        (5, 0.0, 0.0),
    ],
)
def test_node_value_calculation(visit_count, value_sum, expected):
    """Verify mean value calculation across positive, negative, and zero sums."""
    node = Node(action=0, action_space_size=9)
    node.visit_count = visit_count
    node.value_sum = value_sum

    assert node.value() == pytest.approx(expected)


def test_node_children_linking_and_independence():
    """Attaching a child must only update that slot and not leak to other nodes."""
    parent = Node(action=-1, action_space_size=9)
    other_node = Node(action=-1, action_space_size=9)
    child = Node(action=3, action_space_size=9, prior=0.25)

    parent.children[3] = child

    assert parent.children[3] is child
    assert sum(c is not None for c in parent.children) == 1
    assert other_node.children[3] is None


def test_node_stores_hidden_state_and_reward():
    """Verify hidden_state tensor and scalar reward can be assigned cleanly."""
    node = Node(action=1, action_space_size=9)
    latent = torch.randn(1, 64, 3, 3)

    node.hidden_state = latent
    node.reward = 1.0

    assert node.hidden_state is latent
    assert node.reward == 1.0


def test_node_uses_slots():
    """Ensure __slots__ prevents dynamic attribute creation and saves memory."""
    node = Node(action=0, action_space_size=9)

    assert not hasattr(node, "__dict__")
    with pytest.raises(AttributeError):
        node.unexpected_attr = 123


def test_node_zero_action_space():
    """Edge case: action_space_size=0 creates an empty children list."""
    node = Node(action=0, action_space_size=0)
    assert node.children == []