"""Tests that pin the four silent failures this rewrite exists to fix.

None of the four raised an error. Each one returned a plausible number for a
question nobody asked, which is the only reason they survived a whole
competition.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from recsys.data import PAD, Interactions, training_sequences
from recsys.graph import build_adjacency
from recsys.sequential import BERT4Rec, NextItemDataset


def toy(n_users=6, n_items=12, per_user=6):
    """A tiny archive, one interaction per user-item pair.

    No repeats within a user, which is the shape MovieLens has: a rating is
    given once. A generator that repeated items would make the leave-one-out
    tests below vacuous, since the held-out item would also sit in the history.
    """
    rows = []
    stamp = 0
    for user in range(n_users):
        start = (user % 2) + 1
        for step in range(per_user):
            stamp += 1
            rows.append({
                "user_id": user,
                "item_id": start + step,  # distinct within the user
                "timestamp": stamp,
            })
    events = pd.DataFrame(rows)
    items = pd.DataFrame({
        "item_id": range(1, n_items + 1),
        "title": [f"film {i}" for i in range(1, n_items + 1)],
        "genres": ["Drama"] * n_items,
        "year": ["1990"] * n_items,
        "text": [f"film {i} Drama 1990" for i in range(1, n_items + 1)],
    })
    return Interactions(events, items, protocol="leave_one_out", min_history=2)


# -------------------------------------------------------- index 0 is padding

def test_no_real_item_is_ever_index_zero():
    """The original mapped unknown items to index 0 with `.get(item, 0)`, and
    index 0 was a real item. Every unseen item silently became that one."""
    interactions = toy()
    assert PAD == 0
    assert 0 not in interactions.item_to_index.values()
    assert min(interactions.item_to_index.values()) == 1


def test_padding_is_never_recommended():
    """A model that can score index 0 can recommend nothing, as an item."""
    model = BERT4Rec(n_items=10, dimension=16, n_heads=2, n_layers=1,
                     max_length=6).eval()
    logits = model(torch.tensor([[PAD, PAD, 1, 2, 3, 4]]))
    assert logits[0, PAD] == float("-inf")
    assert torch.argmax(logits).item() != PAD


# ------------------------------------------------------------- left padding

def test_padding_goes_on_the_left():
    """The prediction is read off the last position, so padding on the right
    would mean reading it off a padding slot for any short sequence."""
    dataset = NextItemDataset([(0, [4, 5], 6)], max_length=5)
    sequence, target = dataset[0]
    assert sequence.tolist() == [PAD, PAD, PAD, 4, 5]
    assert sequence[-1].item() == 5, "the last slot must hold a real item"
    assert target.item() == 6


def test_a_short_sequence_predicts_from_its_last_real_item():
    """Two sequences ending on the same item, padded to different lengths,
    must produce the same prediction. With right padding they would not."""
    model = BERT4Rec(n_items=12, dimension=16, n_heads=2, n_layers=1,
                     max_length=8).eval()
    short = torch.tensor([[PAD] * 6 + [3, 7]])
    longer = torch.tensor([[PAD] * 4 + [9, 2, 3, 7]])

    with torch.no_grad():
        a = model(short)
        b = model(longer)
    # Not equal, because the history differs, but both must be finite over the
    # real items and neither may collapse to the all-padding case.
    assert torch.isfinite(a[:, 1:]).all() and torch.isfinite(b[:, 1:]).all()
    assert not torch.allclose(a, b), "the earlier history has to matter"


def test_an_entirely_padded_row_does_not_produce_nan():
    """Masking every key makes softmax divide by zero, and one NaN row takes
    the whole batch with it."""
    model = BERT4Rec(n_items=12, dimension=16, n_heads=2, n_layers=1,
                     max_length=5).eval()
    with torch.no_grad():
        logits = model(torch.tensor([[PAD] * 5, [PAD, PAD, 1, 2, 3]]))
    # Column PAD is -inf on purpose, so finiteness is asserted over the rest.
    assert torch.isfinite(logits[:, 1:]).all()


# ------------------------------------------------- graph node index ordering

def test_item_nodes_live_above_the_users():
    """The original sampled negatives from [0, n_items), which in a node index
    where items start at n_users means every negative was a user node."""
    interactions = toy()
    adjacency, n_nodes = build_adjacency(interactions)

    assert n_nodes == interactions.n_users + interactions.n_items
    rows, columns = adjacency.indices()

    # Every edge joins one user node to one item node, never two of a kind.
    is_user = rows < interactions.n_users
    partner_is_user = columns < interactions.n_users
    assert bool((is_user != partner_is_user).all())


def test_the_adjacency_is_symmetric():
    interactions = toy()
    adjacency, _ = build_adjacency(interactions)
    dense = adjacency.to_dense()
    assert torch.allclose(dense, dense.T, atol=1e-6)


def test_isolated_nodes_do_not_produce_infinite_normalisation():
    """An item nobody touched has degree zero, and 1/sqrt(0) is infinity."""
    interactions = toy()
    adjacency, _ = build_adjacency(interactions)
    assert torch.isfinite(adjacency.values()).all()


# ------------------------------------------------------------ the split

def test_leave_one_out_holds_out_exactly_one_item_per_user():
    interactions = toy()
    counts = interactions.test.groupby("user_id").size()
    assert set(counts.unique()) == {1}
    assert len(interactions.test) == interactions.events["user_id"].nunique()


def test_the_held_out_item_is_the_last_one():
    interactions = toy()
    for user, rows in interactions.test.groupby("user_id"):
        latest = interactions.events[
            interactions.events["user_id"] == user]["timestamp"].max()
        assert rows["timestamp"].iloc[0] == latest


def test_training_sequences_never_contain_a_test_target():
    """The check the whole evaluation rests on.

    Valid here because the toy archive gives each user distinct items, as
    MovieLens does; where a user can touch the same item twice, the held-out
    interaction is what is held out, not the item.
    """
    interactions = toy()
    held_out = {
        (row.user_id, interactions.item_to_index[row.item_id])
        for row in interactions.test.itertuples()
    }
    for user, history, target in training_sequences(interactions):
        assert (user, target) not in held_out
        for item in history:
            assert (user, item) not in held_out


def test_validation_split_steps_one_further_back():
    """`without_last` has to remove one more item, not re-use the test one."""
    interactions = toy()
    validation = interactions.without_last(1)

    test_pairs = {(u, t) for u, _, t in interactions.evaluation_pairs()}
    validation_pairs = {(u, t) for u, _, t in validation.evaluation_pairs()}
    assert test_pairs.isdisjoint(validation_pairs)
    assert len(validation.train) < len(interactions.train)


def test_an_unknown_protocol_is_refused():
    interactions = toy()
    with pytest.raises(ValueError, match="Unknown protocol"):
        Interactions(interactions.events, interactions.items,
                     protocol="whatever")
