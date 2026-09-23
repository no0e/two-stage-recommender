"""Tests for the closed-form retrieval model and its recency weighting."""
import numpy as np
import pytest

from recsys.linear import (
    RecencyEASE, fit_ease, interaction_matrix, recency_profile,
)
from tests.test_bugs import toy


def test_the_weight_matrix_has_a_zero_diagonal():
    """Without the constraint the closed form is the identity, which explains
    every item by itself and recommends nothing."""
    weights = fit_ease(interaction_matrix(toy()), l2=10.0)
    assert np.allclose(np.diag(weights), 0.0)


def test_the_weight_matrix_is_square_over_the_catalogue():
    interactions = toy()
    weights = fit_ease(interaction_matrix(interactions), l2=10.0)
    assert weights.shape == (interactions.n_items, interactions.n_items)


def test_stronger_regularisation_shrinks_the_weights():
    matrix = interaction_matrix(toy())
    light = np.abs(fit_ease(matrix, l2=1.0)).sum()
    heavy = np.abs(fit_ease(matrix, l2=1000.0)).sum()
    assert heavy < light


def test_the_most_recent_item_weighs_one():
    profile = recency_profile([3, 5, 7], n_items=10, half_life=20.0)
    assert profile[7] == pytest.approx(1.0)


def test_weights_decay_towards_the_past():
    profile = recency_profile([1, 2, 3, 4], n_items=10, half_life=2.0)
    assert profile[4] > profile[3] > profile[2] > profile[1]


def test_one_half_life_back_halves_the_weight():
    profile = recency_profile([1, 2, 3], n_items=10, half_life=1.0)
    assert profile[2] == pytest.approx(0.5)
    assert profile[1] == pytest.approx(0.25)


def test_no_half_life_reproduces_the_unordered_bag():
    """The behaviour of plain EASE, which this model exists to improve on."""
    profile = recency_profile([2, 5, 9], n_items=12, half_life=None)
    assert set(np.flatnonzero(profile)) == {2, 5, 9}
    assert np.allclose(profile[[2, 5, 9]], 1.0)


def test_a_repeated_item_keeps_its_most_recent_weight():
    """Summing instead would let a rewatched film outrank everything by
    repetition rather than by relevance."""
    profile = recency_profile([4, 1, 4], n_items=10, half_life=1.0)
    assert profile[4] == pytest.approx(1.0)


def test_an_empty_history_gives_an_empty_profile():
    assert not recency_profile([], n_items=10, half_life=5.0).any()


def test_scoring_a_user_with_no_history_returns_none():
    """The cold start path takes over rather than this model inventing an
    answer out of a zero vector."""
    model = RecencyEASE(toy(), l2=10.0)
    assert model.scores(user=0, history=[]) is None
    assert model(0, [], k=5) == []


def test_recommendations_exclude_the_history():
    interactions = toy()
    model = RecencyEASE(interactions, l2=10.0)
    history = [1, 2, 3]
    assert set(model(0, history, k=5)).isdisjoint(history)


def two_taste_archive():
    """Users who watch inside one of two genres, and never across.

    Built so recency has something to say: a user who starts in one genre and
    ends in the other should be recommended the genre they moved to.
    """
    import pandas as pd

    from recsys.data import Interactions

    left, right = list(range(1, 9)), list(range(9, 17))
    rows, stamp = [], 0
    for user in range(30):
        block = left if user % 2 == 0 else right
        for item in block:
            stamp += 1
            rows.append({"user_id": user, "item_id": item, "timestamp": stamp})

    events = pd.DataFrame(rows)
    items = pd.DataFrame({
        "item_id": range(1, 17),
        "title": [f"film {i}" for i in range(1, 17)],
        "genres": ["A"] * 8 + ["B"] * 8,
        "year": ["1990"] * 16,
        "text": [f"film {i}" for i in range(1, 17)],
    })
    return Interactions(events, items, protocol="leave_one_out", min_history=2)


def test_recency_moves_the_ranking_towards_the_recent_taste():
    """A user who switched genres should be recommended what they switched to.

    The unordered bag cannot express that, which is the whole reason the
    recency weight exists.
    """
    interactions = two_taste_archive()
    index = interactions.item_to_index
    # Four films from the first genre, then four from the second.
    history = [index[i] for i in (1, 2, 3, 4)] + [index[i] for i in (9, 10, 11, 12)]
    second_genre = {index[i] for i in range(9, 17)}

    bagged = RecencyEASE(interactions, l2=1.0, half_life=None)
    recent = RecencyEASE(interactions, l2=1.0, half_life=1.0)

    from_bag = bagged(999, history, k=4)
    from_recent = recent(999, history, k=4)

    share_bag = sum(i in second_genre for i in from_bag) / max(len(from_bag), 1)
    share_recent = sum(i in second_genre for i in from_recent) / max(len(from_recent), 1)
    assert share_recent >= share_bag
    assert share_recent > 0.5, "the recent taste should dominate the top of the list"
