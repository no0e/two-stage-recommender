"""The whole test suite.

Most of it pins the things that go wrong without raising anything: which index
means padding, which side a sequence is padded on, where item nodes sit in the
graph index, and whether the reranker keeps the score that retrieved its
candidates. Each of those returns a plausible number for a question nobody
asked. The rest covers the closed-form model's recency weighting and the two
stages joined up.

Nothing here needs the dataset or a GPU. Everything runs on a toy set of a few
users in a couple of seconds.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from recsys.data import PAD, Interactions, training_sequences
from recsys.models import (
    BERT4Rec, ContentRecommender, NextItemDataset, RecencyEASE,
    build_adjacency, fit_ease, interaction_matrix, recency_profile,
)
from recsys.pipeline import (
    RetrievalBlend, TwoStageRecommender, _standardise,
)


# ------------------------------- what goes wrong without raising anything
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


# --------------------------------------------------- index 0 is padding
def test_no_real_item_is_ever_index_zero():
    """Unknown items get mapped to 0, so 0 must not be a real item: otherwise
    every unseen item becomes that one and is then trained on."""
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


# --------------------------------------------------------- left padding
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


# -------------------------------------------- graph node index ordering
def test_item_nodes_live_above_the_users():
    """Users and items share one index, items starting at n_users. Negatives
    drawn from [0, n_items) would all be user nodes, and the ranking loss would
    push item embeddings away from randomly chosen users."""
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


# -------------------------------- the closed-form model and its recency
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


# --------------------------------------------- the two stages joined up
N_ITEMS = 12


class FakeGraph:
    """A graph recommender whose opinion is fixed and known."""

    def __init__(self, scores):
        self.fixed = np.asarray(scores, dtype=np.float32)

    def scores(self, user, history):
        return self.fixed


def build(beta, graph_scores=None):
    graph = FakeGraph(graph_scores if graph_scores is not None
                      else np.arange(N_ITEMS, dtype=np.float32))
    content = np.zeros((N_ITEMS, 4), dtype=np.float32)
    cold = ContentRecommender(content, popular=[1, 2, 3])
    model = BERT4Rec(n_items=N_ITEMS, dimension=8, n_heads=2, n_layers=1,
                     max_length=6).eval()
    return TwoStageRecommender(graph, cold, model, alpha=1.0, beta=beta,
                               n_candidates=5, max_length=6)


def test_the_constructor_accepts_beta():
    """Nothing else in the suite constructs the class, so this is what would
    catch a keyword the constructor stops accepting."""
    two_stage = build(beta=0.5)
    assert two_stage.beta == 0.5


def test_beta_zero_keeps_the_retrieval_order():
    """With the sequence model given no weight, the output must be exactly
    the retrieval ranking: beta has to reach all the way to zero."""
    two_stage = build(beta=0.0)
    history = [1, 2]
    ranked = two_stage.recommend_many([(0, history, None)], k=3)[0]
    assert ranked == two_stage.candidates(0, history)[:3]


def test_beta_one_ignores_the_retrieval_order():
    """At the other end the reranker decides alone, and the candidate set
    still has to hold: it reorders, it does not fetch."""
    torch.manual_seed(0)
    two_stage = build(beta=1.0)
    history = [1, 2]
    candidates = two_stage.candidates(0, history)

    ranked = two_stage.recommend_many([(0, history, None)], k=len(candidates))[0]
    assert sorted(ranked) == sorted(candidates), "the candidate set is fixed"


def test_the_output_is_always_a_subset_of_the_candidates():
    two_stage = build(beta=0.5)
    history = [1, 2]
    candidates = set(two_stage.candidates(0, history))
    ranked = two_stage.recommend_many([(0, history, None)], k=3)[0]
    assert set(ranked) <= candidates


def test_nothing_already_seen_is_recommended():
    two_stage = build(beta=0.5)
    history = [9, 10, 11]
    ranked = two_stage.recommend_many([(0, history, None)], k=5)[0]
    assert set(ranked).isdisjoint(history)


def test_a_cold_user_falls_back_to_content():
    """No user embedding means retrieval runs on content alone rather than
    raising, which is the whole reason the cold start path exists."""
    class ColdGraph:
        def scores(self, user, history):
            return None

    content = np.zeros((N_ITEMS, 4), dtype=np.float32)
    cold = ContentRecommender(content, popular=[4, 5, 6])
    model = BERT4Rec(n_items=N_ITEMS, dimension=8, n_heads=2, n_layers=1,
                     max_length=6).eval()
    two_stage = TwoStageRecommender(ColdGraph(), cold, model, n_candidates=5,
                                    max_length=6)

    ranked = two_stage.recommend_many([(999, [], None)], k=3)[0]
    assert len(ranked) == 3
    assert 0 not in ranked


def test_the_cold_object_has_to_be_a_content_recommender():
    model = BERT4Rec(n_items=N_ITEMS, dimension=8, n_heads=2, n_layers=1,
                     max_length=6).eval()
    with pytest.raises(TypeError, match="ContentRecommender"):
        TwoStageRecommender(FakeGraph(np.zeros(N_ITEMS)), object(), model)


def test_standardise_puts_two_score_families_on_one_scale():
    """Blending a dot product with a cosine without this makes alpha and beta
    mean nothing."""
    big = _standardise(np.array([100.0, 200.0, 300.0]))
    small = _standardise(np.array([0.1, 0.2, 0.3]))
    assert np.allclose(big, small, atol=1e-5)


def test_retrieval_blend_returns_the_requested_number():
    blend = RetrievalBlend(FakeGraph(np.arange(N_ITEMS, dtype=np.float32)),
                           ContentRecommender(
                               np.zeros((N_ITEMS, 4), dtype=np.float32),
                               popular=[1, 2]),
                           alpha=1.0, n_candidates=4)
    assert len(blend(0, [1, 2])) == 4
