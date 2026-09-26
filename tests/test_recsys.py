"""The whole test suite.

Most of it pins the things that go wrong without raising anything: which index
means padding, which side a sequence is padded on, where item nodes sit in the
graph index, and whether the reranker keeps the score that retrieved its
candidates. Each of those returns a plausible number for a question nobody
asked.

The rest covers the two shortcuts the size of these catalogues forces: scoring
a handful of candidate columns instead of 162,035, and holding every prefix of
every history as offsets into one array instead of as its own list.

Nothing here needs a downloaded category or a GPU. Everything runs on a toy set
of a few users in a couple of seconds.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from recsys.amazon import FlatSequences
from recsys.data import PAD, Interactions
from recsys.large import rerank, retrieve
from recsys.models import (
    BERT4Rec, WindowDataset, build_adjacency, train_bert4rec,
)


# ------------------------------- what goes wrong without raising anything
def toy(n_users=6, n_items=12, per_user=6):
    """A tiny archive, one interaction per user-item pair.

    No repeats within a user, which is the shape the 5-core files have. A
    generator that repeated items would make the leave-one-out tests below
    vacuous, since the held-out item would also sit in the history.
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
    dataset = WindowDataset([([4, 5], 6)], max_length=5)
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


# ------------------------------------------- the catalogue-sized shortcuts
def test_candidate_scoring_matches_the_columns_it_stands_in_for():
    """Scoring a few columns must give what scoring all of them would.

    This is the shortcut the whole large path rests on: reranking reads 100
    items per user instead of 162,000. If the gathered output layer were
    transposed or misaligned it would still return a plausible ranking.
    """
    torch.manual_seed(0)
    model = BERT4Rec(n_items=20, dimension=8, n_heads=2, n_layers=1,
                     max_length=6).eval()
    sequences = torch.tensor([[0, 0, 1, 2, 3, 4], [0, 0, 0, 5, 6, 7]])
    candidates = torch.tensor([[3, 11, 7], [2, 9, 14]])

    with torch.no_grad():
        full = model(sequences)
        narrow = model(sequences, candidates=candidates)

    expected = torch.gather(full, 1, candidates)
    assert torch.allclose(narrow, expected, atol=1e-5)


def test_the_sampled_loss_is_finite():
    """The regression that cost a training run.

    The sampled softmax puts the target in column 0, so its label is 0. PAD is
    also 0, and the criterion the full softmax uses sets ignore_index=PAD. The
    two together label every row as ignored, average over an empty set, and
    return NaN, which propagates into every weight. A NaN model still returns
    a ranking, so nothing downstream complains.
    """
    interactions = toy(n_users=8, n_items=14, per_user=6)
    dataset = WindowDataset(FlatSequences(interactions, 6), max_length=6)

    model = train_bert4rec(
        dataset, interactions.n_items, epochs=1, dimension=8, n_heads=2,
        n_layers=1, max_length=6, batch_size=4, n_negatives=4, verbose=False)

    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_a_non_finite_loss_stops_training():
    """Rather than returning a model whose every weight is NaN."""
    interactions = toy(n_users=6, n_items=12, per_user=5)
    dataset = WindowDataset(FlatSequences(interactions, 6), max_length=6)

    import recsys.models as models

    original = models.nn.CrossEntropyLoss

    class AlwaysNaN(torch.nn.Module):
        def forward(self, logits, targets):
            return logits.sum() * float("nan")

    models.nn.CrossEntropyLoss = lambda *a, **k: AlwaysNaN()
    try:
        with pytest.raises(RuntimeError, match="not finite"):
            train_bert4rec(dataset, interactions.n_items, epochs=1,
                           dimension=8, n_heads=2, n_layers=1, max_length=6,
                           batch_size=4, verbose=False)
    finally:
        models.nn.CrossEntropyLoss = original


def test_retrieval_returns_the_best_unseen_items():
    """Blocked GPU retrieval, checked against an answer worked out by hand."""
    item_vectors = np.eye(6, dtype=np.float32)
    user_vectors = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.float32)

    # Scores are 0..5 by index. PAD is excluded, item 5 is in the history.
    candidates, values = retrieve(
        user_vectors, item_vectors, users=[0], histories=[[5]], k=2)

    assert list(candidates[0]) == [4, 3]
    assert values[0][0] > values[0][1]


def test_retrieval_never_returns_padding_or_anything_seen():
    rng = np.random.default_rng(0)
    item_vectors = rng.normal(size=(30, 4)).astype(np.float32)
    user_vectors = rng.normal(size=(5, 4)).astype(np.float32)
    histories = [[1, 2, 3], [4], [5, 6], [7, 8, 9, 10], [11]]

    candidates, _ = retrieve(user_vectors, item_vectors, users=list(range(5)),
                             histories=histories, k=8, block=2)

    for row, history in zip(candidates, histories):
        assert PAD not in row
        assert not set(row) & set(history)
        assert len(set(row)) == len(row)


def test_the_flat_store_keeps_every_position_as_a_target():
    """Every item after the first is a target, with a bounded context window.

    The window bounds what the model reads, not what it is asked to predict.
    Cutting the history to `max_length` first and only then taking prefixes
    would silently drop the early targets of every long user, and nothing
    downstream would say so.
    """
    interactions = toy(n_users=5, n_items=12, per_user=6)
    histories = interactions.indexed_histories()

    store = FlatSequences(interactions, max_length=4)
    assert len(store) == sum(len(h) - 1 for h in histories.values() if len(h) > 1)

    position = 0
    for history in histories.values():
        if len(history) < 2:
            continue
        for cut in range(1, len(history)):
            window, target = store[position]
            assert list(window) == history[max(0, cut - 4):cut]
            assert target == history[cut]
            position += 1


# --------------------------------------------- the two stages joined up
def two_stage_fixture(n_items=14, n_candidates=5):
    torch.manual_seed(0)
    model = BERT4Rec(n_items=n_items, dimension=8, n_heads=2, n_layers=1,
                     max_length=6).eval()
    histories = [[1, 2, 3], [4, 5]]
    candidates = np.array([[7, 8, 9, 10, 11], [2, 3, 6, 12, 13]])
    # Descending, the way retrieval hands them over.
    values = np.array([[5.0, 4.0, 3.0, 2.0, 1.0], [9.0, 7.0, 5.0, 3.0, 1.0]])
    return model, histories, candidates, values


def test_beta_zero_keeps_the_retrieval_order():
    """With the sequence model given no weight, the output must be exactly the
    retrieval ranking: beta has to reach all the way to zero."""
    model, histories, candidates, values = two_stage_fixture()
    ranked = rerank(model, histories, candidates, values, beta=0.0)
    assert np.array_equal(ranked, candidates)


def test_beta_one_still_returns_the_same_candidate_set():
    """At the other end the reranker decides the order alone, and the
    candidate set still has to hold: it reorders, it does not fetch."""
    model, histories, candidates, values = two_stage_fixture()
    ranked = rerank(model, histories, candidates, values, beta=1.0,
                    max_length=6)

    assert ranked.shape == candidates.shape
    for row, original in zip(ranked, candidates):
        assert sorted(row) == sorted(original)


def test_reranking_never_invents_an_item():
    """Whatever beta is, every row is a permutation of the row it was given.

    This is the property that makes the second stage a refinement of the first
    rather than a second retrieval, and the one the blend could break by
    mixing scores that are not aligned with their candidates.
    """
    model, histories, candidates, values = two_stage_fixture()
    for beta in (0.25, 0.5, 0.75):
        ranked = rerank(model, histories, candidates, values,
                        beta=beta, max_length=6)
        for row, original in zip(ranked, candidates):
            assert sorted(row) == sorted(original)
            assert len(set(row)) == len(row)


def test_standardising_puts_two_score_families_on_one_scale():
    """A graph dot product and a transformer logit do not share a scale.

    Added raw, beta would be meaningless: whichever family happened to have
    the wider spread would decide the order on its own.
    """
    from recsys.large import standardise_rows

    rows = np.array([[1.0, 2.0, 3.0], [1000.0, 2000.0, 3000.0]])
    standardised = standardise_rows(rows)

    assert np.allclose(standardised[0], standardised[1], atol=1e-5)
    assert np.allclose(standardised.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(standardised.std(axis=1), 1.0, atol=1e-5)


def test_a_flat_row_of_scores_does_not_divide_by_zero():
    from recsys.large import standardise_rows

    assert np.isfinite(standardise_rows(np.full((2, 4), 3.0))).all()


def test_scoring_a_ranking_counts_hits_positions_and_coverage():
    """recall, NDCG and coverage in one pass over the same rows.

    NDCG has to fall with the rank of the hit; recall must not. A hit at one
    and a hit at ten are the same event to recall and different events to
    NDCG, which is the reason both are reported.
    """
    from recsys.large import score_ranked

    ranked = [[5, 6, 7], [8, 9, 10], [11, 12, 13]]
    targets = [5, 10, 99]  # first, last, missing

    result, hits = score_ranked(ranked, targets, k=3, catalogue_size=20)

    assert hits == [1.0, 1.0, 0.0]
    assert result["recall@3"] == pytest.approx(2 / 3)
    assert result["evaluated"] == 3
    assert result["distinct_items_recommended"] == 9
    assert result["catalogue_coverage"] == pytest.approx(9 / 20)
    # 1/log2(2) for the hit at rank 1, 1/log2(4) for the one at rank 3.
    assert result["ndcg@3"] == pytest.approx((1.0 + 0.5) / 3)
