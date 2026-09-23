"""Tests for the two-stage recommender itself.

Added after the reranker was found to be throwing the retrieval score away, and
after a patch introduced a `beta` the constructor did not accept: nothing in the
suite built this class, so nothing noticed.
"""
import numpy as np
import pytest
import torch

from recsys.content import ContentRecommender
from recsys.pipeline import (
    PopularityRecommender, RetrievalBlend, TwoStageRecommender, _standardise,
)
from recsys.sequential import BERT4Rec

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
    """It did not, for one commit, and no test built the class."""
    two_stage = build(beta=0.5)
    assert two_stage.beta == 0.5


def test_beta_zero_keeps_the_retrieval_order():
    """With the sequence model given no weight, the output must be exactly the
    retrieval ranking. This is the property the first version broke."""
    two_stage = build(beta=0.0)
    history = [1, 2]
    ranked = two_stage.recommend_many([(0, history, None)], k=3)[0]
    assert ranked == two_stage.candidates(0, history)[:3]


def test_beta_one_ignores_the_retrieval_order():
    """At the other end the reranker decides alone, which is what the original
    did unconditionally."""
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
