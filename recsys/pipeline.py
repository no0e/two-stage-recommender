"""The two stages joined up, plus the baselines they are measured against.

Retrieve a hundred candidates cheaply, rerank ten of them expensively. The shape
exists because scoring every item in the catalogue with a transformer, for every
user, is not something that runs inside a competition deadline, and because the
two stages are good at different things: the graph knows who resembles whom, the
sequence model knows what follows what.

Every recommender in this file has the same signature, `(user, history) ->
ranked indices`, so `metrics.evaluate` can run any of them and the comparison
is never between two differently-shaped experiments.
"""
import numpy as np

from .data import PAD
from .models import ContentRecommender, score_sequences


class Recommender:
    """Shared interface: score one, or score a list in whatever way suits.

    `recommend_many` exists so the evaluation loop never has to know which
    recommenders can batch. The default is a loop; the two that wrap a
    transformer override it.
    """

    def recommend_many(self, triples, k=10):
        return [self(user, history, k) for user, history, _ in triples]


class PopularityRecommender(Recommender):
    """The free baseline. Anything that cannot beat this has earned nothing."""

    def __init__(self, popular):
        self.popular = list(popular)

    def __call__(self, user, history, k=10):
        seen = set(history)
        return [i for i in self.popular if i not in seen][:k]


class GraphRecommender(Recommender):
    """LightGCN alone: the retrieval stage, scored on its own."""

    def __init__(self, interactions, user_vectors, item_vectors, fallback):
        self.interactions = interactions
        self.user_vectors = user_vectors
        self.item_vectors = item_vectors
        self.fallback = fallback

    def scores(self, user, history):
        index = self.interactions.user_to_index.get(user)
        if index is None:
            return None
        return self.item_vectors @ self.user_vectors[index]

    def __call__(self, user, history, k=100):
        scores = self.scores(user, history)
        if scores is None:
            return self.fallback(user, history)[:k]
        return _top(scores, history, k)


class SequentialRecommender(Recommender):
    """BERT4Rec alone, over the whole catalogue: the reranker without retrieval."""

    def __init__(self, model, max_length=50, device="cpu"):
        self.model = model
        self.max_length = max_length
        self.device = device

    def __call__(self, user, history, k=100):
        return self.recommend_many([(user, history, None)], k)[0]

    def recommend_many(self, triples, k=100):
        """Score every history in one batched pass.

        Scoring one sequence at a time turns an evaluation into thousands of
        batch-of-one transformer calls, which on a CPU is the difference
        between a minute and an afternoon. Every recommender here has this
        method so the evaluation loop can always use it.
        """
        histories = [history for _, history, _ in triples]
        logits = score_sequences(
            self.model, histories, self.max_length, self.device)
        return [_top(row, history, k)
                for row, history in zip(logits, histories)]


class TwoStageRecommender(Recommender):
    """Retrieve with the graph blended into content, rerank with the sequence.

    `alpha` weights the graph against content at retrieval. It is fitted on a
    validation slice rather than picked, because the right blend depends on how
    much history the dataset has and picking it by hand is how a number gets
    tuned on the test set without anyone noticing.
    """

    def __init__(self, graph, cold, model, alpha=1.0, beta=0.5,
                 n_candidates=100, max_length=50, device="cpu"):
        if not isinstance(cold, ContentRecommender):
            raise TypeError(
                "cold must be a ContentRecommender: it is both the cold start "
                "path and the content half of retrieval, and the two have to "
                "be the same object or they can disagree."
            )
        self.graph = graph
        self.model = model
        self.cold = cold
        self.alpha = alpha
        # How much of the final order the sequence model gets to set. At 1 it
        # reranks on its own opinion and the retrieval score is discarded,
        # which measures below leaving the retrieval order untouched: the
        # second stage is there to refine an order, not to replace it.
        self.beta = beta
        self.n_candidates = n_candidates
        self.max_length = max_length
        self.device = device

    def retrieval_scores(self, user, history):
        """The blended retrieval score over the whole catalogue."""
        graph_scores = self.graph.scores(user, history)
        content_scores = self.cold.scores(history)
        if graph_scores is None:
            # No user embedding: a cold user retrieves on content alone.
            return _standardise(content_scores)
        return (
            self.alpha * _standardise(graph_scores)
            + (1 - self.alpha) * _standardise(content_scores)
        )

    def candidates(self, user, history):
        return _top(self.retrieval_scores(user, history), history,
                    self.n_candidates)

    def __call__(self, user, history, k=10):
        return self.recommend_many([(user, history, None)], k)[0]

    def recommend_many(self, triples, k=10):
        """Retrieve for every triple, then rerank them all in one batch.

        Retrieval is numpy and cheap per user; reranking is a transformer and
        is not. Doing the second in one pass is what makes a full evaluation
        finish in a minute instead of an hour.
        """
        retrieved = []
        for user, history, _ in triples:
            scores = self.retrieval_scores(user, history)
            retrieved.append(
                (user, history, _top(scores, history, self.n_candidates), scores)
            )
        needs_rerank = [
            index for index, (_, history, candidates, _s) in enumerate(retrieved)
            if candidates and history
        ]

        scores = {}
        if needs_rerank:
            logits = score_sequences(
                self.model,
                [retrieved[index][1] for index in needs_rerank],
                self.max_length, self.device,
            )
            scores = dict(zip(needs_rerank, logits))

        out = []
        for index, (_, history, candidates, retrieval) in enumerate(retrieved):
            if not candidates:
                out.append(self.cold.recommend(history, k))
            elif index not in scores:
                # Nothing for the sequence model to read; retrieval order stands.
                out.append(candidates[:k])
            else:
                # Both opinions, each standardised over the candidate set so
                # neither scale dominates the other by accident.
                sequence = _standardise(scores[index][candidates])
                keep = _standardise(retrieval[candidates])
                combined = self.beta * sequence + (1 - self.beta) * keep
                order = np.argsort(-combined)
                out.append([candidates[j] for j in order[:k]])
        return out


def _standardise(scores):
    """Zero mean, unit scale, so two score families can be added at all.

    A dot product against an embedding and a cosine similarity do not live on
    the same scale, and blending them raw makes alpha meaningless.
    """
    scores = np.asarray(scores, dtype=np.float32)
    spread = scores.std()
    return (scores - scores.mean()) / (spread if spread > 1e-8 else 1.0)


def _top(scores, history, k):
    """Highest scoring indices, excluding padding and anything already seen."""
    seen = set(history)
    order = np.argsort(-np.asarray(scores))
    out = []
    for index in order:
        index = int(index)
        if index == PAD or index in seen:
            continue
        out.append(index)
        if len(out) == k:
            break
    return out


class RetrievalBlend:
    """The retrieval half on its own: graph scores blended into content scores.

    Split out from `TwoStageRecommender` so alpha can be chosen by how good the
    candidate set is, which is the only thing alpha controls. Sweeping it
    through the full pipeline would also drag the reranker into a decision it
    has no part in, and would need the reranker retrained for every honest
    validation split.
    """

    def __init__(self, graph, cold, alpha=1.0, n_candidates=100):
        self.graph = graph
        self.cold = cold
        self.alpha = alpha
        self.n_candidates = n_candidates

    def __call__(self, user, history, k=None):
        graph_scores = self.graph.scores(user, history)
        content_scores = self.cold.scores(history)
        if graph_scores is None:
            blended = content_scores
        else:
            blended = (
                self.alpha * _standardise(graph_scores)
                + (1 - self.alpha) * _standardise(content_scores)
            )
        return _top(blended, history, k or self.n_candidates)


def fit_alpha_on_retrieval(blend, pairs, candidates=(0.0, 0.25, 0.5, 0.75, 1.0),
                           verbose=True):
    """Choose the blend by how often the candidate set contains the target.

    `blend` must be built from models trained without the validation targets.
    Choosing alpha against targets the graph was trained on would reward the
    graph for memorising them, which is exactly the answer the sweep would then
    give: alpha = 1, use the graph and nothing else.
    """
    from .metrics import recall_at_k

    sweep, n = {}, blend.n_candidates
    for alpha in candidates:
        blend.alpha = alpha
        hits = [
            recall_at_k(blend(user, history, n), target, n)
            for user, history, target in pairs
        ]
        sweep[alpha] = float(np.mean(hits)) if hits else 0.0
        if verbose:
            print(f"  alpha={alpha:.2f}  recall@{n}={sweep[alpha]:.4f}")

    best = max(sweep, key=sweep.get)
    blend.alpha = best
    return best, sweep


def fit_beta(two_stage, pairs, candidates=(0.0, 0.25, 0.5, 0.75, 1.0), k=10,
             verbose=True):
    """Choose how much of the final order the sequence model sets.

    `two_stage` must be built from models that never saw these targets. beta=0
    is retrieval order untouched, beta=1 is the reranker deciding alone, and
    the sweep is reported whole because a flat curve and a peaked one say
    different things about whether the second stage is worth its cost.
    """
    from .metrics import recall_at_k

    sweep = {}
    for beta in candidates:
        two_stage.beta = beta
        ranked = two_stage.recommend_many(pairs, k)
        sweep[beta] = float(np.mean([
            recall_at_k(row, target, k)
            for row, (_, _, target) in zip(ranked, pairs)
        ])) if pairs else 0.0
        if verbose:
            print(f"  beta={beta:.2f}  recall@{k}={sweep[beta]:.4f}")

    best = max(sweep, key=sweep.get)
    two_stage.beta = best
    return best, sweep
