"""A closed-form item-item model, with the ordering put back in.

EASE solves for an item-item weight matrix in one matrix inverse: no gradient
descent, no epochs, no learning rate. On this data it fits in about a second
and outranks a tuned LightGCN that takes five minutes, which is the sort of
result worth measuring before reaching for a graph network.

The addition here is recency. EASE scores a user by summing the columns of
every item in their history, which treats that history as an unordered bag. The
task is to predict the *next* item, so the ordering is the one thing the task is
built around and the one thing the bag throws away. Weighting the profile so
recent items count for more puts it back, at the cost of a single parameter, and
on validation it moved recall@10 from 0.185 to 0.226.
"""
import numpy as np


def interaction_matrix(interactions, frame=None):
    """Users by items, one where the interaction happened."""
    frame = interactions.train if frame is None else frame
    rows = frame["user_id"].map(interactions.user_to_index)
    columns = frame["item_id"].map(interactions.item_to_index)
    keep = rows.notna() & columns.notna()

    matrix = np.zeros(
        (interactions.n_users, interactions.n_items), dtype=np.float32)
    matrix[rows[keep].to_numpy(int), columns[keep].to_numpy(int)] = 1.0
    return matrix


def fit_ease(matrix, l2=100.0):
    """The EASE weight matrix: B = -P / diag(P), zero on the diagonal.

    The zero diagonal is the whole trick. Without it the closed-form solution
    is the identity, which reconstructs each item from itself perfectly and
    recommends nothing. Constraining it to zero forces every item to be
    explained by the others, and `l2` is what stops that explanation from
    memorising the training matrix.
    """
    gram = (matrix.T @ matrix).astype(np.float64)
    np.fill_diagonal(gram, np.diag(gram) + l2)

    precision = np.linalg.inv(gram)
    weights = -precision / np.diag(precision)
    np.fill_diagonal(weights, 0.0)
    return weights.astype(np.float32)


def recency_profile(history, n_items, half_life=20.0):
    """The user vector to score with, weighted towards recent items.

    The last item weighs one and the weight halves every `half_life` steps
    back. `half_life=None` gives the unordered bag, which is what plain EASE
    uses and what this exists to improve on.
    """
    profile = np.zeros(n_items, dtype=np.float32)
    if not len(history):
        return profile
    if half_life is None:
        profile[list(history)] = 1.0
        return profile

    positions = np.arange(len(history))
    ages = (len(history) - 1) - positions
    weights = np.power(0.5, ages / float(half_life)).astype(np.float32)
    # An item seen more than once keeps its most recent weight, not the sum,
    # so a rewatched film does not outrank everything by repetition alone.
    np.maximum.at(profile, np.asarray(history, dtype=int), weights)
    return profile


class RecencyEASE:
    """Retrieval by item-item weights over a recency-weighted profile.

    Scores from the history alone and never looks the user up, so a user the
    model has never seen is handled by the same code path as everyone else,
    provided they have done something. That is one fewer special case than the
    graph model needs.
    """

    def __init__(self, interactions, l2=100.0, half_life=20.0):
        self.interactions = interactions
        self.half_life = half_life
        self.weights = fit_ease(interaction_matrix(interactions), l2=l2)

    def scores(self, user, history):
        if not len(history):
            return None  # nothing to be similar to; the cold path takes over
        profile = recency_profile(
            history, self.interactions.n_items, self.half_life)
        return profile @ self.weights

    def __call__(self, user, history, k=100):
        from .pipeline import _top

        scores = self.scores(user, history)
        if scores is None:
            return []
        return _top(scores, history, k)

    def recommend_many(self, triples, k=100):
        from .pipeline import _top

        out = []
        for user, history, _ in triples:
            scores = self.scores(user, history)
            out.append([] if scores is None else _top(scores, history, k))
        return out
