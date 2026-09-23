"""Ranking metrics, and the baselines every number has to be read against.

Recall@k on a single held-out target is the competition metric and is the one
quoted first. NDCG@k is reported beside it because recall treats a hit at rank 1
and a hit at rank 10 as the same event, and a reranking stage exists precisely
to tell those apart.

Coverage is here for a less obvious reason: a recommender that returns the same
twenty popular items to everybody scores respectably on both of the above. The
share of the catalogue it actually reaches is what makes that visible.
"""
import numpy as np


def recall_at_k(ranked, target, k=10):
    """1 if the target is in the first k, else 0."""
    return float(target in ranked[:k])


def ndcg_at_k(ranked, target, k=10):
    """Discounted gain for a single relevant item."""
    top = list(ranked[:k])
    if target not in top:
        return 0.0
    return float(1.0 / np.log2(top.index(target) + 2))


def rank_all(recommender, pairs, k=10):
    """Ranked lists for every triple, batched when the recommender can.

    Anything with a `recommend_many` gets to score a whole evaluation in one
    pass. Anything else is called per triple. The distinction never reaches the
    metric functions.
    """
    if hasattr(recommender, "recommend_many"):
        return [list(row)[:k] for row in recommender.recommend_many(pairs, k)]
    return [list(recommender(user, history, k))[:k]
            for user, history, _ in pairs]


def evaluate(recommender, pairs, k=10, catalogue_size=None):
    """Run a recommender over (user, history, target) triples."""
    ranked_lists = rank_all(recommender, pairs, k)

    recalls, gains, recommended = [], [], set()
    for ranked, (_, _, target) in zip(ranked_lists, pairs):
        recalls.append(recall_at_k(ranked, target, k))
        gains.append(ndcg_at_k(ranked, target, k))
        recommended.update(ranked)

    result = {
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(gains)) if gains else 0.0,
        "evaluated": len(recalls),
        "distinct_items_recommended": len(recommended),
    }
    if catalogue_size:
        result["catalogue_coverage"] = len(recommended) / catalogue_size
    return result


def bootstrap_difference(a_hits, b_hits, resamples=2000, seed=0):
    """Paired bootstrap of the difference in recall between two recommenders.

    Paired over the same evaluation triples, so the comparison is not confounded
    by which users happen to be easy. Without this, a difference of a point on a
    few thousand triples reads as a result when it is noise.
    """
    a_hits = np.asarray(a_hits, dtype=float)
    b_hits = np.asarray(b_hits, dtype=float)
    if len(a_hits) != len(b_hits):
        raise ValueError(
            f"Paired bootstrap needs the same triples: got {len(a_hits)} "
            f"and {len(b_hits)}."
        )

    difference = a_hits - b_hits
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), size=(resamples, len(difference)))
    means = difference[draws].mean(axis=1)
    return {
        "difference": float(difference.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": int(len(difference)),
    }


def hits(recommender, pairs, k=10):
    """Per-triple 0/1 outcomes, which is what the paired bootstrap needs."""
    return [
        recall_at_k(ranked, target, k)
        for ranked, (_, _, target) in zip(rank_all(recommender, pairs, k), pairs)
    ]
