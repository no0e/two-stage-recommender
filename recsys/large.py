"""Retrieval and reranking that never build a users-by-items array.

Scoring one user at a time in numpy is the clear way to write this, and at a
few thousand items it costs nothing. At 162,000 items and 20,000 evaluation
users the same loop asks for 13 GB of scores and takes an hour of Python.

So the two hot parts are done differently. Retrieval multiplies a block
of users against the whole item table on the GPU and takes the top k of each
row before the block is released, so the largest array alive is one block.
Reranking scores only those k columns, which is what `score_sequences` does
when it is given candidates.

Nothing here is specific to Amazon. It is specific to a catalogue too large to
hold a dense score matrix for.
"""
import numpy as np
import torch

from .data import PAD
from .metrics import ndcg_at_k, recall_at_k
from .models import score_sequences

NEG = float("-inf")


def _mask_rows(scores, n_rows, lengths, flat_history):
    """Set every already-seen item, and padding, to -inf across the block."""
    scores[:, PAD] = NEG
    if not len(flat_history):
        return
    rows = torch.repeat_interleave(
        torch.arange(n_rows, device=scores.device), lengths)
    scores[rows, flat_history] = NEG


@torch.no_grad()
def retrieve(user_vectors, item_vectors, users, histories, k=100,
             block=512, device="cpu"):
    """Top `k` items per user from an embedding dot product, in blocks.

    Returns (candidates, values), both (n_users, k) numpy arrays. A user with
    no embedding — one the graph never saw — comes back as an empty row, which
    the caller sends to the fallback.
    """
    item_table = torch.as_tensor(item_vectors, device=device)
    user_table = torch.as_tensor(user_vectors, device=device)

    candidates = np.zeros((len(users), k), dtype=np.int64)
    values = np.zeros((len(users), k), dtype=np.float32)

    for start in range(0, len(users), block):
        stop = min(start + block, len(users))
        rows = torch.as_tensor(
            np.asarray(users[start:stop], dtype=np.int64), device=device)
        scores = user_table[rows] @ item_table.T          # (block, n_items)

        chunk = histories[start:stop]
        lengths = torch.tensor([len(h) for h in chunk], device=device)
        flat = torch.as_tensor(
            np.concatenate([np.asarray(h, dtype=np.int64) for h in chunk])
            if any(len(h) for h in chunk) else np.zeros(0, dtype=np.int64),
            device=device,
        )
        _mask_rows(scores, len(chunk), lengths, flat)

        top_values, top_indices = torch.topk(scores, k, dim=1)
        candidates[start:stop] = top_indices.cpu().numpy()
        values[start:stop] = top_values.float().cpu().numpy()
        del scores

    return candidates, values


def standardise_rows(matrix):
    """Per row, zero mean and unit scale, so two score families can be added."""
    matrix = np.asarray(matrix, dtype=np.float32)
    mean = matrix.mean(axis=1, keepdims=True)
    spread = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / np.where(spread > 1e-8, spread, 1.0)


def rerank(model, histories, candidates, retrieval_values, beta=0.5,
           max_length=50, device="cpu", batch_size=256):
    """Reorder each candidate row, blending the sequence score into it.

    beta is how much of the final order the sequence model sets. At 0 the
    retrieval order stands untouched; at 1 the retrieval score is discarded.
    """
    if beta == 0.0:
        return candidates

    sequence = score_sequences(
        model, histories, max_length=max_length, device=device,
        batch_size=batch_size, candidates=[list(row) for row in candidates])

    combined = (beta * standardise_rows(sequence)
                + (1 - beta) * standardise_rows(retrieval_values))
    order = np.argsort(-combined, axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def score_ranked(ranked, targets, k=10, catalogue_size=None):
    """recall@k, ndcg@k, coverage and the per-user hits, in one pass."""
    hits, gains, seen = [], [], set()
    for row, target in zip(ranked, targets):
        top = list(row[:k])
        seen.update(top)
        hits.append(recall_at_k(top, target, k))
        gains.append(ndcg_at_k(top, target, k))

    result = {
        f"recall@{k}": float(np.mean(hits)) if hits else 0.0,
        f"ndcg@{k}": float(np.mean(gains)) if gains else 0.0,
        "evaluated": len(hits),
        "distinct_items_recommended": len(seen),
    }
    if catalogue_size:
        result["catalogue_coverage"] = len(seen) / catalogue_size
    return result, hits


def popularity_ranked(popular, histories, k=10):
    """The free baseline, as a ranked row per user."""
    popular = list(popular)
    rows = []
    for history in histories:
        seen = set(history)
        rows.append([i for i in popular if i not in seen][:k])
    return rows



@torch.no_grad()
def sequential_topk(model, histories, k=10, max_length=50, device="cpu",
                    batch_size=128):
    """Top `k` over the whole catalogue from the sequence model alone.

    The reranker scored as a recommender in its own right, which is the
    baseline that says whether the retrieval stage is buying anything. The top
    k is taken on the device and only that is copied back, so the full logits
    of a block never leave the GPU and never accumulate.
    """
    model.eval()
    out = np.zeros((len(histories), k), dtype=np.int64)

    for start in range(0, len(histories), batch_size):
        chunk = histories[start:start + batch_size]
        padded = [
            [PAD] * (max_length - len(h[-max_length:])) + list(h[-max_length:])
            for h in chunk
        ]
        batch = torch.tensor(padded, dtype=torch.long, device=device)
        scores = model(batch)

        lengths = torch.tensor([len(h) for h in chunk], device=device)
        flat = torch.as_tensor(
            np.concatenate([np.asarray(h, dtype=np.int64) for h in chunk])
            if any(len(h) for h in chunk) else np.zeros(0, dtype=np.int64),
            device=device,
        )
        _mask_rows(scores, len(chunk), lengths, flat)

        out[start:start + len(chunk)] = torch.topk(
            scores, k, dim=1).indices.cpu().numpy()
        del scores

    return out
