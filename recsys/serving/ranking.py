"""The request path: retrieve with numpy, rerank with ONNX Runtime.

The same arithmetic as `recsys.large.retrieve` and `recsys.large.rerank`,
rewritten for one user at a time and without torch. `tests/test_serving.py`
checks the two agree, so the service ranks exactly as the evaluation did.
"""
import numpy as np

from ..data import PAD


def retrieve(user_vector, item_vectors, history, k):
    """Top `k` unseen items by dot product: (indices, scores), best first."""
    scores = item_vectors @ user_vector
    scores[PAD] = -np.inf
    if len(history):
        scores[np.asarray(history, dtype=np.int64)] = -np.inf
    k = min(k, int(np.isfinite(scores).sum()))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]


def standardise(row):
    row = np.asarray(row, dtype=np.float32)
    spread = row.std()
    return (row - row.mean()) / (spread if spread > 1e-8 else 1.0)


def padded_window(history, max_length):
    """The left-padded int64 window the reranker reads, shape (1, max_length)."""
    window = np.asarray(history[-max_length:], dtype=np.int64)
    out = np.full((1, max_length), PAD, dtype=np.int64)
    if len(window):
        out[0, -len(window):] = window
    return out


class OnnxReranker:
    """The exported BERT4Rec candidate scorer under ONNX Runtime."""

    def __init__(self, path, max_length, threads=None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"])
        self.max_length = max_length

    def score(self, history, candidates):
        """Sequence-model scores for `candidates`, shape (len(candidates),)."""
        (scores,) = self.session.run(None, {
            "sequences": padded_window(history, self.max_length),
            "candidates": np.asarray(candidates, dtype=np.int64)[None, :],
        })
        return scores[0]


def blend(sequence_scores, retrieval_scores, beta):
    """beta * sequence + (1 - beta) * retrieval, each standardised per row."""
    return (beta * standardise(sequence_scores)
            + (1 - beta) * standardise(retrieval_scores))
