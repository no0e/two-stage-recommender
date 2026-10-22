"""Turn trained models into the artifacts the service loads. Needs torch.

The reranker is exported as its candidate-scoring path only: (sequences,
candidates) -> scores. That is the only path the service calls, and it is the
one whose cost is the candidate count rather than the catalogue.

Two graphs are written. `reranker.onnx` is fp32. `reranker.int8.onnx` has its
weights dynamically quantized to int8, Gather included, which is where the
size is: two embedding tables of n_items x d each. Both are checked against
the PyTorch model on real histories before anything is written to meta.json.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data import PAD


class CandidateScorer(nn.Module):
    """BERT4Rec with the candidate path as its only forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sequences, candidates):
        return self.model(sequences, candidates=candidates)


def export_reranker(model, path, max_length, n_candidates=100, opset=18):
    """Write the candidate scorer to `path`, batch and candidate count dynamic."""
    model = model.eval().cpu()
    scorer = CandidateScorer(model).eval()
    sequences = torch.randint(1, model.n_items, (2, max_length))
    sequences[0, :max_length // 2] = PAD  # trace a padded row, as served
    candidates = torch.randint(1, model.n_items, (2, n_candidates))

    # The legacy TorchScript exporter handles nn.TransformerEncoder and its
    # key-padding mask without the fast path getting in the way; it is the one
    # tried first, and the dynamo exporter is the fallback where it is gone.
    try:
        torch.onnx.export(
            scorer, (sequences, candidates), str(path),
            input_names=["sequences", "candidates"], output_names=["scores"],
            dynamic_axes={"sequences": {0: "batch"},
                          "candidates": {0: "batch", 1: "n_candidates"},
                          "scores": {0: "batch", 1: "n_candidates"}},
            opset_version=opset, dynamo=False,
        )
    except (TypeError, RuntimeError):
        batch = torch.export.Dim("batch", min=1, max=4096)
        width = torch.export.Dim("n_candidates", min=2, max=4096)
        torch.onnx.export(
            scorer, (sequences, candidates), str(path),
            input_names=["sequences", "candidates"], output_names=["scores"],
            dynamic_shapes={"sequences": {0: batch},
                            "candidates": {0: batch, 1: width}},
            opset_version=opset, dynamo=True,
        )
    return Path(path)


def quantize_reranker(source, target):
    """int8 weights for MatMul and Gather; activations stay fp32."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gather"])
    return Path(target)


def parity(model, onnx_path, histories, candidates, max_length):
    """How closely the ONNX graph reproduces PyTorch, on real inputs.

    Returns the largest absolute score difference and the share of rows whose
    top 10 among the candidates is the same set under both.
    """
    import onnxruntime as ort

    from .ranking import padded_window

    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    model = model.eval().cpu()
    worst, same_top = 0.0, 0
    for history, row in zip(histories, candidates):
        window = padded_window(history, max_length)
        row = np.asarray(row, dtype=np.int64)[None, :]
        with torch.no_grad():
            expected = model(torch.from_numpy(window),
                             candidates=torch.from_numpy(row)).numpy()[0]
        (got,) = session.run(None, {"sequences": window, "candidates": row})
        got = got[0]
        finite = np.isfinite(expected)
        worst = max(worst, float(np.abs(expected[finite] - got[finite]).max()))
        k = min(10, finite.sum())
        same_top += set(np.argsort(-expected)[:k]) == set(np.argsort(-got)[:k])
    return {"max_abs_diff": worst,
            "top10_agreement": same_top / max(len(histories), 1)}


def write_artifacts(directory, model, user_vectors, item_vectors, histories,
                    popular, max_length, beta, asins=None, parity_users=200,
                    n_candidates=100, seed=0):
    """Everything the service needs, plus the export checks, in `directory`.

    `histories` is {user index: list of item indices}, users numbered from 0
    like the rows of `user_vectors`.
    """
    from .ranking import retrieve

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / "items.npy", np.asarray(item_vectors, np.float32))
    np.save(directory / "users.npy", np.asarray(user_vectors, np.float32))
    np.save(directory / "popular.npy", np.asarray(popular, np.int64))

    n_users = len(user_vectors)
    lengths = np.array([len(histories.get(u, ())) for u in range(n_users)])
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    flat = np.concatenate(
        [np.asarray(histories.get(u, ()), np.int32) for u in range(n_users)]
        or [np.zeros(0, np.int32)]).astype(np.int32)
    np.savez(directory / "histories.npz", flat=flat, offsets=offsets)
    if asins is not None:
        (directory / "asins.json").write_text(json.dumps(list(asins)))

    fp32 = export_reranker(model, directory / "reranker.onnx", max_length,
                           n_candidates)
    int8 = quantize_reranker(fp32, directory / "reranker.int8.onnx")

    # Parity on real users with real candidate rows, as the service sees them.
    rng = np.random.default_rng(seed)
    users = rng.choice(n_users, size=min(parity_users, n_users), replace=False)
    sample_histories = [list(histories.get(int(u), ())) for u in users]
    sample_candidates = []
    for user, history in zip(users, sample_histories):
        candidates, _ = retrieve(np.asarray(user_vectors[user], np.float32),
                                 np.asarray(item_vectors, np.float32),
                                 history, n_candidates)
        sample_candidates.append(candidates)

    meta = {
        "n_items": int(model.n_items),
        "n_users": int(n_users),
        "dimension": int(np.asarray(item_vectors).shape[1]),
        "max_length": int(max_length),
        "beta": float(beta),
        "pad": PAD,
        "size_mb": {
            "reranker.onnx": round(fp32.stat().st_size / 2 ** 20, 1),
            "reranker.int8.onnx": round(int8.stat().st_size / 2 ** 20, 1),
        },
        "parity": {
            "fp32": parity(model, fp32, sample_histories, sample_candidates,
                           max_length),
            "int8": parity(model, int8, sample_histories, sample_candidates,
                           max_length),
            "users": int(len(users)),
        },
    }
    # int8 is served only if it ranks like the model it was quantized from.
    meta["default_reranker"] = (
        "reranker.int8.onnx"
        if meta["parity"]["int8"]["top10_agreement"] >= 0.9
        else "reranker.onnx")
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
