"""Train both stages and export what the API serves.

    python scripts/export_serving.py --category Video_Games
    python scripts/export_serving.py --category Toys_and_Games --precision fp16

Writes serving_artifacts/<category>/: the reranker as ONNX (fp32 and int8),
the LightGCN tables, every user's history and the popularity fallback, with
meta.json recording how closely each ONNX graph reproduces PyTorch.

The retrieval stage is read from the cache `recommend.py` writes when it
exists; the reranker is trained here, with the precision and compilation
flags `scripts/benchmark_training.py` measures.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from recsys.amazon import CATEGORIES, FlatSequences, load_amazon  # noqa: E402
from recsys.models import (  # noqa: E402
    WindowDataset,
    popularity,
    train_bert4rec,
    train_lightgcn,
)
from recsys.serving.export import write_artifacts  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Video_Games",
                        choices=sorted(CATEGORIES))
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--lightgcn-epochs", type=int, default=400)
    parser.add_argument("--lightgcn-batch", type=int, default=500_000)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--bert-epochs", type=int, default=8)
    parser.add_argument("--bert-batch", type=int, default=1024)
    parser.add_argument("--negatives", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.75,
                        help="Fitted at 0.75 on both categories.")
    parser.add_argument("--precision", default="fp32",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out or ROOT / "serving_artifacts" / args.category)
    interactions = load_amazon(args.category, args.data,
                               max_users=args.max_users)
    print(f"{args.category}: {interactions.n_users:,} users, "
          f"{interactions.n_items - 1:,} items, "
          f"{len(interactions.events):,} interactions; device {device}")

    suffix = f"_{args.max_users}" if args.max_users else ""
    cache = Path(args.data or ROOT / "data") / \
        f"lightgcn_{args.category}{suffix}.npz"
    stored = np.load(cache) if cache.exists() and not args.retrain else None
    if stored is not None and stored["users"].shape[0] == interactions.n_users:
        print(f"Retrieval stage from {cache.name}")
        user_vectors, item_vectors = stored["users"], stored["items"]
    else:
        started = time.time()
        user_vectors, item_vectors = train_lightgcn(
            interactions, epochs=args.lightgcn_epochs,
            dimension=args.dimension, n_layers=args.layers,
            batch_size=args.lightgcn_batch, device=device)
        print(f"LightGCN in {time.time() - started:.0f}s")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, users=user_vectors, items=item_vectors)

    store = FlatSequences(interactions, args.max_length)
    started = time.time()
    model = train_bert4rec(
        WindowDataset(store, args.max_length), interactions.n_items,
        epochs=args.bert_epochs, max_length=args.max_length,
        batch_size=args.bert_batch, n_negatives=args.negatives or None,
        workers=args.workers, device=device, precision=args.precision,
        compile_model=args.compile)
    print(f"BERT4Rec in {time.time() - started:.0f}s "
          f"({args.precision}{', compiled' if args.compile else ''})")

    # Histories keyed by row of the user table, which is what the service
    # looks a user up by.
    by_id = interactions.indexed_histories()
    histories = {interactions.user_to_index[user]: items
                 for user, items in by_id.items()
                 if user in interactions.user_to_index}
    asin_of = interactions.items.set_index("item_id")["asin"]
    asins = [None] + [str(asin_of[interactions.index_to_item[i]])
                      for i in range(1, interactions.n_items)]

    meta = write_artifacts(
        out, model, user_vectors, item_vectors, histories,
        popularity(interactions, top=500), args.max_length, args.beta,
        asins=asins)
    print(json.dumps(meta, indent=2))
    print(f"\nArtifacts in {out}")


if __name__ == "__main__":
    main()
