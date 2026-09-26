"""Recommend ten items for one user, and show what they actually bought next.

    python scripts/recommend.py
    python scripts/recommend.py --category Video_Games --user 12 --rerank

The retrieval stage has to be trained before it can answer, which is a few
minutes on a GPU and longer on a CPU. It is trained once and the embeddings
are cached under data/, so every run after the first is immediate.

The 5-core files carry interactions, not product names, so items come out as
their ASIN with the link that resolves it. The last item of every user is held
out, so the one they really bought next is known and printed underneath.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from recsys.amazon import CATEGORIES, FlatSequences, load_amazon  # noqa: E402
from recsys.large import rerank, retrieve  # noqa: E402
from recsys.models import (  # noqa: E402
    WindowDataset, train_bert4rec, train_lightgcn,
)

LINK = "https://www.amazon.com/dp/{}"


def embeddings(interactions, category, args, device):
    """The trained retrieval stage, from cache when there is one."""
    cache = Path(args.data or ROOT / "data") / f"lightgcn_{category}.npz"
    if cache.exists() and not args.retrain:
        stored = np.load(cache)
        if stored["users"].shape[0] == interactions.n_users:
            print(f"Loaded the retrieval stage from {cache.name}.")
            return stored["users"], stored["items"]
        print(f"{cache.name} was fitted on a different split; refitting.")

    print(f"Training LightGCN on {len(interactions.train):,} interactions, "
          f"{args.epochs} epochs on {device} ...")
    started = time.time()
    user_vectors, item_vectors = train_lightgcn(
        interactions, epochs=args.epochs, dimension=args.dimension,
        n_layers=args.layers, batch_size=args.batch, device=device,
        verbose=True)
    print(f"  fitted in {time.time() - started:.0f}s")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, users=user_vectors, items=item_vectors)
    print(f"  cached in {cache.name}; the next run will not retrain.")
    return user_vectors, item_vectors


def pick_user(interactions, wanted, min_history=8):
    """The user asked for, or the first held-out one with a real history."""
    histories = interactions.indexed_histories()
    if wanted is not None:
        history = histories.get(wanted)
        if not history:
            known = sorted(histories)
            raise SystemExit(
                f"No history for user {wanted}. This category has users "
                f"{known[0]} to {known[-1]}."
            )
        return wanted, history

    for user in interactions.test["user_id"]:
        history = histories.get(int(user))
        if history and len(history) >= min_history:
            return int(user), history
    raise SystemExit("No held-out user has a history worth showing.")


def show(name, ranked, asin_of, target):
    print(f"\n{name}")
    for rank, index in enumerate(ranked, start=1):
        asin = asin_of[int(index)]
        mark = "   <- the one they bought" if int(index) == int(target) else ""
        print(f"  {rank:>2}  {asin}  {LINK.format(asin)}{mark}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Video_Games",
                        choices=sorted(CATEGORIES))
    parser.add_argument("--user", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch", type=int, default=500_000)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--rerank", action="store_true",
                        help="Add the second stage. Trains BERT4Rec first.")
    parser.add_argument("--bert-epochs", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.75,
                        help="Fitted at 0.75 on both categories.")
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    interactions = load_amazon(args.category, args.data)
    asin_of = {index: interactions.items["asin"].iloc[item]
               for item, index in interactions.item_to_index.items()}

    print(f"{args.category.replace('_', ' ')}: "
          f"{interactions.n_users:,} users, "
          f"{interactions.n_items - 1:,} items, "
          f"{len(interactions.events):,} interactions.")

    user, history = pick_user(interactions, args.user)
    index = interactions.user_to_index[user]

    held_out = interactions.test[interactions.test["user_id"] == user]
    target = (interactions.item_to_index.get(held_out["item_id"].iloc[0])
              if len(held_out) else None)

    user_vectors, item_vectors = embeddings(
        interactions, args.category, args, device)

    print(f"\nUser {user}, last five of {len(history)} items bought")
    for item in history[-5:]:
        print(f"      {asin_of[int(item)]}")

    candidates, values = retrieve(
        user_vectors, item_vectors, [index], [history],
        k=args.candidates if args.rerank else args.k, device=device)
    show(f"Top {args.k}, retrieval only", candidates[0][:args.k], asin_of,
         target)

    if args.rerank:
        print(f"\nTraining the reranker on {device}, beta={args.beta} ...")
        started = time.time()
        store = FlatSequences(interactions, args.max_length)
        model = train_bert4rec(
            WindowDataset(store, args.max_length), interactions.n_items,
            epochs=args.bert_epochs, max_length=args.max_length,
            batch_size=1024, n_negatives=2048, workers=4, device=device,
            verbose=True)
        print(f"  BERT4Rec in {time.time() - started:.0f}s")

        reranked = rerank(model, [history], candidates, values,
                          beta=args.beta, max_length=args.max_length,
                          device=device)
        show(f"Top {args.k}, both stages", reranked[0][:args.k], asin_of,
             target)

    if target is None:
        print("\nThis user has nothing held out.")
    else:
        print(f"\nThey actually bought: {asin_of[int(target)]}")
        print("It lands in the top ten for about 9% of users on Video Games; "
              "recall@10 over 20,000 of them is 0.0860.")


if __name__ == "__main__":
    main()
