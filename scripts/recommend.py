"""Recommend ten films for one user, and show what they actually watched next.

    python scripts/recommend.py --user 42
    python scripts/recommend.py --user 42 --rerank

The retrieval stage alone takes a second or two on a laptop: one matrix inverse
over the item-item matrix, then the user's history weighted towards what they
did most recently. That is the default, and it is enough to see the system
answer.

`--rerank` adds the second stage, which trains BERT4Rec first and so takes a
few minutes on a CPU. It then prints the reranked ten under the retrieved
ten, which is the clearest picture of what the second stage changes.

The last film of every user is held out, so the one they really watched next is
known and printed underneath. It lands in the top ten about 18% of the time.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recsys.data import load, training_sequences  # noqa: E402
from recsys.models import (  # noqa: E402
    ContentRecommender, RecencyEASE, build_content_matrix, popularity,
    train_bert4rec,
)
from recsys.pipeline import TwoStageRecommender  # noqa: E402

FITTED = ROOT / "docs" / "results.json"


def titles(interactions):
    """Item index to title, so the output reads as films and not integers."""
    lookup = interactions.items.set_index("item_id")["title"].to_dict()
    return {
        index: lookup.get(item_id, f"item {item_id}")
        for index, item_id in interactions.index_to_item.items()
    }


def fitted_weights():
    """alpha and beta as the last full evaluation chose them.

    Read from the results file rather than written in here, so the demo cannot
    quietly use different weights from the ones the table reports.
    """
    if FITTED.exists():
        saved = json.loads(FITTED.read_text(encoding="utf-8"))
        return saved.get("alpha", 1.0), saved.get("beta", 0.5)
    return 1.0, 0.5


def show(name, ranked, title_of, target):
    print(f"\n{name}")
    for rank, index in enumerate(ranked, start=1):
        marker = "  <- the one they watched" if index == target else ""
        print(f"  {rank:>2}  {title_of.get(index, index)}{marker}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", type=int, default=12)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--data", default=None)
    parser.add_argument("--rerank", action="store_true",
                        help="Add the second stage. Trains BERT4Rec first.")
    parser.add_argument("--bert-epochs", type=int, default=30)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    interactions = load(args.data)
    title_of = titles(interactions)
    histories = interactions.indexed_histories()

    history = histories.get(args.user)
    if not history:
        known = sorted(histories)
        raise SystemExit(
            f"No history for user {args.user}. This dataset has users "
            f"{known[0]} to {known[-1]}."
        )

    held_out = interactions.test[interactions.test["user_id"] == args.user]
    target = (interactions.item_to_index.get(held_out["item_id"].iloc[0])
              if len(held_out) else None)

    print(f"MovieLens 100k: {interactions.n_users:,} users, "
          f"{interactions.n_items - 1:,} films.")
    print(f"\nUser {args.user}, last five of {len(history)} films watched")
    for index in history[-5:]:
        print(f"      {title_of.get(index, index)}")

    started = time.time()
    ease = RecencyEASE(interactions)
    print(f"\nRetrieval stage fitted in {time.time() - started:.1f}s.")

    retrieved = ease(args.user, history, k=args.k)
    show(f"Top {args.k}, retrieval only", retrieved, title_of, target)

    if args.rerank:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        alpha, beta = fitted_weights()
        print(f"\nTraining the reranker on {device}, alpha={alpha}, "
              f"beta={beta} ...")

        started = time.time()
        model = train_bert4rec(
            training_sequences(interactions), interactions.n_items,
            epochs=args.bert_epochs, device=device, verbose=False)
        print(f"  BERT4Rec in {time.time() - started:.0f}s")

        popular = popularity(interactions)
        two_stage = TwoStageRecommender(
            ease, ContentRecommender(build_content_matrix(interactions),
                                     popular),
            model, alpha=alpha, beta=beta, device=device)
        show(f"Top {args.k}, both stages", two_stage(args.user, history, args.k),
             title_of, target)

    if target is None:
        print("\nThis user has nothing held out.")
    else:
        print(f"\nThey actually watched: {title_of.get(target, target)}")
        print("It lands in the top ten for about 18% of users; "
              "recall@10 over all 943 is 0.176.")


if __name__ == "__main__":
    main()
