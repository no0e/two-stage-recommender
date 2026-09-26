"""LightGCN retrieval and BERT4Rec reranking on an Amazon category.

    python scripts/evaluate.py --category Video_Games
    python scripts/evaluate.py --category Toys_and_Games

Trains both stages, scores them against each other and against popularity on
the same held-out users, and writes docs/amazon_<category>.json. Once both
categories have been run it redraws docs/results.png from the pair.

Every weight is fitted on a split taken one step further back, with models
retrained without those targets, and the test set is touched once at the end.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from recsys.amazon import (  # noqa: E402
    CATEGORIES, FlatSequences, evaluation_sample, load_amazon,
)
from recsys.large import (  # noqa: E402
    popularity_ranked, rerank, retrieve, score_ranked, sequential_topk,
)
from recsys.metrics import bootstrap_difference  # noqa: E402
from recsys.plotting import figure_from_docs  # noqa: E402
from recsys.models import (  # noqa: E402
    WindowDataset, popularity, train_bert4rec, train_lightgcn,
)


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Toys_and_Games",
                        choices=sorted(CATEGORIES))
    parser.add_argument("--data", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--eval-users", type=int, default=20_000)
    parser.add_argument("--max-users", type=int, default=None,
                        help="Subsample users. For smoke tests.")

    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lightgcn-epochs", type=int, default=400)
    parser.add_argument("--lightgcn-batch", type=int, default=500_000)

    parser.add_argument("--bert-epochs", type=int, default=10)
    parser.add_argument("--bert-batch", type=int, default=512)
    parser.add_argument("--negatives", type=int, default=2048,
                        help="Sampled-softmax negatives. 0 uses the full one.")
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def train_both(interactions, args, device, label):
    """The two models, on whichever split is handed in."""
    print(f"\n[{label}] LightGCN: {interactions.n_users:,} users, "
          f"{interactions.n_items - 1:,} items, "
          f"{len(interactions.train):,} training events")
    started = time.time()
    user_vectors, item_vectors = train_lightgcn(
        interactions, epochs=args.lightgcn_epochs, dimension=args.dimension,
        n_layers=args.layers, batch_size=args.lightgcn_batch, device=device,
        verbose=True)
    print(f"[{label}] LightGCN in {time.time() - started:.0f}s")

    store = FlatSequences(interactions, args.max_length)
    print(f"[{label}] BERT4Rec: {len(store):,} training sequences, "
          f"{store.nbytes() / 2 ** 20:.0f} MiB of index")
    started = time.time()
    model = train_bert4rec(
        WindowDataset(store, args.max_length), interactions.n_items,
        epochs=args.bert_epochs, max_length=args.max_length,
        batch_size=args.bert_batch, n_negatives=args.negatives or None,
        workers=args.workers, device=device, verbose=True)
    print(f"[{label}] BERT4Rec in {time.time() - started:.0f}s")
    return user_vectors, item_vectors, model


def main():
    args = parse()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    recall_key = f"recall@{args.k}"
    ndcg_key = f"ndcg@{args.k}"

    interactions = load_amazon(args.category, args.data,
                               max_users=args.max_users)
    n_items = interactions.n_items - 1
    print(json.dumps(interactions.summary(), indent=2))
    print(f"\nDevice: {device}")

    # --------------------------------------------------------- validation
    validation = interactions.without_last(1)
    val_users, val_histories, val_targets = evaluation_sample(
        validation, args.eval_users, seed=1)
    print(f"\nValidation: {len(val_users):,} users")

    val_user_vectors, val_item_vectors, val_model = train_both(
        validation, args, device, "val")

    print("\nSweeping beta on validation ...")
    candidates, values = retrieve(
        val_user_vectors, val_item_vectors, val_users, val_histories,
        k=args.candidates, device=device)

    beta_sweep = {}
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        ranked = rerank(val_model, val_histories, candidates, values,
                        beta=beta, max_length=args.max_length, device=device)
        result, _ = score_ranked(ranked, val_targets, k=args.k)
        beta_sweep[beta] = result[recall_key]
        print(f"  beta={beta:.2f}  recall@{args.k}={beta_sweep[beta]:.4f}")
    best_beta = max(beta_sweep, key=beta_sweep.get)
    print(f"  beta = {best_beta}")

    del val_user_vectors, val_item_vectors, val_model, candidates, values
    if device != "cpu":
        torch.cuda.empty_cache()

    # --------------------------------------------------------------- test
    users, histories, targets = evaluation_sample(
        interactions, args.eval_users, seed=0)
    print(f"\nTest: {len(users):,} users")

    user_vectors, item_vectors, model = train_both(
        interactions, args, device, "test")

    popular = popularity(interactions, top=max(500, args.candidates))
    results, outcomes = {}, {}

    def record(name, ranked):
        result, hits = score_ranked(ranked, targets, k=args.k,
                                    catalogue_size=n_items)
        results[name], outcomes[name] = result, hits
        print(f"  {name:<12} recall@{args.k} {result[recall_key]:.4f}"
              f"   ndcg@{args.k} {result[ndcg_key]:.4f}"
              f"   coverage {result['catalogue_coverage']:.2%}")

    print("\nEvaluating ...")
    record("popularity", popularity_ranked(popular, histories, k=args.k))
    record("bert4rec", sequential_topk(
        model, histories, k=args.k, max_length=args.max_length, device=device))

    lightgcn_only, _ = retrieve(user_vectors, item_vectors, users, histories,
                                k=args.k, device=device)
    record("lightgcn", lightgcn_only)

    candidates, values = retrieve(
        user_vectors, item_vectors, users, histories, k=args.candidates,
        device=device)
    record("two_stage", rerank(model, histories, candidates, values,
                               beta=best_beta, max_length=args.max_length,
                               device=device))

    # ---------------------------------------------------------- intervals
    print("\nPaired against popularity, 2000 bootstrap resamples:")
    comparisons = {}
    for name in results:
        if name == "popularity":
            continue
        comparison = bootstrap_difference(outcomes[name],
                                          outcomes["popularity"])
        comparisons[f"{name}_vs_popularity"] = comparison
        flag = "" if comparison["ci_low"] > 0 else "   (interval spans zero)"
        print(f"  {name:<12} {comparison['difference']:+.4f} "
              f"[{comparison['ci_low']:+.4f}, "
              f"{comparison['ci_high']:+.4f}]{flag}")

    for a, b in (("two_stage", "lightgcn"), ("two_stage", "bert4rec")):
        if a in outcomes and b in outcomes:
            comparison = bootstrap_difference(outcomes[a], outcomes[b])
            comparisons[f"{a}_vs_{b}"] = comparison
            print(f"\n  {a} minus {b}: {comparison['difference']:+.4f} "
                  f"[{comparison['ci_low']:+.4f}, "
                  f"{comparison['ci_high']:+.4f}]")

    summary = {
        "dataset": {"category": args.category, **interactions.summary()},
        "k": args.k,
        "settings": {
            "dimension": args.dimension, "layers": args.layers,
            "lightgcn_epochs": args.lightgcn_epochs,
            "lightgcn_batch": args.lightgcn_batch,
            "bert_epochs": args.bert_epochs, "bert_batch": args.bert_batch,
            "negatives": args.negatives, "candidates": args.candidates,
            "eval_users": len(users), "device": device,
        },
        "beta": best_beta,
        "beta_sweep": {str(b): v for b, v in beta_sweep.items()},
        "results": results,
        "comparisons": comparisons,
        "seconds": round(time.time() - started, 1),
    }

    out = Path(args.out or ROOT / "docs" / f"amazon_{args.category}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out} in {time.time() - started:.0f}s")

    figure = figure_from_docs(out.parent, out.parent / "results.png")
    if figure:
        print(f"Redrew {figure} from both categories.")
    else:
        print("Run the other category too and this will redraw the figure.")


if __name__ == "__main__":
    main()
