"""Train every system and measure them against each other and against popularity.

    python scripts/evaluate.py
    python scripts/evaluate.py --bert-epochs 5        # a quick pass

Every system is scored on the same evaluation triples, so the comparisons are
paired and the intervals mean something. Every hyperparameter is fitted on a
split taken one step further back than the test split, using models that never
saw a test target, and the test set is touched once at the end.

On Windows, set OMP_NUM_THREADS=1 before running. Some BLAS builds deadlock on
matrix inversions above roughly 800 by 800, which is smaller than the item-item
matrix this fits.

Writes docs/results.json and redraws docs/results.png from it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from recsys.data import load, training_sequences  # noqa: E402
from recsys.metrics import bootstrap_difference, evaluate, hits  # noqa: E402
from recsys.models import (  # noqa: E402
    ContentRecommender, RecencyEASE, build_content_matrix, popularity,
    train_bert4rec, train_lightgcn,
)
from recsys.pipeline import (  # noqa: E402
    GraphRecommender, PopularityRecommender, RetrievalBlend,
    SequentialRecommender, TwoStageRecommender, fit_alpha_on_retrieval, fit_beta,
)
from recsys.plotting import results_figure  # noqa: E402


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None)
    parser.add_argument("--protocol", default="leave_one_out",
                        choices=["leave_one_out", "temporal"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--encoder", default="tfidf", choices=["tfidf", "sbert"])
    parser.add_argument("--min-rating", type=int, default=1)
    parser.add_argument("--lightgcn-epochs", type=int, default=300)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--l2", type=float, default=100.0,
                        help="EASE regularisation, chosen on validation.")
    parser.add_argument("--half-life", type=float, default=20.0,
                        help="Recency half life, in interactions.")
    parser.add_argument("--bert-epochs", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(ROOT / "docs" / "results.json"))
    return parser.parse_args()


def main():
    args = parse()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()

    interactions = load(args.data, protocol=args.protocol,
                        min_rating=args.min_rating)
    print(json.dumps(interactions.summary(), indent=2))
    print(f"\nDevice: {device}\n")

    validation = interactions.without_last(1)
    validation_pairs = validation.evaluation_pairs()

    print("Fitting the retrieval models ...")
    t = time.time()
    ease = RecencyEASE(interactions, l2=args.l2, half_life=args.half_life)
    print(f"  EASE in {time.time() - t:.1f}s")

    t = time.time()
    user_vectors, item_vectors = train_lightgcn(
        interactions, epochs=args.lightgcn_epochs, dimension=args.dimension,
        n_layers=args.layers, device=device, verbose=False)
    print(f"  LightGCN in {time.time() - t:.0f}s")

    print("\nTraining BERT4Rec ...")
    sequences = training_sequences(interactions, args.max_length)
    print(f"  {len(sequences):,} training sequences")
    t = time.time()
    model = train_bert4rec(
        sequences, interactions.n_items, epochs=args.bert_epochs,
        max_length=args.max_length, device=device, verbose=False)
    print(f"  BERT4Rec in {time.time() - t:.0f}s")

    popular = popularity(interactions)
    cold = ContentRecommender(
        build_content_matrix(interactions, encoder=args.encoder), popular)

    popularity_model = PopularityRecommender(popular)
    graph_model = GraphRecommender(
        interactions, user_vectors, item_vectors, popularity_model)
    sequential_model = SequentialRecommender(model, args.max_length, device)

    # ----------------------------------------------------------- validation
    print("\nFitting the blends on validation ...")
    print("  refitting the retrieval models without the validation targets")
    val_ease = RecencyEASE(validation, l2=args.l2, half_life=args.half_life)
    val_popular = popularity(validation)
    val_cold = ContentRecommender(
        build_content_matrix(validation, encoder=args.encoder), val_popular)

    blend = RetrievalBlend(val_ease, val_cold, n_candidates=args.candidates)
    best_alpha, alpha_sweep = fit_alpha_on_retrieval(blend, validation_pairs)
    print(f"  alpha = {best_alpha}")

    print("  a second BERT4Rec, for the rerank weight")
    val_model = train_bert4rec(
        training_sequences(validation, args.max_length), validation.n_items,
        epochs=args.bert_epochs, max_length=args.max_length, device=device,
        verbose=False)
    val_two_stage = TwoStageRecommender(
        val_ease, val_cold, val_model, alpha=best_alpha,
        n_candidates=args.candidates, max_length=args.max_length, device=device)
    best_beta, beta_sweep = fit_beta(val_two_stage, validation_pairs, k=args.k)
    print(f"  beta = {best_beta}")

    # ----------------------------------------------------------------- test
    two_stage = TwoStageRecommender(
        ease, cold, model, alpha=best_alpha, beta=best_beta,
        n_candidates=args.candidates, max_length=args.max_length, device=device)

    pairs = interactions.evaluation_pairs()
    print(f"\nEvaluating on {len(pairs):,} held-out users ...\n")

    systems = {
        "popularity": popularity_model,
        "bert4rec": sequential_model,
        "lightgcn": graph_model,
        "ease": ease,
        "two_stage": two_stage,
    }

    results, outcomes = {}, {}
    for name, system in systems.items():
        results[name] = evaluate(
            system, pairs, k=args.k, catalogue_size=interactions.n_items - 1)
        outcomes[name] = hits(system, pairs, k=args.k)
        print(
            f"  {name:<12} recall@{args.k} {results[name][f'recall@{args.k}']:.4f}"
            f"   ndcg@{args.k} {results[name][f'ndcg@{args.k}']:.4f}"
            f"   coverage {results[name]['catalogue_coverage']:.1%}"
        )

    print("\nPaired against popularity, 2000 bootstrap resamples:")
    comparisons = {}
    for name in systems:
        if name == "popularity":
            continue
        comparison = bootstrap_difference(outcomes[name], outcomes["popularity"])
        comparisons[f"{name}_vs_popularity"] = comparison
        flag = "" if comparison["ci_low"] > 0 else "   (interval spans zero)"
        print(f"  {name:<12} {comparison['difference']:+.4f} "
              f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]{flag}")

    for a, b in (("two_stage", "ease"), ("ease", "lightgcn")):
        comparison = bootstrap_difference(outcomes[a], outcomes[b])
        comparisons[f"{a}_vs_{b}"] = comparison
        print(f"\n  {a} minus {b}: {comparison['difference']:+.4f} "
              f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]")

    summary = {
        "dataset": interactions.summary(),
        "k": args.k,
        "settings": {
            "encoder": args.encoder, "l2": args.l2,
            "half_life": args.half_life, "dimension": args.dimension,
            "layers": args.layers, "lightgcn_epochs": args.lightgcn_epochs,
            "bert_epochs": args.bert_epochs, "candidates": args.candidates,
        },
        "alpha": best_alpha,
        "alpha_sweep": {str(a): v for a, v in alpha_sweep.items()},
        "beta": best_beta,
        "beta_sweep": {str(b): v for b, v in beta_sweep.items()},
        "results": results,
        "comparisons": comparisons,
        "seconds": round(time.time() - started, 1),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure = results_figure(summary, out.parent / "results.png")
    print(f"\nWrote {out} and {figure} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
