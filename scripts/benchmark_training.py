"""What mixed precision and torch.compile buy the reranker, measured.

    python scripts/benchmark_training.py                    # Toys and Games size
    python scripts/benchmark_training.py --items 25613      # Video Games size

Times the training step `train_bert4rec` runs (the same `make_train_step`),
at the catalogue size of Toys and Games, under each configuration in turn:
fp32, fp16 autocast with a gradient scaler, bf16 autocast, and each of those
under torch.compile. Batches are random sequences of the real shape, so the
arithmetic is the arithmetic of a real epoch and no download is needed.

For each it reports training samples per second over the timed steps (after
warm-up, which is where compilation happens) and peak GPU memory, and writes
the table to docs/training_benchmark.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from recsys.models import BERT4Rec, make_train_step  # noqa: E402

CONFIGS = [("fp32", False), ("fp16", False), ("bf16", False),
           ("fp32", True), ("fp16", True), ("bf16", True)]


def run(args, precision, compiled, device):
    torch.manual_seed(0)
    model = BERT4Rec(args.items, args.dimension, args.heads, args.layers,
                     args.max_length).to(device).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=0.01)
    step = make_train_step(model, optimiser, args.items,
                           n_negatives=args.negatives, precision=precision,
                           compile_model=compiled)

    sequences = torch.randint(1, args.items, (args.batch, args.max_length),
                              device=device)
    targets = torch.randint(1, args.items, (args.batch,), device=device)

    for _ in range(args.warmup):
        step(sequences, targets)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    for _ in range(args.steps):
        loss = step(sequences, targets)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return {
        "precision": precision,
        "compiled": compiled,
        "samples_per_sec": round(args.batch * args.steps / elapsed, 1),
        "step_ms": round(1e3 * elapsed / args.steps, 2),
        "peak_memory_mib": (round(torch.cuda.max_memory_allocated() / 2 ** 20)
                            if device == "cuda" else None),
        "final_loss_finite": bool(torch.isfinite(loss)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=162_036,
                        help="Catalogue size including PAD (Toys and Games).")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--negatives", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--configs", default=None,
                        help="Comma list like fp32,fp16,fp16+compile")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(ROOT / "docs"
                                             / "training_benchmark.json"))
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    configs = CONFIGS
    if args.configs:
        configs = [(c.split("+")[0], c.endswith("+compile"))
                   for c in args.configs.split(",")]

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"{gpu}, torch {torch.__version__}, {args.items:,} items, "
          f"batch {args.batch}, {args.negatives} negatives\n")

    rows = []
    for precision, compiled in configs:
        name = precision + (" + compile" if compiled else "")
        try:
            row = run(args, precision, compiled, device)
        except Exception as error:  # a config the hardware cannot run
            print(f"  {name:16s} failed: {type(error).__name__}: {error}")
            continue
        rows.append(row)
        print(f"  {name:16s} {row['samples_per_sec']:>10,.0f} samples/s  "
              f"{row['step_ms']:>8.1f} ms/step  "
              f"{row['peak_memory_mib'] or 0:>7,} MiB")
        if device == "cuda":
            torch.cuda.empty_cache()

    baseline = next((r for r in rows
                     if r["precision"] == "fp32" and not r["compiled"]), None)
    if baseline:
        for row in rows:
            row["speedup_vs_fp32"] = round(
                row["samples_per_sec"] / baseline["samples_per_sec"], 2)
            if baseline["peak_memory_mib"]:
                row["memory_saved_vs_fp32"] = round(
                    1 - row["peak_memory_mib"] / baseline["peak_memory_mib"], 3)

    result = {"device": gpu, "torch": torch.__version__,
              "items": args.items, "batch": args.batch,
              "negatives": args.negatives, "max_length": args.max_length,
              "steps": args.steps, "warmup": args.warmup, "results": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
