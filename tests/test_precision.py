"""The mixed-precision and compiled training paths train, and stay finite.

The speed they buy is measured by scripts/benchmark_training.py on a GPU;
these only pin that each configuration produces a usable model on CPU.
"""
import pytest
import torch

from recsys.models import BERT4Rec, make_train_step


def batch(n_items=40, size=16, length=8):
    torch.manual_seed(0)
    return (torch.randint(1, n_items, (size, length)),
            torch.randint(1, n_items, (size,)))


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_each_precision_lowers_the_loss(precision):
    sequences, targets = batch()
    model = BERT4Rec(40, dimension=16, n_heads=2, n_layers=1, max_length=8,
                     dropout=0.0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)
    step = make_train_step(model, optimiser, 40, n_negatives=8,
                           precision=precision)
    torch.manual_seed(1)
    first = step(sequences, targets).item()
    for _ in range(30):
        last = step(sequences, targets).item()
    assert torch.isfinite(torch.tensor(last))
    assert last < first
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_an_unknown_precision_is_refused():
    model = BERT4Rec(40, dimension=16, n_heads=2, n_layers=1, max_length=8)
    optimiser = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="precision"):
        make_train_step(model, optimiser, 40, precision="fp8")


def test_fp16_on_cpu_falls_back_without_a_scaler_error():
    """The scaler is only enabled on CUDA; on CPU fp16 autocast still runs."""
    sequences, targets = batch()
    model = BERT4Rec(40, dimension=16, n_heads=2, n_layers=1, max_length=8)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step = make_train_step(model, optimiser, 40, n_negatives=8,
                           precision="fp16")
    assert torch.isfinite(step(sequences, targets))
