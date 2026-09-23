"""BERT4Rec reranking: the second stage.

Three fixes against the version this grew from, all of them silent failures
rather than errors.

Sequences are padded on the **left**. The original padded on the right and then
read the prediction off `x[:, -1, :]`, which for any sequence shorter than the
window is a padding slot. Most sequences are shorter than the window, so most
predictions were being read out of a padding token.

Padding is **masked** in attention. Without the mask every real position attends
to the padding, and with left padding and no mask a short sequence is mostly
attending to nothing.

Padding is **excluded from the loss and from the output**. Index 0 can never be
a target, and scoring it as a candidate would let the model recommend "nothing".
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import PAD


class NextItemDataset(Dataset):
    """(padded history, next item), with the padding on the left."""

    def __init__(self, sequences, max_length=50):
        self.sequences = sequences
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        _, history, target = self.sequences[index]
        history = history[-self.max_length:]
        padding = [PAD] * (self.max_length - len(history))
        return (
            torch.tensor(padding + history, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


class BERT4Rec(nn.Module):
    """A transformer encoder over the item sequence, scoring the next item."""

    def __init__(self, n_items, dimension=128, n_heads=4, n_layers=2,
                 max_length=50, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.max_length = max_length

        self.item_embedding = nn.Embedding(n_items, dimension, padding_idx=PAD)
        self.position_embedding = nn.Embedding(max_length, dimension)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=dimension, nhead=n_heads, dim_feedforward=dimension * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output = nn.Linear(dimension, n_items)

    def forward(self, sequences):
        batch, length = sequences.shape
        positions = torch.arange(length, device=sequences.device)
        x = self.item_embedding(sequences) + self.position_embedding(positions)
        x = self.dropout(x)

        padding_mask = sequences == PAD
        # A row that is entirely padding would make softmax divide by zero, so
        # its last slot is unmasked. It contributes nothing either way, but a
        # NaN here would poison the whole batch.
        padding_mask[padding_mask.all(dim=1), -1] = False

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        logits = self.output(x[:, -1, :])  # left padding, so this is the last
        logits[:, PAD] = float("-inf")     # never recommend nothing
        return logits


def train_bert4rec(sequences, n_items, epochs=20, dimension=128, n_heads=4,
                   n_layers=2, max_length=50, batch_size=128,
                   learning_rate=1e-3, device="cpu", seed=0, verbose=True):
    torch.manual_seed(seed)
    model = BERT4Rec(n_items, dimension, n_heads, n_layers, max_length).to(device)
    loader = DataLoader(
        NextItemDataset(sequences, max_length),
        batch_size=batch_size, shuffle=True,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    best, patience, since_best = float("inf"), 5, 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch_sequences, targets in loader:
            batch_sequences = batch_sequences.to(device)
            targets = targets.to(device)

            loss = criterion(model(batch_sequences), targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item()

        average = total / len(loader)
        if verbose:
            print(f"  BERT4Rec epoch {epoch + 1:2d}  loss {average:.4f}")

        if average < best - 1e-4:
            best, since_best = average, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                if verbose:
                    print(f"  stopped early at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def score_sequences(model, histories, max_length=50, device="cpu",
                    batch_size=256):
    """Logits for a list of histories, in the order they were given."""
    model.eval()
    scores = []
    for start in range(0, len(histories), batch_size):
        chunk = histories[start:start + batch_size]
        padded = [
            [PAD] * (max_length - len(h[-max_length:])) + list(h[-max_length:])
            for h in chunk
        ]
        batch = torch.tensor(padded, dtype=torch.long, device=device)
        scores.append(model(batch).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros((0, model.n_items))
