"""The two models the pipeline runs, in the order it runs them.

`train_lightgcn` is the retrieval stage. User and item embeddings are smoothed
over the interaction graph by a normalised sparse adjacency, one matrix
multiply per layer. `torch_geometric` is not a dependency: a graph library
would have bought a single line and cost an install that frequently fails.

`train_bert4rec` is the reranking stage: a small transformer over the user's
sequence, which is the only model here that knows what follows what. It scores
a handful of candidates rather than the catalogue, and trains against a sampled
softmax, because at 162,035 items the output layer is otherwise the whole cost
of a step.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import PAD


def popularity(interactions, top=500):
    """Item indices by training frequency. The baseline, and the last resort."""
    counts = interactions.train["item_id"].value_counts()
    ranked = [
        interactions.item_to_index[item]
        for item in counts.index if item in interactions.item_to_index
    ]
    return ranked[:top]


# ------------------------------------------ LightGCN, the first stage
def build_adjacency(interactions, device="cpu"):
    """The symmetric, degree-normalised user-item graph as a sparse tensor.

    Nodes are users first, then items, so item `i` is node `n_users + i`.
    """
    train = interactions.train
    pairs = train[["user_id", "item_id"]].drop_duplicates()
    users = pairs["user_id"].map(interactions.user_to_index)
    items = pairs["item_id"].map(interactions.item_to_index)

    keep = users.notna() & items.notna()
    users = users[keep].to_numpy(dtype=np.int64)
    items = items[keep].to_numpy(dtype=np.int64) + interactions.n_users

    rows = np.concatenate([users, items])
    columns = np.concatenate([items, users])
    n_nodes = interactions.n_users + interactions.n_items

    degrees = np.bincount(rows, minlength=n_nodes).astype(np.float32)
    # An isolated node keeps a degree of one so its normalisation stays finite;
    # it has no edges, so the value never reaches anything.
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1.0))
    values = inverse_sqrt[rows] * inverse_sqrt[columns]

    indices = torch.tensor(np.stack([rows, columns]), dtype=torch.long)
    adjacency = torch.sparse_coo_tensor(
        indices, torch.tensor(values, dtype=torch.float32),
        (n_nodes, n_nodes),
    ).coalesce()
    return adjacency.to(device), n_nodes


class LightGCN(nn.Module):
    """Embeddings smoothed over the interaction graph.

    No weights beyond the embedding table itself, which is the point of
    LightGCN: the layers have nothing to learn, they only average a node's
    neighbourhood, and the final representation is the mean across depths.
    """

    def __init__(self, n_nodes, n_users, dimension=64, n_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, dimension)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.n_layers = n_layers
        self.n_users = n_users

    def forward(self, adjacency):
        x = self.embedding.weight
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adjacency, x)
            layers.append(x)
        return torch.stack(layers).mean(dim=0)

    def split(self, adjacency):
        """(user embeddings, item embeddings), item index aligned with PAD at 0."""
        nodes = self.forward(adjacency)
        return nodes[:self.n_users], nodes[self.n_users:]


def bpr_loss(user_embeddings, positive, negative, weight_decay=1e-4):
    """Bayesian personalised ranking: the observed item outranks a random one."""
    positive_scores = (user_embeddings * positive).sum(dim=1)
    negative_scores = (user_embeddings * negative).sum(dim=1)
    ranking = -F.logsigmoid(positive_scores - negative_scores).mean()
    regularisation = weight_decay * (
        user_embeddings.pow(2).sum()
        + positive.pow(2).sum()
        + negative.pow(2).sum()
    ) / len(user_embeddings)
    return ranking + regularisation


def sample_negatives(users, n_items, observed_codes, generator, rounds=4):
    """One unobserved item per user, sampled without a Python loop.

    The obvious implementation walks the batch and resamples inside a `while`,
    which on a hundred thousand edges is most of the training time. Here the
    observed pairs are encoded as a single sorted integer array, membership is
    one `searchsorted`, and only the colliding entries are redrawn. A handful of
    rounds leaves a negligible number of collisions, and a collision is a
    false negative rather than a crash.
    """
    negatives = generator.integers(1, n_items, size=len(users))
    for _ in range(rounds):
        codes = users * n_items + negatives
        position = np.searchsorted(observed_codes, codes)
        position = np.clip(position, 0, len(observed_codes) - 1)
        collides = observed_codes[position] == codes
        if not collides.any():
            break
        negatives[collides] = generator.integers(
            1, n_items, size=int(collides.sum()))
    return negatives


def train_lightgcn(interactions, epochs=300, dimension=64, n_layers=3,
                   learning_rate=1e-2, weight_decay=1e-4, batch_size=None,
                   device="cpu", seed=0, verbose=True):
    """Fit the retrieval stage and return the two embedding tables.

    Full batch by default. Every step needs the propagation recomputed over the
    whole graph, so a mini-batch costs the same as a full one and buys only a
    noisier gradient; with `batch_size=None` there is one propagation per epoch
    instead of one per mini-batch.
    """
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)

    adjacency, n_nodes = build_adjacency(interactions, device)
    model = LightGCN(n_nodes, interactions.n_users, dimension, n_layers).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    pairs = interactions.train[["user_id", "item_id"]].drop_duplicates()
    keep = pairs["user_id"].isin(interactions.user_to_index) \
        & pairs["item_id"].isin(interactions.item_to_index)
    pairs = pairs[keep]
    user_index = pairs["user_id"].map(interactions.user_to_index).to_numpy(np.int64)
    item_index = pairs["item_id"].map(interactions.item_to_index).to_numpy(np.int64)

    # Sampling a negative the user has actually seen teaches the model the
    # opposite of the truth, so the observed pairs are held as sorted codes.
    observed_codes = np.sort(user_index * interactions.n_items + item_index)

    users = torch.tensor(user_index, device=device)
    items = torch.tensor(item_index, device=device)
    size = batch_size or len(users)

    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(users))
        total, steps = 0.0, 0

        for start in range(0, len(order), size):
            chunk = order[start:start + size]
            batch_users = users[chunk]
            negatives = torch.tensor(
                sample_negatives(user_index[chunk], interactions.n_items,
                                 observed_codes, generator),
                device=device,
            )

            user_vectors, item_vectors = model.split(adjacency)
            loss = bpr_loss(
                user_vectors[batch_users], item_vectors[items[chunk]],
                item_vectors[negatives], weight_decay=weight_decay,
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item()
            steps += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  LightGCN epoch {epoch + 1:4d}  "
                  f"BPR loss {total / max(steps, 1):.4f}")

    model.eval()
    with torch.no_grad():
        user_vectors, item_vectors = model.split(adjacency)
    return user_vectors.cpu().numpy(), item_vectors.cpu().numpy()


# ---------------------------------------- BERT4Rec, the reranking stage
class WindowDataset(Dataset):
    """Pads whatever a source yields as (window, target) to a fixed length.

    The flat store in `recsys.amazon` hands out a numpy view into one shared
    array rather than a copied list, and this is what turns it into the padded
    tensor the model reads.
    """

    def __init__(self, source, max_length=50):
        self.source = source
        self.max_length = max_length

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        window, target = self.source[index]
        window = np.asarray(window, dtype=np.int64)[-self.max_length:]
        # PAD is 0, so an empty row is already padded.
        padded = np.zeros(self.max_length, dtype=np.int64)
        if len(window):
            padded[-len(window):] = window
        return torch.from_numpy(padded), torch.tensor(target, dtype=torch.long)


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

    def forward(self, sequences, candidates=None):
        """Logits over the whole catalogue, or only over `candidates`.

        `candidates` is (batch, n) of item indices. With it, the output layer
        is gathered per row and multiplied against the pooled state, which
        turns the cost from the size of the catalogue into the size of the
        candidate set. Two places need that: reranking reads 100 items per
        user, and a sampled softmax reads a target plus its negatives. Scoring
        20,000 users over 162,000 items would otherwise ask for 13 GB of
        logits that nothing looks at.
        """
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

        pooled = x[:, -1, :]  # left padding, so the last slot is a real item

        if candidates is None:
            logits = self.output(pooled)
            logits[:, PAD] = float("-inf")  # never recommend nothing
            return logits

        weights = self.output.weight[candidates]          # (batch, n, dim)
        biases = self.output.bias[candidates]             # (batch, n)
        scores = torch.bmm(weights, pooled.unsqueeze(-1)).squeeze(-1) + biases
        return scores.masked_fill(candidates == PAD, float("-inf"))


def train_bert4rec(dataset, n_items, epochs=20, dimension=128, n_heads=4,
                   n_layers=2, max_length=50, batch_size=128,
                   learning_rate=1e-3, device="cpu", seed=0, verbose=True,
                   n_negatives=None, workers=0):
    """Fit the reranker.

    `n_negatives` turns the loss into a sampled softmax: the target is scored
    against that many uniform negatives instead of against the whole
    catalogue. At 1,682 items the full softmax is free and the default leaves
    it alone. At 162,000 items it is the dominant cost of every step, and
    scoring a target against 2,048 negatives is eighty times less arithmetic
    for a gradient that points the same way.

    Negatives are drawn uniformly and are not checked against the target. A
    collision makes one example unlearnable rather than wrong, and at this
    catalogue size it happens to about one row in eighty.
    """
    torch.manual_seed(seed)
    model = BERT4Rec(n_items, dimension, n_heads, n_layers, max_length).to(device)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=(workers > 0 and device != "cpu"),
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    # The sampled loss needs its own criterion. Its label is always column 0,
    # the position the target is placed at, and PAD is also 0: the shared
    # criterion would call every single row an ignored one and average over an
    # empty set, which is a NaN loss and a model of NaN weights.
    sampled_criterion = nn.CrossEntropyLoss()

    best, patience, since_best = float("inf"), 5, 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch_sequences, targets in loader:
            batch_sequences = batch_sequences.to(device)
            targets = targets.to(device)

            if n_negatives:
                negatives = torch.randint(
                    1, n_items, (len(targets), n_negatives), device=device)
                candidates = torch.cat(
                    [targets.unsqueeze(1), negatives], dim=1)
                # The target is column 0 of every row, so the label is 0.
                loss = sampled_criterion(
                    model(batch_sequences, candidates=candidates),
                    torch.zeros(len(targets), dtype=torch.long, device=device),
                )
            else:
                loss = criterion(model(batch_sequences), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "The reranker loss is not finite on the first steps of "
                    f"epoch {epoch + 1}. Training from here would only "
                    "produce NaN weights, and a model of NaN weights still "
                    "returns a ranking, so this stops rather than reports one."
                )

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
                    batch_size=256, candidates=None):
    """Logits for a list of histories, in the order they were given.

    With `candidates`, a list of per-history index lists of equal length, only
    those columns are computed and the result is (n_histories, n_candidates)
    instead of (n_histories, n_items).
    """
    model.eval()
    scores = []
    for start in range(0, len(histories), batch_size):
        chunk = histories[start:start + batch_size]
        padded = [
            [PAD] * (max_length - len(h[-max_length:])) + list(h[-max_length:])
            for h in chunk
        ]
        batch = torch.tensor(padded, dtype=torch.long, device=device)

        if candidates is None:
            scores.append(model(batch).cpu().numpy())
        else:
            columns = torch.tensor(
                candidates[start:start + batch_size], dtype=torch.long,
                device=device)
            scores.append(model(batch, candidates=columns).cpu().numpy())

    if scores:
        return np.concatenate(scores)
    width = model.n_items if candidates is None else 0
    return np.zeros((0, width))
