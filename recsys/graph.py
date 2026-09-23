"""LightGCN retrieval: the first stage.

Written against a sparse adjacency matrix rather than torch_geometric. LightGCN
propagation is one normalised sparse matrix multiply per layer and nothing else,
so pulling in a graph library for it costs an install that frequently fails and
buys a single line. The arithmetic below is the same.

The other change from the original is the negative sampling. It drew negatives
with `torch.randint(0, num_items, ...)` over a node index where items live at
`[n_users, n_users + n_items)`, so every negative was a *user* node. The BPR
loss was pushing item embeddings away from user embeddings picked at random,
which is not the objective anyone intended.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
