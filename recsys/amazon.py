"""Amazon Reviews 2023, at a size where the shortcuts stop working.

MovieLens has 1,682 items, which is small enough that everything fits in a
dense array and a closed-form item-item model is the best thing in the
repository. That stops being true quickly. The item-item matrix EASE inverts
grows with the square of the catalogue and the inverse with its cube:

    catalogue     Gram matrix (float64)
        1,682                    23 MB
       25,612                   5.2 GB
      162,035                   196 GiB

The last row is the point. On the machine this was run on, 200 GiB is the
whole memory limit, and `np.linalg.inv` needs a second array the same size as
the first. So above a certain catalogue the choice is not which model scores
better, it is which model runs at all.

This module loads the 5-core benchmark files published by the McAuley lab at
UCSD. They are public, one file per category, and already filtered so every
user and every item has at least five interactions.

    https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/
"""
import gzip
import shutil
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR, Interactions

BASE = ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
        "/benchmark/5core/rating_only")

CATEGORIES = {
    # category: (interactions, users, items) as published, for the record
    "Video_Games": (814_586, 94_762, 25_612),
    "Toys_and_Games": (3_861_886, 432_264, 162_035),
    "Office_Products": (1_243_186, 223_213, 77_551),
    "Baby_Products": (943_073, 152_296, 35_504),
}


def download(category, data_dir=None):
    """Fetch one category's 5-core ratings. Returns the local path."""
    target = Path(data_dir or DATA_DIR) / "amazon"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{category}.csv.gz"
    if path.exists():
        return path

    url = f"{BASE}/{category}.csv.gz"
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=600, context=context) as response:
        with open(path, "wb") as out:
            shutil.copyfileobj(response, out)
    print(f"  {path}, {path.stat().st_size / 1e6:.0f} MB")
    return path


def load_amazon(category="Toys_and_Games", data_dir=None, min_history=4,
                max_users=None, seed=0):
    """One category as an `Interactions`, same object the MovieLens path uses.

    The columns are renamed to match, and a stub catalogue frame is built from
    the item ids: the rating-only files carry no titles, and nothing in the
    graph path reads them. `max_users` subsamples users, which is how a run is
    made small enough to debug without changing any other code.
    """
    if category not in CATEGORIES:
        print(f"Note: {category} is not one of the sizes recorded here; "
              "it will still load if the file exists.")

    path = download(category, data_dir)
    events = pd.read_csv(gzip.open(path, "rt"),
                         dtype={"user_id": str, "parent_asin": str})
    events = events.rename(columns={"parent_asin": "item_id"})
    events = events[["user_id", "item_id", "timestamp"]]

    if max_users is not None:
        users = events["user_id"].unique()
        chosen = np.random.default_rng(seed).choice(
            users, size=min(max_users, len(users)), replace=False)
        events = events[events["user_id"].isin(set(chosen))]

    # The item ids are ASIN strings. Interned to a small integer range here so
    # every downstream structure is an int array rather than 3.9M Python
    # strings held twice over.
    codes, uniques = pd.factorize(events["item_id"], sort=True)
    events = events.assign(item_id=codes.astype(np.int64))
    items = pd.DataFrame({
        "item_id": np.arange(len(uniques), dtype=np.int64),
        "asin": uniques,
        "title": uniques,
        "genres": "",
        "year": "",
        "text": uniques,
    })

    user_codes, _ = pd.factorize(events["user_id"], sort=True)
    events = events.assign(user_id=user_codes.astype(np.int64))

    return Interactions(events, items, protocol="leave_one_out",
                        min_history=min_history)


class FlatSequences:
    """Every prefix of every history, without copying any of them.

    `training_sequences` builds one Python list per prefix. On MovieLens that
    is 99,000 small lists and nobody notices. On a few million interactions it
    is the largest object in the process, because the prefixes of one history
    of length n hold n(n+1)/2 integers between them.

    Here the histories are concatenated into one int32 array and a prefix is
    two offsets into it. Memory is the number of interactions, not the sum of
    the squares of the history lengths, and the padded window is cut straight
    out of the flat array.

    One behavioural difference, and it is not incidental. `training_sequences`
    truncates a history to `max_length` and then takes its prefixes, so a user
    with more interactions than the window contributes nothing from before it.
    Here the window bounds what the model reads and every position after the
    first is still a target. On these Amazon categories the median history is
    seven against a window of fifty, so the two agree for almost every user;
    on a dataset of long histories they would not.
    """

    def __init__(self, interactions, max_length=50):
        self.max_length = max_length
        histories = interactions.indexed_histories()

        flat, starts, ends = [], [], []
        offset = 0
        for history in histories.values():
            if len(history) < 2:
                continue
            flat.extend(history)
            # One example per position after the first: predict history[cut]
            # from everything before it.
            for cut in range(1, len(history)):
                starts.append(offset)
                ends.append(offset + cut)
            offset += len(history)

        self.flat = np.asarray(flat, dtype=np.int32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.ends = np.asarray(ends, dtype=np.int64)

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, index):
        start, end = self.starts[index], self.ends[index]
        window = self.flat[max(start, end - self.max_length):end]
        target = int(self.flat[end])
        return window, target

    def nbytes(self):
        return self.flat.nbytes + self.starts.nbytes + self.ends.nbytes


def evaluation_sample(interactions, n_users=20_000, seed=0, min_history=1):
    """(users, histories, targets) for a random sample of held-out users.

    `Interactions.evaluation_pairs` groups the test frame by user, which is a
    few hundred thousand pandas groups here and takes minutes. This walks two
    aligned arrays instead, and samples because recall over twenty thousand
    users already has a standard error near a thousandth.
    """
    histories = interactions.indexed_histories()
    test = interactions.test[["user_id", "item_id"]].to_numpy()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(test))

    users, prefixes, targets = [], [], []
    for row in order:
        user, item = int(test[row][0]), test[row][1]
        history = histories.get(user)
        index = interactions.item_to_index.get(item)
        if index is None or history is None or len(history) < min_history:
            continue
        users.append(interactions.user_to_index.get(user))
        prefixes.append(history)
        targets.append(index)
        if len(users) >= n_users:
            break

    keep = [i for i, u in enumerate(users) if u is not None]
    return ([users[i] for i in keep], [prefixes[i] for i in keep],
            [targets[i] for i in keep])
