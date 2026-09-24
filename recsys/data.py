"""Loading interactions, splitting them in time, and turning them into sequences.

The competition this was built for used a private Amazon dataset that is not
redistributable, so the public version runs on MovieLens, which has the same
shape: interactions with a timestamp, and items with a title and categories to
build content features from.

One decision here is load-bearing. Index 0 is reserved as padding and never
belongs to an item. The original version mapped unknown items to index 0 with
`item_id_to_idx.get(item_id, 0)`, which silently rewrote every unseen item into
whichever real item happened to sort first, and then trained on it.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PAD = 0
"""Reserved index. Real items start at 1, so a padded slot can never be
mistaken for an interaction."""

MIN_POSITIVE_RATING = 1
"""The rating at which an interaction counts as a positive.

One, meaning every interaction counts. The competition this system was built
for had implicit data with no ratings at all: a user touched an item or they
did not. Thresholding MovieLens at four stars adds a filter the original task
never had, and throws away 45% of the graph's edges while it does so. Raise it
with `--min-rating` to get the stricter version, which the sequential
recommendation literature often reports on."""


class Interactions:
    """Interactions, their split, and the index maps that go with them.

    Kept as one object because the maps and the split have to agree: an index
    built over the full catalogue and a split built over the training rows are
    the sort of pair that drifts apart silently.
    """

    def __init__(self, events, items, protocol="leave_one_out",
                 test_quantile=0.8, min_history=3):
        self.events = events.sort_values("timestamp").reset_index(drop=True)
        self.items = items
        self.protocol = protocol
        self.min_history = min_history

        if protocol == "leave_one_out":
            self.train, self.test, self.cutoff = self._leave_one_out()
        elif protocol == "temporal":
            self.train, self.test, self.cutoff = self._global_cut(test_quantile)
        else:
            raise ValueError(
                f"Unknown protocol {protocol!r}. Use 'leave_one_out' or "
                "'temporal'."
            )

        # The catalogue is every item that exists, not only the ones that were
        # interacted with: the cold start path has to be able to reach an item
        # nobody has touched, which is the whole point of having one.
        catalogue = sorted(set(items["item_id"]) | set(self.events["item_id"]))
        self.item_to_index = {item: i + 1 for i, item in enumerate(catalogue)}
        self.index_to_item = {i: item for item, i in self.item_to_index.items()}

        users = sorted(self.train["user_id"].unique())
        self.user_to_index = {user: i for i, user in enumerate(users)}

        self.n_items = len(catalogue) + 1  # +1 for PAD
        self.n_users = len(users)

    def _global_cut(self, quantile):
        """One timestamp splits everybody: the deployment-shaped question.

        Realistic, and hard on sequential models: a user's history is frozen at
        the cutoff and every one of their later interactions is predicted from
        that same frozen state. Kept because it is the question a deployed
        system faces, but it is not the protocol the literature reports.
        """
        cutoff = self.events["timestamp"].quantile(quantile)
        return (
            self.events[self.events["timestamp"] <= cutoff],
            self.events[self.events["timestamp"] > cutoff],
            cutoff,
        )

    def _leave_one_out(self):
        """The standard sequential protocol: the last item of each user tests.

        Everything before it trains. This is what BERT4Rec, SASRec and the rest
        report on, and it is the only way a number here is comparable to a
        published one. It is also the honest test of a sequential model, which
        exists to use a history that ends immediately before the thing it is
        predicting.

        Users with fewer than `min_history` + 1 interactions are dropped: a
        history of one item is not a sequence.
        """
        ordered = self.events.sort_values(["user_id", "timestamp"])
        position = ordered.groupby("user_id").cumcount(ascending=False)
        length = ordered.groupby("user_id")["item_id"].transform("size")

        long_enough = length > self.min_history
        test = ordered[long_enough & (position == 0)]
        train = ordered[~(long_enough & (position == 0))]
        return (
            train.sort_values("timestamp"),
            test.sort_values("timestamp"),
            None,
        )

    def histories(self, frame=None):
        """Each user's items in time order."""
        frame = self.train if frame is None else frame
        ordered = frame.sort_values(["user_id", "timestamp"])
        return ordered.groupby("user_id")["item_id"].apply(list).to_dict()

    def indexed_histories(self, frame=None):
        return {
            user: [self.item_to_index[i] for i in items
                   if i in self.item_to_index]
            for user, items in self.histories(frame).items()
        }

    def evaluation_pairs(self, min_history=1):
        """(user, history, target) triples for the held-out interactions.

        The history is everything that user did in the training part. Under
        leave-one-out that is their whole sequence bar the last item, so there
        is exactly one triple per user. Under the temporal protocol the history
        is frozen at the cutoff and a user contributes one triple per later
        interaction.

        Users with no training history are left out and belong to the cold
        start path; mixing them in would hide which of the two is failing.
        """
        train_histories = self.indexed_histories()
        pairs = []
        for user, items in self.test.sort_values("timestamp").groupby("user_id"):
            history = train_histories.get(user, [])
            if len(history) < min_history:
                continue
            for item in items["item_id"]:
                index = self.item_to_index.get(item)
                if index is not None:
                    pairs.append((user, history, index))
        return pairs

    def without_last(self, n=1):
        """A copy whose training set is shortened by `n` more items per user.

        This is how the validation split is taken under leave-one-out: drop one
        further item from every history and call that the target. Fitting a
        hyperparameter on the real test targets is the mistake this exists to
        avoid.
        """
        ordered = self.train.sort_values(["user_id", "timestamp"])
        position = ordered.groupby("user_id").cumcount(ascending=False)
        length = ordered.groupby("user_id")["item_id"].transform("size")
        keep = ~((length > self.min_history) & (position < n))
        return Interactions(
            ordered[keep], self.items, protocol=self.protocol,
            min_history=self.min_history,
        )

    def summary(self):
        return {
            "protocol": self.protocol,
            "events": len(self.events),
            "users": self.events["user_id"].nunique(),
            "items_in_catalogue": self.n_items - 1,
            "train_events": len(self.train),
            "test_events": len(self.test),
            "cold_start_users": len(
                set(self.test["user_id"]) - set(self.train["user_id"])
            ),
        }


def training_sequences(interactions, max_length=50):
    """Every prefix of every training history, paired with the next item.

    One interaction becomes many examples: a user with ten items gives nine
    (prefix, next) pairs. That is what makes a sequential model trainable on a
    dataset this size.
    """
    sequences = []
    for user, history in interactions.indexed_histories().items():
        history = history[-max_length:]
        for cut in range(1, len(history)):
            sequences.append((user, history[:cut], history[cut]))
    return sequences


def load_movielens(data_dir=None, min_rating=MIN_POSITIVE_RATING):
    """MovieLens 100k, as implicit feedback.

    Ratings at or above `min_rating` are kept. At the default of one that is
    every interaction, which is the shape the competition data had.
    """
    root = Path(data_dir or DATA_DIR) / "ml-100k"
    if not root.exists():
        raise FileNotFoundError(
            f"{root} is missing. Run scripts/fetch_data.py first."
        )

    events = pd.read_csv(
        root / "u.data", sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    events = events[events["rating"] >= min_rating].drop(columns="rating")

    genres = [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    items = pd.read_csv(
        root / "u.item", sep="|", encoding="latin-1", header=None,
        names=["item_id", "title", "released", "video", "url"] + genres,
    )
    items["genres"] = items[genres].apply(
        lambda row: " ".join(g for g, flag in zip(genres, row) if flag == 1),
        axis=1,
    )
    items["year"] = (
        items["released"].astype(str).str.extract(r"(\d{4})")[0]
        .fillna("").astype(str)
    )
    # The text the content model reads. Nothing here comes from the
    # interactions, which is what lets it score an item with no history.
    items["text"] = (
        items["title"].fillna("") + " " + items["genres"] + " " + items["year"]
    )
    return events, items[["item_id", "title", "genres", "year", "text"]]


def load(data_dir=None, protocol="leave_one_out", test_quantile=0.8,
         min_rating=MIN_POSITIVE_RATING):
    events, items = load_movielens(data_dir, min_rating=min_rating)
    return Interactions(
        events, items, protocol=protocol, test_quantile=test_quantile)
