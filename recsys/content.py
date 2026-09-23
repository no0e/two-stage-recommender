"""Content features, and the cold start path that depends on them.

This is the part of the system that can score an item nobody has ever touched,
and recommend to a user nobody has ever seen. Both stages above are built on the
interaction graph and are blind in exactly those two cases.

The text encoder is pluggable and defaults to TF-IDF. Sentence embeddings are
better and are available with `encoder="sbert"`, but the default has to install
and run anywhere, including in CI, and a heavyweight default that fails to
install is a repository nobody can reproduce.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import PAD


def tfidf_features(texts, max_features=4096):
    matrix = TfidfVectorizer(
        max_features=max_features, stop_words="english", sublinear_tf=True,
    ).fit_transform(texts)
    return normalize(np.asarray(matrix.todense(), dtype=np.float32))


def sbert_features(texts, model_name="all-MiniLM-L6-v2"):
    """Sentence embeddings. Optional: the import is deliberately local."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(
        list(texts), batch_size=256, show_progress_bar=False)
    return normalize(np.asarray(embeddings, dtype=np.float32))


def build_content_matrix(interactions, encoder="tfidf"):
    """One row per item index, aligned with `item_to_index`, PAD row at zero.

    Rows are L2 normalised, so a dot product between two of them is a cosine
    similarity and the scoring code never has to normalise again.
    """
    items = interactions.items.copy()
    items["index"] = items["item_id"].map(interactions.item_to_index)
    items = items[items["index"].notna()].sort_values("index")

    texts = items["text"].fillna("").tolist()
    features = (
        sbert_features(texts) if encoder == "sbert" else tfidf_features(texts)
    )

    matrix = np.zeros((interactions.n_items, features.shape[1]),
                      dtype=np.float32)
    matrix[items["index"].to_numpy(dtype=int)] = features
    return matrix  # row PAD stays zero, so it scores zero against everything


def popularity(interactions, top=500):
    """Item indices by training frequency. The fallback of last resort."""
    counts = interactions.train["item_id"].value_counts()
    ranked = [
        interactions.item_to_index[item]
        for item in counts.index if item in interactions.item_to_index
    ]
    return ranked[:top]


class ContentRecommender:
    """Cold start: score by similarity to what the user has already liked.

    With no history at all there is nothing to be similar to, and the honest
    answer is the popular list rather than an arbitrary ranking dressed up as a
    personalised one.
    """

    def __init__(self, content_matrix, popular):
        self.content = content_matrix
        self.popular = popular

    def user_profile(self, history):
        """The mean of the content rows of what the user has seen."""
        history = [i for i in history if i != PAD]
        if not history:
            return None
        profile = self.content[history].mean(axis=0)
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else None

    def scores(self, history):
        profile = self.user_profile(history)
        if profile is None:
            scores = np.zeros(len(self.content), dtype=np.float32)
            # Descending so the most popular item keeps the highest score.
            for rank, index in enumerate(self.popular):
                scores[index] = 1.0 - rank / max(len(self.popular), 1)
            return scores
        return self.content @ profile

    def recommend(self, history, k=10):
        scores = self.scores(history)
        seen = set(history)
        order = np.argsort(-scores)
        out = []
        for index in order:
            if index == PAD or index in seen:
                continue
            out.append(int(index))
            if len(out) == k:
                break
        return out
