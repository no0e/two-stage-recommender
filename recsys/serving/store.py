"""Where the per-user features live at request time: history and embedding.

Two stores behind one async interface. `MemoryStore` reads the exported
artifacts into the process, which is what a single container on Cloud Run
uses. `RedisStore` keeps them in Redis, which is what lets several API
replicas share one copy and lets a user's history be updated without
redeploying anything.

Both hold the history as packed int32 and the embedding as packed float32:
one round trip per request fetches both, and no JSON is parsed on the hot path.
"""
import numpy as np

HISTORY_KEY = "hist:{}"
VECTOR_KEY = "uvec:{}"


def load_histories(path):
    """{user: int32 array} from the flat array + offsets in histories.npz."""
    stored = np.load(path)
    flat, offsets = stored["flat"], stored["offsets"]
    return {
        user: flat[offsets[user]:offsets[user + 1]]
        for user in range(len(offsets) - 1)
    }


class MemoryStore:
    """Features held in the API process itself."""

    def __init__(self, user_vectors, histories):
        self.user_vectors = np.asarray(user_vectors, dtype=np.float32)
        self.histories = histories

    @classmethod
    def from_artifacts(cls, directory):
        from pathlib import Path

        directory = Path(directory)
        return cls(np.load(directory / "users.npy"),
                   load_histories(directory / "histories.npz"))

    async def get(self, user):
        """(history, vector), or (None, None) for a user it has never seen."""
        if user < 0 or user >= len(self.user_vectors):
            return None, None
        return self.histories.get(user, np.zeros(0, np.int32)), \
            self.user_vectors[user]

    async def close(self):
        return None


class RedisStore:
    """Features in Redis, fetched with one pipelined round trip per request."""

    def __init__(self, client, dimension):
        self.client = client
        self.dimension = dimension

    @classmethod
    def from_url(cls, url, dimension):
        import redis.asyncio as redis

        return cls(redis.from_url(url), dimension)

    async def get(self, user):
        async with self.client.pipeline(transaction=False) as pipe:
            pipe.get(HISTORY_KEY.format(user))
            pipe.get(VECTOR_KEY.format(user))
            history, vector = await pipe.execute()
        if vector is None:
            return None, None
        history = (np.frombuffer(history, dtype=np.int32)
                   if history else np.zeros(0, np.int32))
        vector = np.frombuffer(vector, dtype=np.float32)
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"user {user} has a {vector.shape[0]}-d embedding in Redis, "
                f"the item table is {self.dimension}-d: the store was loaded "
                "from a different export")
        return history, vector

    async def close(self):
        await self.client.aclose()


async def fill_redis(client, user_vectors, histories, batch=10_000):
    """Write every user's features to Redis. Returns the number written."""
    users = list(range(len(user_vectors)))
    for start in range(0, len(users), batch):
        async with client.pipeline(transaction=False) as pipe:
            for user in users[start:start + batch]:
                history = np.asarray(histories.get(user, ()), dtype=np.int32)
                pipe.set(HISTORY_KEY.format(user), history.tobytes())
                pipe.set(VECTOR_KEY.format(user),
                         np.asarray(user_vectors[user], np.float32).tobytes())
            await pipe.execute()
    return len(users)
