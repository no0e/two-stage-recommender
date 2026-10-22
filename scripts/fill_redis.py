"""Load every user's history and embedding from the artifacts into Redis.

    python scripts/fill_redis.py serving_artifacts/Toys_and_Games \
        --url redis://localhost:6379/0
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from recsys.serving.store import fill_redis, load_histories  # noqa: E402


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts")
    parser.add_argument("--url", default="redis://localhost:6379/0")
    args = parser.parse_args()

    import redis.asyncio as redis

    directory = Path(args.artifacts)
    users = np.load(directory / "users.npy")
    histories = load_histories(directory / "histories.npz")
    client = redis.from_url(args.url)
    started = time.time()
    written = await fill_redis(client, users, histories)
    info = await client.info("memory")
    await client.aclose()
    print(f"{written:,} users written in {time.time() - started:.0f}s, "
          f"Redis now uses {info['used_memory_human']}")


if __name__ == "__main__":
    asyncio.run(main())
