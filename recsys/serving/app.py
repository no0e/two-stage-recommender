"""FastAPI service: POST /recommend returns the two-stage top k for a user.

    ARTIFACTS_DIR=serving_artifacts/Video_Games \
        uvicorn --factory recsys.serving.app:create_app

Environment:
    ARTIFACTS_DIR   exported artifacts (see recsys/serving/__init__.py)
    REDIS_URL       if set, user features come from Redis; else from memory
    RERANKER        reranker.int8.onnx (default) or reranker.onnx
    ORT_THREADS     intra-op threads per worker for ONNX Runtime (default 1)

Scale with uvicorn workers, not threads. One request is a 50-step sequence
and 100 candidates, far too small to parallelise inside; ONNX Runtime's
default of one thread per core, times several workers, oversubscribes the
machine. Measured on 72 cores: 9.6 req/s with the default, 2,276 with one
thread per worker and 8 workers.

The handler is async so a slow Redis call never blocks the event loop. The
ONNX call is CPU-bound and synchronous, so it runs in the thread pool, where
ONNX Runtime releases the GIL and requests overlap.
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .ranking import OnnxReranker, blend, retrieve
from .store import MemoryStore, RedisStore


class Request(BaseModel):
    user_id: int = Field(ge=0)
    k: int = Field(default=10, ge=1, le=100)
    candidates: int = Field(default=100, ge=1, le=500)
    beta: float | None = Field(default=None, ge=0.0, le=1.0)


class Item(BaseModel):
    index: int
    asin: str | None
    score: float


class Response(BaseModel):
    user_id: int
    source: str
    items: list[Item]
    latency_ms: float


class Recommender:
    """Everything loaded once at startup and shared by every request."""

    def __init__(self, directory, store=None, reranker_file=None, threads=None):
        directory = Path(directory)
        self.meta = json.loads((directory / "meta.json").read_text())
        self.item_vectors = np.load(directory / "items.npy").astype(np.float32)
        self.popular = np.load(directory / "popular.npy")
        asins_path = directory / "asins.json"
        self.asins = (json.loads(asins_path.read_text())
                      if asins_path.exists() else None)
        reranker_file = reranker_file or self.meta.get(
            "default_reranker", "reranker.onnx")
        self.reranker = OnnxReranker(
            directory / reranker_file, self.meta["max_length"], threads)
        self.reranker_file = reranker_file
        self.store = store or MemoryStore.from_artifacts(directory)

    def asin(self, index):
        return self.asins[index] if self.asins else None

    async def recommend(self, request):
        started = time.perf_counter()
        beta = self.meta["beta"] if request.beta is None else request.beta
        history, vector = await self.store.get(request.user_id)

        if vector is None:
            # A user the graph never saw has no embedding to retrieve with.
            ranked = [int(i) for i in self.popular[:request.k]]
            items = [Item(index=i, asin=self.asin(i), score=0.0) for i in ranked]
            source = "popularity"
        else:
            candidates, retrieval_scores = retrieve(
                vector, self.item_vectors, history, request.candidates)
            if beta > 0:
                sequence_scores = await run_in_threadpool(
                    self.reranker.score, history, candidates)
                combined = blend(sequence_scores, retrieval_scores, beta)
            else:
                combined = retrieval_scores
            order = np.argsort(-combined)[:request.k]
            items = [Item(index=int(candidates[i]),
                          asin=self.asin(int(candidates[i])),
                          score=float(combined[i])) for i in order]
            source = "two-stage" if beta > 0 else "retrieval"

        return Response(user_id=request.user_id, source=source, items=items,
                        latency_ms=(time.perf_counter() - started) * 1e3)


def build_store(directory):
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    dimension = int(np.load(Path(directory) / "items.npy", mmap_mode="r")
                    .shape[1])
    return RedisStore.from_url(url, dimension)


def create_app(directory=None, store=None, reranker_file=None, threads=None):
    directory = directory or os.environ.get("ARTIFACTS_DIR", "artifacts")

    @asynccontextmanager
    async def lifespan(app):
        app.state.recommender = Recommender(
            directory, store or build_store(directory),
            reranker_file or os.environ.get("RERANKER"),
            threads or int(os.environ.get("ORT_THREADS", "1")))
        yield
        await app.state.recommender.store.close()

    app = FastAPI(title="Two-stage recommender", lifespan=lifespan)

    @app.get("/health")
    async def health():
        recommender = app.state.recommender
        return {"status": "ok",
                "store": type(recommender.store).__name__,
                "reranker": recommender.reranker_file,
                "items": int(recommender.item_vectors.shape[0] - 1)}

    @app.post("/recommend", response_model=Response)
    async def recommend(request: Request):
        try:
            return await app.state.recommender.recommend(request)
        except ValueError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    return app

