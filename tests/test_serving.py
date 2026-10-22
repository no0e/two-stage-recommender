"""The serving path: export, parity with PyTorch, the API, the stores.

What would go wrong silently here is a service that ranks differently from
the model that was evaluated: an exported graph that drifts, a numpy
retrieval that forgets to mask what the user already bought, a store that
hands back another user's history. Each test pins one of those.

Needs onnx, onnxruntime, fastapi and httpx (requirements-dev.txt); skipped
without them. No download and no GPU.
"""
import asyncio

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from recsys.data import PAD  # noqa: E402
from recsys.large import rerank as torch_rerank  # noqa: E402
from recsys.large import retrieve as torch_retrieve  # noqa: E402
from recsys.models import BERT4Rec  # noqa: E402
from recsys.serving.app import Recommender, Request, create_app  # noqa: E402
from recsys.serving.export import write_artifacts  # noqa: E402
from recsys.serving.ranking import OnnxReranker, blend, retrieve  # noqa: E402
from recsys.serving.store import MemoryStore, load_histories  # noqa: E402

N_ITEMS, N_USERS, DIM, MAX_LEN = 60, 25, 16, 8


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = BERT4Rec(N_ITEMS, dimension=16, n_heads=2, n_layers=1,
                     max_length=MAX_LEN, dropout=0.0).eval()
    users = rng.normal(size=(N_USERS, DIM)).astype(np.float32)
    items = rng.normal(size=(N_ITEMS, DIM)).astype(np.float32)
    histories = {u: list(rng.choice(np.arange(1, N_ITEMS), size=3 + u % 7,
                                    replace=False)) for u in range(N_USERS)}
    histories[3] = []  # a user with an embedding and no history
    directory = tmp_path_factory.mktemp("artifacts")
    meta = write_artifacts(directory, model, users, items, histories,
                           popular=list(range(1, 31)), max_length=MAX_LEN,
                           beta=0.75, asins=[None] + [f"A{i:04d}" for i in
                                                      range(1, N_ITEMS)],
                           parity_users=N_USERS, n_candidates=20)
    return directory, model, users, items, histories, meta


def test_the_exported_graph_scores_like_pytorch(artifacts):
    _, _, _, _, _, meta = artifacts
    assert meta["parity"]["fp32"]["max_abs_diff"] < 1e-4
    assert meta["parity"]["fp32"]["top10_agreement"] == 1.0


def test_the_quantized_graph_is_smaller_and_still_finite(artifacts):
    directory, _, _, _, histories, meta = artifacts
    sizes = meta["size_mb"]
    assert sizes["reranker.int8.onnx"] <= sizes["reranker.onnx"]
    scores = OnnxReranker(directory / "reranker.int8.onnx", MAX_LEN).score(
        histories[5], [1, 2, 3, 4, 5])
    assert np.isfinite(scores).all()


def test_numpy_retrieval_matches_the_evaluation_retrieval(artifacts):
    _, _, users, items, histories, _ = artifacts
    for user in range(N_USERS):
        ours, _ = retrieve(users[user], items, histories[user], 15)
        theirs, _ = torch_retrieve(users, items, [user], [histories[user]], k=15)
        assert list(ours) == list(theirs[0])


def test_service_ranking_matches_the_evaluation_reranking(artifacts):
    directory, model, users, items, histories, _ = artifacts
    reranker = OnnxReranker(directory / "reranker.onnx", MAX_LEN)
    for user in (0, 5, 11):
        candidates, values = retrieve(users[user], items, histories[user], 20)
        ours = candidates[np.argsort(-blend(
            reranker.score(histories[user], candidates), values, 0.75))]
        theirs = torch_rerank(model, [histories[user]], candidates[None, :],
                              values[None, :], beta=0.75, max_length=MAX_LEN)
        assert list(ours) == list(theirs[0])


def test_the_api_never_recommends_padding_or_a_seen_item(artifacts):
    directory, _, _, _, histories, _ = artifacts
    with TestClient(create_app(directory)) as client:
        for user in range(N_USERS):
            body = client.post("/recommend",
                               json={"user_id": user, "k": 10}).json()
            got = [item["index"] for item in body["items"]]
            assert len(got) == 10 and len(set(got)) == 10
            assert PAD not in got
            assert not set(got) & set(int(i) for i in histories[user])
            assert body["source"] == "two-stage"


def test_an_unknown_user_gets_the_popularity_fallback(artifacts):
    directory, *_ = artifacts
    with TestClient(create_app(directory)) as client:
        body = client.post("/recommend",
                           json={"user_id": N_USERS + 5, "k": 4}).json()
    assert body["source"] == "popularity"
    assert [item["index"] for item in body["items"]] == [1, 2, 3, 4]


def test_beta_zero_serves_the_retrieval_order(artifacts):
    directory, _, users, items, histories, _ = artifacts
    with TestClient(create_app(directory)) as client:
        body = client.post("/recommend",
                           json={"user_id": 7, "k": 5, "beta": 0.0}).json()
    expected, _ = retrieve(users[7], items, histories[7], 5)
    assert [item["index"] for item in body["items"]] == list(expected)


def test_health_reports_what_is_loaded(artifacts):
    directory, *_ = artifacts
    with TestClient(create_app(directory)) as client:
        health = client.get("/health").json()
    assert health["store"] == "MemoryStore"
    assert health["items"] == N_ITEMS - 1


def test_histories_round_trip_through_the_flat_file(artifacts):
    directory, _, _, _, histories, _ = artifacts
    loaded = load_histories(directory / "histories.npz")
    for user, history in histories.items():
        assert list(loaded[user]) == [int(i) for i in history]


def test_redis_store_returns_each_users_own_features(artifacts):
    fakeredis = pytest.importorskip("fakeredis")
    from recsys.serving.store import RedisStore, fill_redis

    directory, _, users, _, histories, _ = artifacts

    async def scenario():
        client = fakeredis.FakeAsyncRedis()
        await fill_redis(client, users, histories, batch=7)
        store = RedisStore(client, DIM)
        recommender = Recommender(directory, store=store)
        for user in (0, 3, 9):
            history, vector = await store.get(user)
            assert list(history) == [int(i) for i in histories[user]]
            assert np.allclose(vector, users[user])
        assert (await store.get(N_USERS + 1)) == (None, None)
        response = await recommender.recommend(Request(user_id=9, k=5))
        await store.close()
        return response

    response = asyncio.run(scenario())
    assert response.source == "two-stage" and len(response.items) == 5


def test_memory_store_rejects_out_of_range_users(artifacts):
    directory, *_ = artifacts
    store = MemoryStore.from_artifacts(directory)
    assert asyncio.run(store.get(-1)) == (None, None)
    assert asyncio.run(store.get(N_USERS)) == (None, None)
