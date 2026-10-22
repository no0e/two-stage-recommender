"""The two stages behind an HTTP endpoint.

Nothing in this subpackage imports torch. The reranker runs as an ONNX graph
under ONNX Runtime and retrieval is a numpy dot product, so the serving image
carries neither PyTorch nor a GPU runtime. `export.py` is the one module that
needs torch, and it runs once, offline, to produce the artifacts.

    artifacts/
        reranker.onnx          the BERT4Rec candidate scorer, fp32
        reranker.int8.onnx     the same graph, weights quantized to int8
        items.npy              LightGCN item embeddings, (n_items, d)
        users.npy              LightGCN user embeddings, (n_users, d)
        histories.npz          every user's history, flat array + offsets
        popular.npy            the cold-start fallback, item indices
        asins.json             item index -> ASIN
        meta.json              shapes, beta, export parity checks
"""
