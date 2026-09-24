# Two-Stage Hybrid Recommender

[![tests](https://github.com/no0e/two-stage-recommender/actions/workflows/tests.yml/badge.svg)](https://github.com/no0e/two-stage-recommender/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

Predicting the next item a user will touch. Retrieve a hundred candidates
cheaply, rerank ten of them with a sequence model.

Built for an in-class Kaggle competition on a private Amazon dataset, where it
finished first of eighty. This is the rewrite: it runs on public MovieLens data
and the whole thing is three commands.

## Try it

```bash
pip install -r requirements.txt
python scripts/fetch_data.py     # MovieLens 100k, 5 MB
python scripts/recommend.py      # ten films for one user, about two seconds
```

```
User 12, last five of 50 films watched
      Dances with Wolves (1990)
      Ghost (1990)
      Kolya (1996)
      Sleepless in Seattle (1993)
      Raising Arizona (1987)

Retrieval stage fitted in 0.9s.

Top 10, retrieval only
   1  Princess Bride, The (1987)
   2  When Harry Met Sally... (1989)  <- the one they watched
   3  Gandhi (1982)
   4  Quiz Show (1994)
   ...

They actually watched: When Harry Met Sally... (1989)
It lands in the top ten for about 18% of users; recall@10 over all 943 is 0.176.
```

The last film of every user is held out, so the answer is known and printed
underneath. `--user` picks another one, `--rerank` adds the second stage.

## What it does

**Retrieval** narrows 1,682 films to 100 candidates. EASE solves for an
item-item weight matrix in a single matrix inverse — no epochs, no learning
rate, about a second — and scores a user by their history weighted so recent
films count for more. LightGCN over the interaction graph is the alternative,
and a TF-IDF content model covers users and items with no history.

**Reranking** orders those 100 with BERT4Rec, a small transformer over the
user's sequence, blended with the retrieval score rather than replacing it.

The two blend weights are fitted on a validation split taken one step further
back, using models that never saw a test target.

## Results

<p align="center">
  <img src="docs/results.png" width="100%" alt="Recall@10 for five systems, and recall against the share of the catalogue each one reaches">
</p>

| | Recall@10 | NDCG@10 | Catalogue reached |
|---|---|---|---|
| Most popular | 0.079 | 0.039 | 5.4% |
| BERT4Rec alone | 0.125 | 0.064 | 73.1% |
| LightGCN alone | 0.132 | 0.069 | 40.0% |
| Recency EASE | 0.157 | 0.085 | 44.2% |
| **Two-stage** | **0.176** | **0.097** | 44.5% |

MovieLens 100k, leave-one-out: the last film of each of 943 users is held out
and everything before it trains. Margins are a paired bootstrap over users,
2000 resamples — two-stage beats popularity by +0.098 [+0.071, +0.125] and its
own best single stage by +0.019 [+0.004, +0.034].

Reproduce with `python scripts/evaluate.py`, which retrains everything and
rewrites the table and the figure. About four minutes on a GPU, half an hour on
a laptop CPU.

## Layout

```
recsys/data.py       loading, the split, the padding-safe index
recsys/models.py     content, EASE, LightGCN, BERT4Rec
recsys/pipeline.py   the two stages joined, and the baselines
recsys/metrics.py    recall, NDCG, coverage, paired bootstrap
scripts/             fetch_data, recommend, evaluate
tests/               34 tests, no dataset or GPU needed
notebooks/           the competition notebook, as it was run
```

The rewrite fixes four bugs the competition notebook shipped with, none of
which raised an error: negatives sampled from user nodes instead of items,
predictions read off a padding position, unknown items mapped onto a real film,
and a reported metric that measured a different system from the one described.
`tests/test_recsys.py` pins all four.

`torch_geometric` is not a dependency — LightGCN propagation is one normalised
sparse matrix multiply per layer.

On Windows, set `OMP_NUM_THREADS=1` first: some BLAS builds deadlock on matrix
inversions larger than about 800 by 800, which the item-item matrix is.

## Licence

MIT, see [LICENSE](LICENSE). MovieLens belongs to GroupLens and carries its own
terms.
