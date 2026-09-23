# Two-Stage Hybrid Recommender

[![tests](https://github.com/no0e/two-stage-recommender/actions/workflows/tests.yml/badge.svg)](https://github.com/no0e/two-stage-recommender/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

Predicting the next item a user will touch. Retrieve a hundred candidates
cheaply, rerank ten of them expensively, and measure every stage against
something free.

Built for an in-class Kaggle competition on a private Amazon dataset, which it
won, first of eighty. This is the rewrite: it runs on public MovieLens data,
four silent bugs from the competition notebooks are fixed and pinned by tests,
and every number below comes from one command.

<p align="center">
  <img src="docs/results.png" width="100%" alt="Four panels: recall@10 for five systems, each one's margin over the popularity baseline with bootstrap intervals, the reranking weight sweep, and recall against catalogue coverage">
</p>

## What it does

| | Recall@10 | NDCG@10 | Catalogue reached |
|---|---|---|---|
| Most popular | 0.079 | 0.039 | 5.4% |
| BERT4Rec alone | 0.125 | 0.064 | 73.1% |
| LightGCN alone | 0.132 | 0.069 | 40.0% |
| Recency EASE | 0.157 | 0.085 | 44.2% |
| **Two-stage** | **0.176** | **0.097** | 44.5% |

MovieLens 100k, leave-one-out: the last item of each of 943 users is held out
and everything before it trains. Every margin below is a paired bootstrap over
users, 2000 resamples.

**Two-stage beats popularity by +0.098** [+0.071, +0.125], and beats its own
best single stage by **+0.019** [+0.004, +0.034]. Both intervals clear zero, so
the shape of the system is doing work rather than the headline hiding a tie.

**Popularity is not a weak baseline.** It reaches 5% of the catalogue, returns
the same sixty films to everyone, and still gets 8% of users right. Any
recommender that cannot beat it has earned nothing, which is why it is the
first row of the table rather than a footnote.

## Three things worth knowing

**A closed-form model beats the graph network.** EASE solves for an item-item
weight matrix in a single matrix inverse: no epochs, no learning rate. It fits
in about a second and scores 0.157 against a tuned LightGCN's 0.132, a margin
of +0.026 [+0.002, +0.048] that clears zero. Six minutes of gradient descent
lost to one second of linear algebra, and knowing that is worth more than the
two points.

**Ordering is the thing EASE throws away, so it is given back.** EASE scores a
user by summing the columns of every item in their history, which treats that
history as an unordered bag. The task is to predict the *next* item, so the
ordering is what the task is built around. Weighting the profile so recent
items count more, with a single half-life, moved validation recall from 0.185
to 0.226. That is the largest single gain in the project and it costs one
parameter.

**Reranking helps only if it keeps the retrieval score.** The competition
version reordered the hundred candidates on the sequence model's opinion alone
and discarded the score that retrieved them. Sweeping that weight shows it is
the worst setting available, below even leaving the retrieval order untouched:
0.172 against 0.207. Blending the two, at a weight fitted on validation, gives
0.218 and is what the table above reports.

## The four bugs

None of these raised an error. Each returned a plausible number for a question
nobody asked, which is how they survived a whole competition.
`tests/test_bugs.py` pins all four.

**LightGCN's negatives were users.** Negatives were drawn from `[0, n_items)`
over a node index where items start at `n_users`, so every negative sample was
a user node. The ranking loss was pushing item embeddings away from randomly
chosen users.

**Predictions came out of padding.** Sequences were padded on the right and the
prediction read off the last position. For any sequence shorter than the
window, and most are, that position is padding. Padding now goes on the left,
and attention masks it, which it did not.

**Unknown items became a real film.** `item_id_to_idx.get(item_id, 0)` mapped
anything unseen to index 0, and index 0 was a real item. Index 0 is now
reserved for padding and can never be recommended.

**The reported metric was not the reported system.** The evaluation scored
BERT4Rec alone over the whole catalogue, not the two-stage pipeline the write-up
described.

The original notebooks are in `notebooks/`, unmodified, so the claims above can
be checked against them rather than taken on trust.

## Quick start

```bash
git clone https://github.com/no0e/two-stage-recommender.git
cd two-stage-recommender
pip install -r requirements.txt

python scripts/fetch_data.py     # MovieLens 100k, 5 MB
python scripts/evaluate.py       # trains everything, prints the table
python scripts/figure.py         # redraws docs/results.png from the results
```

Under four minutes on a modest GPU, about half an hour on a laptop CPU.

On Windows, set `OMP_NUM_THREADS=1` first. Some BLAS builds deadlock on matrix
inversions above roughly 800 by 800, which is smaller than the item-item matrix
this fits; on the machine it was developed on, a 1683 by 1683 inverse went from
never finishing to 0.7 seconds.

## How the measurement is kept honest

**The split is leave-one-out**, which is what the sequential recommendation
literature reports, so the numbers are comparable to published ones. A harsher
deployment-shaped split, where one timestamp cuts every user and their history
freezes there, is available with `--protocol temporal`. It scores far lower and
is in the code because it is the question a live system faces.

**Every weight is fitted one step further back.** The retrieval blend and the
reranking weight are chosen on a split that removes one *more* item per user,
using models refitted without those targets. Choosing them against the test
targets would reward the graph for memorising its own training edges, which is
exactly the answer such a sweep gives when you let it.

**Every comparison is paired.** The same held-out users, the same triples, a
bootstrap over users rather than over interactions. A difference of a point on
943 users is not a result and the intervals say so.

## Layout

```
recsys/
  data.py        loading, the two splits, and the padding-safe index
  linear.py      the closed-form model and its recency weighting
  graph.py       LightGCN over a sparse adjacency, no graph library
  sequential.py  BERT4Rec, left-padded and masked
  content.py     content features and the cold start path
  pipeline.py    the two stages joined, and the baselines
  metrics.py     recall, NDCG, coverage, paired bootstrap
scripts/         fetch_data, evaluate, figure
tests/           34 tests, most of them about the four bugs
notebooks/       the competition notebooks, as they were run
```

`torch_geometric` is not a dependency. LightGCN propagation is one normalised
sparse matrix multiply per layer, so a graph library bought a single line and
cost an install that frequently fails.

## Requirements

Python 3.10 or later, PyTorch, NumPy, pandas, scikit-learn, matplotlib.

```bash
pip install -r requirements.txt
pytest tests/ -q
```

## License

MIT for the code. See [LICENSE](LICENSE). MovieLens belongs to GroupLens and
carries its own terms.
