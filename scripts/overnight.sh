#!/usr/bin/env bash
# The two Amazon runs, in the order that matters if only one of them finishes.
#
#     bash scripts/overnight.sh
#
# Video_Games first, because it is the smaller of the two and the only one
# where the closed-form model can be fitted at all: 25,612 items means a Gram
# matrix of 4.9 GiB, which is large but allocatable, so the graph and the
# closed form can be measured against each other on the same users.
#
# Toys_and_Games second. At 162,035 items the same matrix is 195.6 GiB and the
# inverse needs another, so only the graph runs. That is the whole point of
# the pair: the first run says how the two compare when there is a choice, the
# second says what happens when there is not.
#
# Both write docs/amazon_<category>.json. Logs land beside them.
set -u

cd "$(dirname "$0")/.."
mkdir -p logs docs

echo "=== $(date -Is) Video_Games ==="
python -u scripts/evaluate_amazon.py \
    --category Video_Games \
    --with-ease \
    --lightgcn-epochs 400 \
    --bert-epochs 10 \
    --bert-batch 1024 \
    --workers 12 \
    > logs/video_games.log 2>&1
echo "Video_Games exited $?"

echo "=== $(date -Is) Toys_and_Games ==="
python -u scripts/evaluate_amazon.py \
    --category Toys_and_Games \
    --lightgcn-epochs 600 \
    --bert-epochs 8 \
    --bert-batch 1024 \
    --workers 12 \
    > logs/toys_and_games.log 2>&1
echo "Toys_and_Games exited $?"

echo "=== $(date -Is) done ==="
ls -la docs/amazon_*.json 2>/dev/null
