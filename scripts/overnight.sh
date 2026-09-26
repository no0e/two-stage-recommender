#!/usr/bin/env bash
# The two Amazon runs, in the order that matters if only one of them finishes.
#
#     bash scripts/overnight.sh
#
# Video_Games first, because it is the smaller of the two: 25,612 items
# against 162,035, and about a quarter of the interactions. If only one of the
# two finishes, that is the one worth having.
#
# Both write docs/amazon_<category>.json, and the second run redraws
# docs/results.png from the pair. Logs land beside them.
set -u

cd "$(dirname "$0")/.."
mkdir -p logs docs

echo "=== $(date -Is) Video_Games ==="
python -u scripts/evaluate.py \
    --category Video_Games \
    --lightgcn-epochs 400 \
    --bert-epochs 10 \
    --bert-batch 1024 \
    --workers 12 \
    > logs/video_games.log 2>&1
echo "Video_Games exited $?"

echo "=== $(date -Is) Toys_and_Games ==="
python -u scripts/evaluate.py \
    --category Toys_and_Games \
    --lightgcn-epochs 600 \
    --bert-epochs 8 \
    --bert-batch 1024 \
    --workers 12 \
    > logs/toys_and_games.log 2>&1
echo "Toys_and_Games exited $?"

echo "=== $(date -Is) done ==="
ls -la docs/amazon_*.json 2>/dev/null
