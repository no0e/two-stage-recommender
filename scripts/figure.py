"""Draw the results figure from docs/results.json.

    python scripts/figure.py

Separate from the evaluation so the figure can be redrawn without retraining,
and so the numbers on it can only come from a file that a run produced.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recsys.plotting import results_figure  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "docs" / "results.json"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "results.png"))
    args = parser.parse_args()

    results = Path(args.results)
    if not results.exists():
        raise SystemExit(
            f"{results} is missing. Run scripts/evaluate.py first; this script "
            "only draws what that one measured."
        )

    path = results_figure(json.loads(results.read_text(encoding="utf-8")),
                          args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
