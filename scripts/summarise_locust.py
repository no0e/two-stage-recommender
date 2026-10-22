"""Turn a headless Locust run's CSV into docs/serving_benchmark.json.

    python scripts/summarise_locust.py docs/locust_stats.csv \
        --note "Toys_and_Games, int8 reranker, Redis store, 4 uvicorn workers"
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats", help="the *_stats.csv Locust writes")
    parser.add_argument("--note", default="")
    parser.add_argument("--out", default=str(ROOT / "docs"
                                             / "serving_benchmark.json"))
    args = parser.parse_args()

    with open(args.stats) as handle:
        rows = {row["Name"]: row for row in csv.DictReader(handle)}
    row = rows.get("/recommend") or rows["Aggregated"]

    result = {
        "note": args.note,
        "requests": int(row["Request Count"]),
        "failures": int(row["Failure Count"]),
        "requests_per_sec": round(float(row["Requests/s"]), 1),
        "latency_ms": {
            "median": float(row["50%"]),
            "p95": float(row["95%"]),
            "p99": float(row["99%"]),
            "max": float(row["Max Response Time"]),
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
